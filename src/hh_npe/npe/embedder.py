"""Small Transformer summary network for fixed-length household wave trajectories.

Input:  ``(batch, seq_len, n_features=3)`` — observation waves x (income,
        consumption, liquid_assets).
Output: ``(batch, output_dim=32)`` — fixed-dim embedding for the NPE
        density estimator to condition on.

Architecture::

    Input
      |- normalization                     # embedder owns normalization
      |- Linear(n_features -> d_model)     # + learnable positional embedding
      |- N x TransformerEncoderLayer(pre-LN, gelu)
      |- mean pool over sequence dim
      |- Linear(d_model -> output_dim)

Owning normalization lets us set sbi's ``z_score_x='none'`` and avoid sbi's
1D-flattening assumptions about 3D ``x``. Pre-LN attention is preferred for
small models where post-LN can be unstable in early training.

Two normalizations are available, and which one is right depends on whether the
features share a scale:

``LayerNorm`` (default)
    Normalizes *across features within each wave*. Fine when the features are
    comparable in magnitude, as in Phases 1-2 (income, consumption,
    liquid_assets, all positive and six-figure).

fixed per-feature standardization (pass ``feature_mean``/``feature_std``)
    Normalizes *each feature across the dataset*, so the relative scale between
    features is set once by the data rather than per timestep. Phase 3 needs
    this: ``liquid_assets`` has mean -$181 against ``income``'s $101k, and
    LayerNorm's per-wave rescaling squashes the small signed feature -- which
    is the credit-card borrowing that identifies ``beta``.

The statistics are buffers, so they persist through ``state_dict`` and travel
with the checkpoint; an ``x`` at inference is normalized exactly as in training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryTransformer(nn.Module):
    """Small Transformer encoder over a fixed-length sequence of feature vectors."""

    def __init__(
        self,
        n_features: int = 3,
        seq_len: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        output_dim: int = 32,
        dim_feedforward_mult: int = 4,
        dropout: float = 0.0,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
        per_sequence: bool = False,
        per_sequence_skip: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

        # Per-household (per-sequence) normalisation: subtract this
        # household's own mean across its waves and divide by its own sd.
        #
        # The point of it is that a proportional measurement bias cancels
        # exactly -- if c_obs = k * c_true with k constant within a household,
        # the normalised series is identical. That is the PSID consumption
        # problem in one line.
        #
        # The cost is that it removes LEVELS, and levels are what identifies
        # this model: Laibson et al.'s moments are %Visa, meanVisa, wealth|debt
        # and wealth|no debt, all level quantities. A household permanently at
        # -$20,000 liquid and one at -$200 normalise to the same series.
        #
        # `per_sequence_skip` names feature indices to leave on the global
        # scale. Age must normally be skipped: it advances by exactly
        # wave_years each wave, so per-sequence normalisation maps every
        # household to the identical ramp and the age channel -- worth ~1.9
        # nats (SIMULATOR_SPEC 6.1.1) -- carries nothing at all.
        self.per_sequence = per_sequence
        self.per_sequence_skip = tuple(per_sequence_skip)
        self.standardize = feature_mean is not None
        if self.standardize:
            if feature_std is None:
                raise ValueError("feature_mean and feature_std must both be given")
            mean = torch.as_tensor(feature_mean, dtype=torch.float32).reshape(-1)
            std = torch.as_tensor(feature_std, dtype=torch.float32).reshape(-1)
            if mean.numel() != n_features or std.numel() != n_features:
                raise ValueError(
                    f"feature_mean/feature_std must have {n_features} entries; got "
                    f"{mean.numel()} and {std.numel()}"
                )
            self.register_buffer("feature_mean", mean)
            # A feature that never varies would divide by zero; clamp rather than
            # fail, since a constant feature is uninformative, not fatal.
            self.register_buffer("feature_std", std.clamp_min(1e-8))
            self.input_norm = nn.Identity()
        else:
            self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, {self.seq_len}, {self.n_features}); "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[-2] != self.seq_len or x.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected (..., {self.seq_len}, {self.n_features}); "
                f"got {tuple(x.shape)}"
            )
        h = (x - self.feature_mean) / self.feature_std if self.standardize else x
        # getattr, not attribute access: posteriors are pickled whole, and
        # unpickling an embedder saved before `per_sequence` existed does not
        # call __init__, so the attribute is simply absent. Older checkpoints
        # must keep loading -- outputs/flow_fix predates this field and is the
        # arm the PSID results are built on.
        if getattr(self, "per_sequence", False):
            # dim=1 is the wave axis: statistics are per (household, feature).
            m = h.mean(dim=1, keepdim=True)
            # A feature that never moves within a household (no credit-card
            # debt in any wave, illiquid stuck at zero) has sd 0; clamping
            # sends it to 0 rather than to inf.
            sd = h.std(dim=1, keepdim=True).clamp_min(1e-6)
            hp = (h - m) / sd
            if getattr(self, "per_sequence_skip", ()):
                keep = torch.zeros(h.shape[-1], dtype=torch.bool, device=h.device)
                keep[list(self.per_sequence_skip)] = True
                hp = torch.where(keep, h, hp)
            h = hp
        h = self.input_norm(h)
        h = self.input_proj(h) + self.pos_emb
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.output_proj(h)
