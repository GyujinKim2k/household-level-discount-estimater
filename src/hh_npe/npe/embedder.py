"""Small Transformer summary network for 5-wave biennial household trajectories.

Input:  ``(batch, seq_len=5, n_features=3)`` — biennial waves x (income,
        consumption, liquid_assets).
Output: ``(batch, output_dim=32)`` — fixed-dim embedding for the NPE
        density estimator to condition on.

Architecture::

    Input
      |- LayerNorm(n_features)             # embedder owns normalization
      |- Linear(n_features -> d_model)     # + learnable positional embedding
      |- N x TransformerEncoderLayer(pre-LN, gelu)
      |- mean pool over sequence dim
      |- Linear(d_model -> output_dim)

The input ``LayerNorm`` lets us set sbi's ``z_score_x='none'`` and avoid sbi's
1D-flattening assumptions about 3D ``x``. Pre-LN attention is preferred for
small models where post-LN can be unstable in early training.
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
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

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
        h = self.input_norm(x)
        h = self.input_proj(h) + self.pos_emb
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.output_proj(h)
