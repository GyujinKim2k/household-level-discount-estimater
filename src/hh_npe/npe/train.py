"""Amortized SNPE-C training for the household-level NPE.

Wraps sbi's ``SNPE_C`` with our :class:`TrajectoryTransformer` embedder and the
Sobol-uniform prior. Single-round training (no proposal adaptation) so the
trained posterior is amortized — usable on any new ``x`` without re-training,
which is what the headline contribution requires.

We pass ``z_score_x='none'`` because our embedder handles input normalization
via its leading ``LayerNorm``, avoiding sbi's flatten-first-then-z-score
assumption for 3D ``x``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from sbi.inference import SNPE_C
from sbi.neural_nets import posterior_nn

from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PriorBox, make_sbi_prior

log = logging.getLogger(__name__)


def train_npe(
    theta: torch.Tensor,
    x: torch.Tensor,
    *,
    embedder: TrajectoryTransformer | None = None,
    box: PriorBox | None = None,
    flow: str = "nsf",
    hidden_features: int = 50,
    num_transforms: int = 5,
    max_num_epochs: int = 200,
    stop_after_epochs: int = 20,
    learning_rate: float = 5e-4,
    batch_size: int = 256,
    validation_fraction: float = 0.1,
    device: str = "cpu",
    show_progress: bool = False,
):
    """Train SNPE-C on ``(theta, x)`` and return ``(posterior, density_estimator, inference)``."""
    if embedder is None:
        embedder = TrajectoryTransformer()
    if box is None:
        box = PriorBox()

    prior = make_sbi_prior(box)
    de_builder = posterior_nn(
        model=flow,
        embedding_net=embedder,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        z_score_x="none",
        z_score_theta="independent",
    )

    inference = SNPE_C(
        prior=prior,
        density_estimator=de_builder,
        device=device,
        show_progress_bars=show_progress,
    )
    inference.append_simulations(theta.float(), x.float())

    log.info(
        f"Training SNPE-C: n_samples={theta.shape[0]}, "
        f"x_shape={tuple(x.shape[1:])}, device={device}, "
        f"max_epochs={max_num_epochs}, batch_size={batch_size}"
    )
    density_estimator = inference.train(
        max_num_epochs=max_num_epochs,
        stop_after_epochs=stop_after_epochs,
        learning_rate=learning_rate,
        training_batch_size=batch_size,
        validation_fraction=validation_fraction,
    )
    posterior = inference.build_posterior(density_estimator)
    return posterior, density_estimator, inference


def save_posterior(
    posterior,
    embedder: TrajectoryTransformer,
    box: PriorBox,
    path: str | Path,
) -> None:
    """Persist the trained posterior + embedder state + prior box."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "posterior": posterior,
            "embedder_state_dict": embedder.state_dict(),
            "box": box,
        },
        p,
    )


def load_posterior(path: str | Path) -> dict:
    """Load a checkpoint saved by :func:`save_posterior` (returns the raw dict)."""
    return torch.load(Path(path), weights_only=False)
