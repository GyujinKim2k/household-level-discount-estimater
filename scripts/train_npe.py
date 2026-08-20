"""CLI: train the amortized NPE on the Phase 2 dataset.

Usage::

    uv run python scripts/train_npe.py
    uv run python scripts/train_npe.py npe.training.max_num_epochs=50 seed=1

Reads :file:`configs/config.yaml` (composing ``simulator/mvp``, ``npe/phase3``,
``eval/sbc``). Writes ``posterior.pt`` to the Hydra-resolved output dir.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from hh_npe.data.dataset import load_dataset
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PriorBox
from hh_npe.npe.train import save_posterior, train_npe
from hh_npe.utils.seeding import seed_all


def build_prior_box(cfg: DictConfig) -> PriorBox:
    """Prior bounds, from ``npe.prior`` (Phase 3) or ``simulator.prior`` (1-2).

    Phase 3 unlocks ``beta``, so its bounds live with the NPE config rather
    than the simulator config; ``beta`` is absent for Phases 1-2, which lock it
    at 1.0.
    """
    src = cfg.npe.get("prior") or cfg.simulator.prior
    kwargs = dict(
        delta_low=src.delta.low, delta_high=src.delta.high,
        crra_low=src.crra.low, crra_high=src.crra.high,
    )
    if "beta" in src and isinstance(src.beta, DictConfig):
        kwargs.update(beta_low=src.beta.low, beta_high=src.beta.high)
    return PriorBox(**kwargs)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("train_npe")
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    seed_all(cfg.seed)

    npe_cfg = cfg.npe
    theta, x = load_dataset(npe_cfg.dataset.path)
    log.info(f"Loaded dataset: theta={tuple(theta.shape)}, x={tuple(x.shape)}")

    # Per-feature statistics over the whole training set, so the embedder's
    # normalization is fixed by the data rather than recomputed per wave. See
    # the embedder module docstring for why Phase 3's feature scales need this.
    stats = {}
    if npe_cfg.embedder.get("standardize", False):
        stats = {"feature_mean": x.mean(dim=(0, 1)), "feature_std": x.std(dim=(0, 1))}
        log.info(
            "Standardizing features: mean=%s std=%s",
            stats["feature_mean"].tolist(), stats["feature_std"].tolist(),
        )

    embedder = TrajectoryTransformer(
        n_features=x.shape[-1],
        seq_len=npe_cfg.dataset.n_waves,
        d_model=npe_cfg.embedder.d_model,
        n_heads=npe_cfg.embedder.n_heads,
        n_layers=npe_cfg.embedder.n_layers,
        output_dim=npe_cfg.embedder.output_dim,
        **stats,
    )

    box = build_prior_box(cfg)
    log.info(f"Estimating {box.names} over {box.low} .. {box.high}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Training on device={device}")

    posterior, _, _ = train_npe(
        theta, x,
        embedder=embedder,
        box=box,
        flow=npe_cfg.training.flow,
        max_num_epochs=npe_cfg.training.max_num_epochs,
        stop_after_epochs=npe_cfg.training.stop_after_epochs,
        learning_rate=npe_cfg.training.learning_rate,
        batch_size=npe_cfg.training.batch_size,
        validation_fraction=npe_cfg.training.validation_fraction,
        device=device,
        show_progress=True,
    )

    out_path = Path(cfg.output_dir) / "posterior.pt"
    save_posterior(posterior, embedder, box, out_path)
    log.info(f"Done. Posterior saved to {out_path}")


if __name__ == "__main__":
    main()
