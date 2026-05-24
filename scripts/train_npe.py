"""CLI: train the amortized NPE on the Phase 2 dataset.

Usage::

    uv run python scripts/train_npe.py
    uv run python scripts/train_npe.py npe.training.max_num_epochs=50 seed=1

Reads :file:`configs/config.yaml` (composing ``simulator/mvp``, ``npe/phase2``,
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

    embedder = TrajectoryTransformer(
        n_features=x.shape[-1],
        seq_len=npe_cfg.dataset.n_waves,
        d_model=npe_cfg.embedder.d_model,
        n_heads=npe_cfg.embedder.n_heads,
        n_layers=npe_cfg.embedder.n_layers,
        output_dim=npe_cfg.embedder.output_dim,
    )

    box = PriorBox(
        delta_low=cfg.simulator.prior.delta.low,
        delta_high=cfg.simulator.prior.delta.high,
        crra_low=cfg.simulator.prior.crra.low,
        crra_high=cfg.simulator.prior.crra.high,
    )

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
