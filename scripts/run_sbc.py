"""CLI: run SBC on a trained NPE.

Loads ``posterior.pt`` from the Hydra-resolved output dir, draws fresh
``thetas`` from the prior, simulates ``xs`` end-to-end through the lifecycle
simulator, and computes rank-distribution diagnostics (Talts et al. 2018).

Usage::

    uv run python scripts/run_sbc.py output_dir=outputs/<run>
    uv run python scripts/run_sbc.py eval.n_simulations=200
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig

from hh_npe.evaluation.sbc import compute_ranks, coverage_at_level, plot_sbc_ranks
from hh_npe.npe.prior import make_sbi_prior
from hh_npe.npe.train import load_posterior
from hh_npe.data.dataset import read_solver_config
from hh_npe.simulator.dispatch import SIMULATORS, simulate_batch_twoasset_gpu
from hh_npe.utils.seeding import seed_all


def simulate_for_sbc(
    thetas: torch.Tensor,
    start_age: int,
    n_waves: int,
    wave_years: int,
    seed_base: int,
    simulator: str = "twoasset",
    grid: str = "coarse",
    n_jobs: int = -1,
    solver_config: dict | None = None,
) -> torch.Tensor:
    """Simulate SBC draws through the *same* simulator that made the training set.

    "Same" has to be taken literally for the two-asset model. Its grids are built
    from round dollar amounts, so ~99% of states hold two ``(X', Z')`` choices
    leaving bitwise-identical cash; where their continuation values tie too, CPU
    and GPU pick different -- equally optimal -- portfolios. Both reproduce
    Laibson et al.'s table 3 (fidelity 0.0102 vs 0.0104), but they are not the
    same simulator: across six draws the choice moved mean liquid assets by 25%.
    Running SBC on the CPU against a GPU-generated training set would therefore
    show miscalibration that belongs to the mismatch, not to the posterior.

    So when the training set records a CUDA configuration, reproduce it exactly
    -- device, theta_batch and chunk -- rather than defaulting to the CPU path.
    """
    if solver_config and solver_config.get("device") == "cuda":
        x, _alive = simulate_batch_twoasset_gpu(
            thetas.numpy(), seed_base, start_age, n_waves, wave_years,
            grid=solver_config.get("grid", grid),
            theta_batch=solver_config["theta_batch"],
            chunk=solver_config["chunk"],
        )
        return torch.from_numpy(x).float()

    fn = SIMULATORS[simulator]
    n = thetas.shape[0]
    out = Parallel(n_jobs=n_jobs)(
        delayed(fn)(
            thetas[i].numpy(), seed_base + i, start_age, n_waves, wave_years, grid,
        )
        for i in range(n)
    )
    return torch.from_numpy(np.stack([o[0] for o in out])).float()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("run_sbc")

    seed_all(cfg.seed + 1000)  # different seed from training/generation

    posterior_path = Path(cfg.output_dir) / "posterior.pt"
    loaded = load_posterior(posterior_path)
    posterior = loaded["posterior"]
    box = loaded["box"]
    prior = make_sbi_prior(box)

    log.info(f"Sampling {cfg.eval.n_simulations} thetas from prior...")
    thetas = prior.sample((cfg.eval.n_simulations,))

    solver_config = read_solver_config(cfg.npe.dataset.path)
    if solver_config:
        log.info(f"Training set solver config: {solver_config}")
    else:
        log.warning(
            "No solver_config.json beside the training set, so SBC cannot verify "
            "it is using the same simulator. For a CUDA-generated two-asset "
            "dataset the CPU path is NOT equivalent -- see simulate_for_sbc."
        )

    log.info("Simulating xs for SBC (full lifecycle solve per theta)...")
    xs = simulate_for_sbc(
        thetas,
        start_age=cfg.npe.dataset.start_age,
        n_waves=cfg.npe.dataset.n_waves,
        wave_years=cfg.npe.dataset.wave_years,
        seed_base=cfg.seed + 2000,
        simulator=cfg.npe.get("simulator", "twoasset"),
        grid=cfg.npe.get("grid", "coarse"),
        solver_config=solver_config,
    )

    log.info(f"Computing ranks with {cfg.eval.n_posterior_samples} posterior samples per point...")
    ranks = compute_ranks(
        posterior, thetas, xs, n_posterior_samples=cfg.eval.n_posterior_samples
    )

    cov = coverage_at_level(ranks, cfg.eval.n_posterior_samples, level=0.9)
    log.info(
        "90%% CI empirical coverage: "
        + ", ".join(f"{n}={c:.3f}" for n, c in zip(box.names, cov))
        + " (target 0.9)"
    )

    plot_path = Path(cfg.output_dir) / "sbc_ranks.png"
    plot_sbc_ranks(
        ranks, cfg.eval.n_posterior_samples, list(box.names), plot_path
    )
    log.info(f"SBC rank plot saved to {plot_path}")

    # Save raw ranks alongside the plot for downstream analysis.
    ranks_path = Path(cfg.output_dir) / "sbc_ranks.npz"
    np.savez(ranks_path, ranks=ranks, coverage_90=cov,
             n_posterior_samples=cfg.eval.n_posterior_samples)
    log.info(f"Ranks saved to {ranks_path}")


if __name__ == "__main__":
    main()
