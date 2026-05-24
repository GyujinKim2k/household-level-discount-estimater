"""Generate the Phase 2 training dataset.

Sobol-samples ``(delta, crra)``, solves the lifecycle for each draw, simulates
one household trajectory, biennial-aggregates to 5 waves starting at age 30,
filters out households whose lineage was replaced by a newborn within the
window, and saves the resulting ``(theta, x)`` to a ``.pt`` file.

Usage::

    uv run python scripts/generate_dataset.py --n_samples 32 --n_jobs 4

For the planned Phase 2 pilot of 8,192 samples on CPU, expect 4-8 hours
depending on core count.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from hh_npe.data.biennial import aggregate_biennial
from hh_npe.data.dataset import save_dataset
from hh_npe.npe.prior import PriorBox, sample_sobol
from hh_npe.simulator.forward import simulate_households
from hh_npe.simulator.lifecycle import build_lifecycle_agent, solve_lifecycle
from hh_npe.utils.seeding import seed_all

AGE_START_SIM = 20
AGE_END_SIM = 90

log = logging.getLogger("generate_dataset")


def simulate_one(
    delta: float,
    crra: float,
    sim_seed: int,
    start_age: int,
    n_waves: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve + simulate + biennial-aggregate one household."""
    agent = build_lifecycle_agent(
        delta=delta,
        crra=crra,
        age_start=AGE_START_SIM,
        age_end=AGE_END_SIM,
        n_agents=1,
    )
    solve_lifecycle(agent)
    panel = simulate_households(agent, n_households=1, seed=sim_seed)
    x, alive = aggregate_biennial(
        panel,
        age_start_sim=AGE_START_SIM,
        start_age=start_age,
        n_waves=n_waves,
    )
    return x[0], alive[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_samples", type=int, default=8192,
                        help="Total Sobol draws of (delta, crra). Power of 2 preferred.")
    parser.add_argument("--start_age", type=int, default=30,
                        help="First wave's first year. Default 30.")
    parser.add_argument("--n_waves", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="joblib parallelism; -1 uses all cores.")
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/phase2_dataset.pt"))
    parser.add_argument("--verbose", type=int, default=5,
                        help="joblib verbose level (0 silent, 10 every batch).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_all(args.seed)

    log.info(f"Sobol-sampling {args.n_samples} (delta, crra) from PriorBox()...")
    theta_np = sample_sobol(args.n_samples, PriorBox(), seed=args.seed)

    log.info(f"Generating trajectories with n_jobs={args.n_jobs} ...")
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
        delayed(simulate_one)(
            float(theta_np[i, 0]),
            float(theta_np[i, 1]),
            args.seed + i + 1,
            args.start_age,
            args.n_waves,
        )
        for i in range(args.n_samples)
    )
    elapsed = time.time() - t0
    log.info(
        f"Generated {args.n_samples} samples in {elapsed:.1f}s "
        f"({elapsed / args.n_samples * 1000:.1f} ms/sample)"
    )

    x_np = np.stack([r[0] for r in results])
    alive_np = np.stack([r[1] for r in results])
    fully_alive = alive_np.all(axis=1)
    n_alive = int(fully_alive.sum())
    log.info(
        f"Survival filter: {n_alive}/{args.n_samples} households "
        f"({100 * n_alive / args.n_samples:.1f}%) fully alive across all waves."
    )

    theta = torch.from_numpy(theta_np[fully_alive]).float()
    x = torch.from_numpy(x_np[fully_alive]).float()

    save_dataset(theta, x, args.out)
    log.info(f"Saved (theta {tuple(theta.shape)}, x {tuple(x.shape)}) to {args.out}")


if __name__ == "__main__":
    main()
