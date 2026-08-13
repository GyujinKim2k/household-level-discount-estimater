"""Generate the NPE training dataset.

Sobol-samples the structural parameters, solves the lifecycle model for each
draw, simulates one household trajectory, aggregates to biennial observation
waves, drops households whose lineage was replaced by a newborn inside the
window, and saves ``(theta, x)`` to a ``.pt`` file.

Two simulators are available:

``twoasset`` (Phase 3, default)
    Port of Laibson et al.: credit cards, illiquid asset, naive quasi-hyperbolic
    discounting. Estimates ``(beta, delta, crra)``. Slow -- one solve is tens of
    seconds to ~10 minutes depending on ``--grid``.

``hark`` (Phases 1-2)
    Single liquid asset, no borrowing, ``beta`` locked at 1. Estimates
    ``(delta, crra)``. Kept so the Phase 2 results stay reproducible.

Usage::

    uv run python scripts/generate_dataset.py --n_samples 32 --n_jobs 4
    uv run python scripts/generate_dataset.py --simulator hark
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from hh_npe.data.dataset import save_dataset
from hh_npe.npe.prior import PHASE3, PriorBox, sample_sobol
from hh_npe.simulator.dispatch import SIMULATORS
from hh_npe.utils.seeding import seed_all

log = logging.getLogger("generate_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", choices=sorted(SIMULATORS), default="twoasset")
    parser.add_argument("--grid", choices=["coarse", "full"], default="coarse",
                        help="twoasset only: 'full' is Laibson et al.'s exact "
                             "190x84 grid, roughly 18x slower than 'coarse'.")
    parser.add_argument("--n_samples", type=int, default=8192,
                        help="Total Sobol draws. Power of 2 preferred.")
    parser.add_argument("--start_age", type=int, default=30,
                        help="First wave's first year. Default 30.")
    parser.add_argument("--n_waves", type=int, default=5)
    parser.add_argument("--wave_years", type=int, default=2,
                        help="Years per wave. 2 = biennial (PSID); 1 = annual.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="joblib parallelism; -1 uses all cores.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Defaults to data/processed/<simulator>_dataset.pt")
    parser.add_argument("--verbose", type=int, default=5,
                        help="joblib verbose level (0 silent, 10 every batch).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_all(args.seed)

    box = PHASE3 if args.simulator == "twoasset" else PriorBox()
    out = args.out or Path(f"data/processed/{args.simulator}_dataset.pt")
    fn = SIMULATORS[args.simulator]

    log.info(
        f"simulator={args.simulator} grid={args.grid} params={box.names} "
        f"waves={args.n_waves}x{args.wave_years}y from age {args.start_age}"
    )
    log.info(f"Sobol-sampling {args.n_samples} draws...")
    theta_np = sample_sobol(args.n_samples, box, seed=args.seed)

    log.info(f"Generating trajectories with n_jobs={args.n_jobs} ...")
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
        delayed(fn)(
            theta_np[i], args.seed + i + 1,
            args.start_age, args.n_waves, args.wave_years, args.grid,
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

    save_dataset(theta, x, out)
    log.info(f"Saved (theta {tuple(theta.shape)}, x {tuple(x.shape)}) to {out}")


if __name__ == "__main__":
    main()
