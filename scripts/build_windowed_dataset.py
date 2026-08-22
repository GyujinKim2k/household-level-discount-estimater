"""Build a random-age windowed training set from the stored annual panels.

No GPU and no re-solving: shards carry the full ages-20-90 panel, so cutting
windows is an aggregation pass over data already on disk.

Usage::

    uv run python scripts/build_windowed_dataset.py
    uv run python scripts/build_windowed_dataset.py --k 1 --fixed_start 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hh_npe.data.windows import build_windowed, max_start_age, save_windowed
from hh_npe.npe.prior import PHASE3, sample_sobol

log = logging.getLogger("build_windowed_dataset")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", type=Path,
                   default=Path("data/processed/phase3_dataset_shards"))
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/phase3_windowed.pt"))
    p.add_argument("--n_total", type=int, default=65536,
                   help="Sobol draws the shards were generated from.")
    p.add_argument("--n_waves", type=int, default=10)
    p.add_argument("--wave_years", type=int, default=2)
    p.add_argument("--start_low", type=int, default=25)
    p.add_argument("--start_high", type=int, default=40,
                   help="Windows end at start+2*n_waves-1, so 40 ends at 59 -- "
                        "before retirement at 64, and short of the ages where "
                        "the mortality-free forward pass starts to matter.")
    p.add_argument("--k", type=int, default=8,
                   help="Windows per panel. The effective independent sample "
                        "stays the panel count; this teaches the age mapping.")
    p.add_argument("--fixed_start", type=int, default=None,
                   help="Reproduce the old fixed window at this age (k ignored).")
    p.add_argument("--no_age", action="store_true",
                   help="Omit the per-wave age channel.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    shard_files = sorted(args.shards.glob("shard_*.npz"))
    if not shard_files:
        raise SystemExit(f"no shards in {args.shards}")
    limit = max_start_age(args.n_waves, args.wave_years)
    log.info(f"{len(shard_files)} shards; latest usable start age is {limit}")

    theta_all = sample_sobol(args.n_total, PHASE3, seed=0)
    theta, x, panel_id = build_windowed(
        shard_files, theta_all,
        start_low=args.start_low, start_high=args.start_high, k=args.k,
        n_waves=args.n_waves, wave_years=args.wave_years, seed=args.seed,
        with_age=not args.no_age, fixed_start=args.fixed_start,
    )
    n_panels = len(panel_id.unique())
    meta = {
        "n_waves": args.n_waves, "wave_years": args.wave_years,
        "start_low": args.start_low, "start_high": args.start_high,
        "k": args.k, "fixed_start": args.fixed_start,
        "with_age": not args.no_age, "seed": args.seed,
        "n_panels": n_panels, "n_shards": len(shard_files),
    }
    save_windowed(theta, x, panel_id, meta, args.out)
    log.info(
        f"Saved {len(theta)} windows from {n_panels} panels "
        f"(x {tuple(x.shape)}) to {args.out}. Effective independent sample is "
        f"{n_panels} panels, not {len(theta)} rows."
    )


if __name__ == "__main__":
    main()
