"""Contour figure for a trained posterior, on held-out draws with known truth.

Standing requirement: every estimation result ships one of these. See
`hh_npe.evaluation.plots` for why.

Usage::

    uv run python scripts/plot_posteriors.py \
        --run_dirs outputs/flow_fix/w7_s{0,1,2,3,4} --waves 7 \
        --out outputs/figures/flowfix_w7.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hh_npe.data.windows import build_windowed
from hh_npe.evaluation.plots import contour_corner, posterior_samples
from hh_npe.npe.prior import PHASE3, sample_sobol
from hh_npe.npe.train import load_posterior

START_HIGH = {7: 46, 10: 40}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dirs", type=Path, nargs="+", required=True,
                   help="One or more runs; >1 is combined into an ensemble.")
    p.add_argument("--waves", type=int, default=7)
    p.add_argument("--shards", type=Path,
                   default=Path("data/processed/phase3_dataset_shards"))
    p.add_argument("--train_n", type=int, default=57344)
    p.add_argument("--n_total", type=int, default=65536)
    p.add_argument("--n_show", type=int, default=3,
                   help="Held-out households to overlay.")
    p.add_argument("--n_post", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    w = args.waves
    posts = [load_posterior(d / f"posterior_{w}w.pt")["posterior"]
             for d in args.run_dirs]
    if len(posts) > 1:
        from scripts.ensemble_eval import _ensemble
        post = _ensemble(posts)
        what = f"{len(posts)}-member ensemble"
    else:
        post, what = posts[0], "single model"

    # Held-out shards only: a figure drawn on training draws would flatter the
    # posterior exactly where it should be scrutinised.
    shard_files = sorted(args.shards.glob("shard_*.npz"))
    held = [f for f in shard_files if int(np.load(f)["lo"]) >= args.train_n]
    theta_all = sample_sobol(args.n_total, PHASE3, seed=0)
    th, x, _ = build_windowed(held, theta_all, k=1, n_waves=w, seed=999,
                              start_low=25, start_high=START_HIGH[w],
                              wave_years=2, with_age=True)

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(th), size=args.n_show, replace=False)
    series, truth = {}, {}
    for r, idx in enumerate(pick):
        t = th[idx].numpy()
        lab = (f"household {r + 1}:  "
               r"$\beta$=" f"{t[0]:.2f}  " r"$\delta$=" f"{t[1]:.3f}  "
               r"$\rho$=" f"{t[2]:.2f}")
        series[lab] = posterior_samples(post, x[idx], n=args.n_post)
        truth[lab] = t

    contour_corner(
        series, PHASE3, truth=truth, path=args.out,
        title=(f"Phase 3 posterior — {w} waves, {what}\n"
               f"held-out households, stars mark the true parameters"),
    )
    print(f"wrote {args.out}")
    for lab, s in series.items():
        m, sd = s.mean(0), s.std(0)
        print(f"  {lab}\n    posterior mean "
              f"beta {m[0]:.3f}+-{sd[0]:.3f}  "
              f"delta {m[1]:.4f}+-{sd[1]:.4f}  crra {m[2]:.3f}+-{sd[2]:.3f}")


if __name__ == "__main__":
    main()
