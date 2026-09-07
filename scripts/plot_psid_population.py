"""The project's headline figure: heterogeneity against a single point estimate.

Two things are overlaid because they answer different questions and are easily
confused:

**Between-household spread** -- the cloud of per-household posterior *means*.
This is the heterogeneity the project exists to measure, and it is what Laibson
et al.'s single MSM point cannot show.

**Within-household uncertainty** -- one household's full posterior. If this is
as wide as the cloud, the apparent heterogeneity is estimation noise rather
than real variation across households, and the contribution evaporates. Showing
them on the same axes is the only honest way to present the claim.

Laibson et al.'s benchmark is marked, so the reader can see both where the
population sits relative to it and how much of the gap is spread.

Usage::

    uv run python scripts/plot_psid_population.py --out outputs/psid_matched
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hh_npe.evaluation.plots import contour_corner
from hh_npe.evaluation.sbc import posterior_device
from hh_npe.npe.prior import PHASE3
from hh_npe.npe.train import load_posterior

LAIBSON = np.array([0.5305, 0.9891, 1.9355])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--posterior_npz", type=Path,
                    default=Path("outputs/psid_matched/posterior_uncorrected.npz"))
    ap.add_argument("--x", type=Path,
                    default=Path("data/processed/psid_x_matched_net.pt"))
    ap.add_argument("--run_dirs", type=Path, nargs="+",
                    default=[Path(f"outputs/flow_fix/w7_s{s}") for s in range(5)])
    ap.add_argument("--waves", type=int, default=7)
    ap.add_argument("--n_post", type=int, default=6000)
    ap.add_argument("--out", type=Path, default=Path("outputs/psid_matched"))
    args = ap.parse_args()

    z = np.load(args.posterior_npz)
    means = z["mean"]
    ok = np.isfinite(means[:, 0])
    means = means[ok]
    print(f"{len(means)} households with a summarisable posterior")

    from scripts.ensemble_eval import _ensemble
    posts = [load_posterior(d / f"posterior_{args.waves}w.pt")["posterior"]
             for d in args.run_dirs]
    post = _ensemble(posts)
    dev = posterior_device(post)
    x = torch.load(args.x, weights_only=False)["x"][ok].to(dev)

    # A median household by posterior mean, so the within-household contour is
    # typical rather than cherry-picked from a tail.
    d2 = np.abs((means - np.median(means, axis=0))
                / means.std(axis=0)).sum(axis=1)
    i = int(np.argmin(d2))
    with torch.no_grad():
        draws = torch.cat([m.posterior_estimator.sample((args.n_post,),
                                                        condition=x[i:i + 1])
                           for m in posts], dim=0).squeeze(1).cpu().numpy()
    lo, hi = np.asarray(PHASE3.low), np.asarray(PHASE3.high)
    draws = draws[((draws >= lo) & (draws <= hi)).all(axis=1)]
    print(f"representative household: index {i}, "
          f"posterior mean {means[i].round(3)}, {len(draws)} admissible draws")

    series = {
        f"between households  (posterior means, N={len(means)})": means,
        "within one household  (its full posterior)": draws,
    }
    contour_corner(
        series, PHASE3,
        truth={"Laibson et al. MSM": LAIBSON,
               "_": LAIBSON},          # same marker on both hues
        path=args.out / "psid_population.png",
        title=("PSID households, Laibson-matched sample — 7 waves, "
               "5-member ensemble\nstar = Laibson et al.'s single MSM estimate "
               "for the whole population"),
    )
    print(f"wrote {args.out}/psid_population.png")

    print(f"\n{'':10s}{'between-hh sd':>15s}{'within-hh sd':>15s}{'ratio':>8s}")
    for j, n in enumerate(PHASE3.names):
        b, w = means[:, j].std(), draws[:, j].std()
        print(f"{n:10s}{b:15.4f}{w:15.4f}{b / w:8.2f}")
    print("\nratio > 1 means households differ by more than any one of them is\n"
          "uncertain -- the heterogeneity is real. ratio < 1 means the spread\n"
          "is estimation noise and a single population estimate would do.")


if __name__ == "__main__":
    main()
