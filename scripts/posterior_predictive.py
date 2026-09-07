"""Can the model reproduce PSID behaviour at the parameters it inferred?

The PSID posteriors are pinned against the prior bounds -- 92% of households
have a delta interval running into 1.0, 71% a beta interval running into 1.0 --
which is the signature of misspecification rather than estimation. This asks the
question that separates the two: simulate the model AT the inferred parameters
and see whether the simulated behaviour matches the data. If it does, the
parameters are simply extreme. If it does not, no parameter value in the box can
reproduce PSID and the model itself is the problem.

Two comparisons, deliberately kept apart because they answer different things
and are routinely conflated:

**Representative-agent (f(E[theta])).** Solve the model ONCE at a single
representative parameter -- the median or mean posterior across households --
and compare that simulation to the representative household in the data. This
is what "the model at the estimated parameters" means.

**Posterior-predictive (E[f(theta)]).** Solve at each household's OWN inferred
parameters and pool. This is what the fitted model predicts the population looks
like.

These are not the same object and need not agree: the model is nonlinear in
theta, so the simulation at the average parameter is not the average simulation.
Reporting one while describing the other is a common way to make a
misspecified model look adequate.

Usage::

    uv run python scripts/posterior_predictive.py --n_draws 192
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hh_npe.data.dataset import read_solver_config
from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu

LAIBSON = np.array([0.5305, 0.9891, 1.9355])
AGE_START_SIM = 20


def slope(y: np.ndarray, c: np.ndarray, n: int = 3) -> float:
    """Cross-sectional consumption-income slope across income terciles."""
    m = (y > 1000) & (c > 0)
    y, c = y[m], c[m]
    t = pd.qcut(y, n, labels=False)
    ys = [np.median(y[t == k]) for k in range(n)]
    cs = [np.median(c[t == k]) for k in range(n)]
    return (cs[-1] - cs[0]) / (ys[-1] - ys[0])


def sim(thetas: np.ndarray, cfg: dict, seed: int = 7):
    _x, _a, panels = simulate_batch_twoasset_gpu(
        thetas, seed, 30, n_waves=5, wave_years=2,
        grid=cfg.get("grid", "full"), theta_batch=cfg["theta_batch"],
        chunk=cfg["chunk"], return_panels=True,
    )
    return panels


def band(panels: dict, lo_age: int, hi_age: int):
    """Pool annual observations over an age band."""
    t0, t1 = lo_age - AGE_START_SIM, hi_age - AGE_START_SIM + 1
    return (panels["income"][:, t0:t1].ravel(),
            panels["consumption"][:, t0:t1].ravel(),
            panels["liquid_assets"][:, t0:t1].ravel(),
            panels["illiquid_assets"][:, t0:t1].ravel())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--posterior", type=Path,
                    default=Path("outputs/psid/posterior_uncorrected.npz"))
    ap.add_argument("--x", type=Path, default=Path("data/processed/psid_x.pt"))
    ap.add_argument("--dataset", type=Path,
                    default=Path("data/processed/phase3_dataset.pt"))
    ap.add_argument("--n_draws", type=int, default=192,
                    help="Households drawn for the posterior-predictive arm. "
                         "Each is a full backward induction, ~12.4 s.")
    ap.add_argument("--out", type=Path, default=Path("outputs/psid"))
    args = ap.parse_args()

    cfg = read_solver_config(args.dataset)
    if not cfg or cfg.get("device") != "cuda":
        raise SystemExit("no CUDA solver config beside the dataset")

    z = np.load(args.posterior)
    m = z["mean"]
    ok = np.isfinite(m[:, 0])
    m = m[ok]
    d = torch.load(args.x, weights_only=False)
    xr = d["x"].numpy()[ok]

    theta_med = np.median(m, axis=0)
    theta_mean = m.mean(axis=0)
    print(f"representative theta   median {theta_med.round(4)}")
    print(f"                       mean   {theta_mean.round(4)}")
    print(f"Laibson et al.                {LAIBSON.round(4)}")

    # PSID target, ages 35-44 -- the band used for the earlier slope estimate.
    a = xr[:, :, 4].ravel()
    sel = (a >= 35) & (a <= 44)
    py, pc = xr[:, :, 0].ravel()[sel], xr[:, :, 1].ravel()[sel]
    pl, pi = xr[:, :, 2].ravel()[sel], xr[:, :, 3].ravel()[sel]
    print(f"\nPSID (ages 35-44, {sel.sum()} household-waves)")
    print(f"  consumption-income slope {slope(py, pc):.3f}")
    print(f"  median  income {np.median(py):8.0f}  cons {np.median(pc):8.0f}"
          f"  liquid {np.median(pl):9.0f}  illiquid {np.median(pi):10.0f}")

    # --- representative agent: solve ONCE at each single theta -------------
    reps = {"median posterior": theta_med, "mean posterior": theta_mean,
            "Laibson MSM": LAIBSON}
    print("\n=== representative agent  f(E[theta]) ===")
    print(f"{'theta':20s}{'slope':>8s}{'income':>10s}{'cons':>10s}"
          f"{'liquid':>11s}{'illiquid':>12s}")
    print(f"{'PSID (data)':20s}{slope(py, pc):8.3f}{np.median(py):10.0f}"
          f"{np.median(pc):10.0f}{np.median(pl):11.0f}{np.median(pi):12.0f}")
    for name, th in reps.items():
        # Many households, ONE parameter: the population the model predicts if
        # every household shared the representative preference.
        thetas = np.tile(th, (64, 1))
        p = sim(thetas, cfg)
        y, c, l, i = band(p, 35, 44)
        print(f"{name:20s}{slope(y, c):8.3f}{np.median(y):10.0f}"
              f"{np.median(c):10.0f}{np.median(l):11.0f}{np.median(i):12.0f}")

    # --- posterior predictive: each household at its OWN theta -------------
    rng = np.random.default_rng(0)
    pick = rng.choice(len(m), size=min(args.n_draws, len(m)), replace=False)
    print(f"\n=== posterior predictive  E[f(theta)]  ({len(pick)} households) ===")
    p = sim(m[pick].astype(np.float64), cfg)
    y, c, l, i = band(p, 35, 44)
    print(f"{'pooled over theta':20s}{slope(y, c):8.3f}{np.median(y):10.0f}"
          f"{np.median(c):10.0f}{np.median(l):11.0f}{np.median(i):12.0f}")

    np.savez(args.out / "posterior_predictive.npz",
             theta_med=theta_med, theta_mean=theta_mean,
             psid_slope=slope(py, pc), pp_slope=slope(y, c),
             pp_income=y, pp_cons=c, pp_liquid=l, pp_illiquid=i)
    print(f"\nwrote {args.out}/posterior_predictive.npz")
    print("\nIf the representative-agent slope sits near PSID's but the pooled")
    print("one does not (or vice versa), the two summaries disagree and only")
    print("one of them is the quantity being claimed.")


if __name__ == "__main__":
    main()
