"""Validate the two-asset port against Laibson et al.'s table-3 moments.

Solves at their published benchmark estimates, simulates their population size,
recomputes their 16 target moments, and reports the MSM objective. Repeats at
several grid resolutions so the cost of coarsening for NPE throughput is
measured rather than assumed.

Their moment block is ``LifecycleSim.m:290-325``; the objective and weighting
matrix are ``MSMfunction.m:236`` and ``EDFbatch_baseline.m:96``.

Usage::

    uv run python scripts/validate_twoasset.py --pop 10000
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np

from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.twoasset import COARSE, ModelSpec, simulate, solve

# Their age bands, as 0-based indices into the age-20..90 panel.
# MATLAB ``ag = reshape(2:41, [10,4])'`` = ages 21-30, 31-40, 41-50, 51-60.
BANDS = [np.arange(1 + 10 * i, 11 + 10 * i) for i in range(4)]


def simulated_moments(panel: dict[str, np.ndarray]) -> np.ndarray:
    """Their 16 moments: [%Visa, meanVisa, wealth|debt, wealth|no debt] x 4 bands.

    ``simL = simX - simY`` is the liquid position before income arrives, so
    ``simL < 0`` is exactly "carries credit-card debt". ``simW = simZ + simL``
    is total wealth. Both are normalized by mean income at that age and
    averaged within a band using survival shares as weights.
    """
    liquid = panel["liquid_assets"]
    wealth = panel["illiquid_assets"] + liquid
    avg_y = panel["income"].mean(axis=0)
    alive = cal.survival_share()

    has_debt = liquid < 0
    frac_debt = has_debt.mean(axis=0)
    mean_debt = -np.minimum(liquid, 0.0).mean(axis=0) / avg_y

    # Wealth conditional on debt status, age by age (their inner ``a`` loop).
    T = liquid.shape[1]
    w_debt = np.full(T, np.nan)
    w_nodebt = np.full(T, np.nan)
    for a in range(T):
        d = has_debt[:, a]
        if d.any():
            w_debt[a] = wealth[d, a].mean()
        if (~d).any():
            w_nodebt[a] = wealth[~d, a].mean()
    w_debt = w_debt / avg_y
    w_nodebt = w_nodebt / avg_y

    def band_avg(series: np.ndarray) -> list[float]:
        out = []
        for band in BANDS:
            v, w = series[band], alive[band]
            ok = np.isfinite(v)
            out.append(float((v[ok] * w[ok]).sum() / w[ok].sum()))
        return out

    return np.array(
        band_avg(frac_debt) + band_avg(mean_debt)
        + band_avg(w_debt) + band_avg(w_nodebt)
    )


def their_simulated_moments(run: str = "table3") -> np.ndarray:
    """Their own simulated moments at their optimum (``MSMout.optMoments``).

    Comparing against these rather than the data moments isolates *port
    fidelity* from their model's own misfit -- their objective at the optimum
    is q = 77.2, so a perfect port still sits well away from the data.
    """
    from scipy.io import loadmat

    d = loadmat(
        f"replication-package-LLMRT/LifecycleSimulation/output/simulations/"
        f"table3/EDFbatch_{run}.mat",
        struct_as_record=False, squeeze_me=True,
    )
    return np.asarray(d["MSMout"].optMoments).ravel()


def msm_objective(sim: np.ndarray) -> float:
    """``q = dev' W dev`` with their benchmark diagonal weighting matrix."""
    dev = sim - cal.TARGET_MOMENTS
    w = 1.0 / cal.TARGET_MOMENT_SE**2
    return float((dev**2 * w).sum())


def report(name: str, spec: ModelSpec, prefs: tuple[float, float, float],
           pop: int, seed: int) -> np.ndarray:
    from hh_npe.simulator import grids

    age = grids.ages(spec.age_start, spec.age_end)
    X, _ = grids.liquid_grid(age, spec.xjump, spec.xmax, spec.x_cells_per_step)
    Z = grids.illiquid_grid(spec.zjump, spec.zmax, spec.z_cells_per_step)

    t0 = time.time()
    sol = solve(*prefs, spec)
    solve_s = time.time() - t0
    panel = simulate(sol, n_households=pop, seed=seed)
    sim = simulated_moments(panel)

    print(f"\n=== {name}: grid {len(X)}x{len(Z)}, solve {solve_s:.1f}s, "
          f"prefs beta={prefs[0]:.4f} delta={prefs[1]:.4f} rho={prefs[2]:.4f}")
    theirs = their_simulated_moments()
    print(f"{'moment':<16}{'band':<8}{'data':>9}{'THEIR sim':>10}{'our sim':>10}"
          f"{'t-stat':>8}{'ours/theirs':>12}")
    labels = ["%Visa", "meanVisa", "wealth|debt", "wealth|no debt"]
    bands = ["21-30", "31-40", "41-50", "51-60"]
    for i in range(16):
        tgt, se = cal.TARGET_MOMENTS[i], cal.TARGET_MOMENT_SE[i]
        print(f"{labels[i // 4]:<16}{bands[i % 4]:<8}{tgt:9.4f}{theirs[i]:10.4f}"
              f"{sim[i]:10.4f}{(sim[i] - tgt) / se:8.2f}{sim[i] / theirs[i]:12.2f}")
    print(f"MSM objective q = {msm_objective(sim):.1f}   "
          f"(theirs at the same prefs = {msm_objective(theirs):.1f})")
    print(f"port fidelity: mean |log(ours/theirs)| = "
          f"{np.abs(np.log(sim / theirs)).mean():.4f}")
    return sim


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pop", type=int, default=10000, help="their setup.pop")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full", action="store_true",
                   help="also run their exact 190x84 grid (slow)")
    args = p.parse_args()

    mid = dataclasses.replace(
        COARSE, xjump=2000.0, x_cells_per_step=25, zjump=8000.0, z_cells_per_step=7
    )
    runs = [("coarse (NPE candidate)", COARSE), ("mid", mid)]
    if args.full:
        runs.append(("full (their exact grid)", ModelSpec()))

    print(f"Laibson et al. table 3, benchmark naive quasi-hyperbolic; "
          f"pop={args.pop}, seed={args.seed}")
    print(f"Their reported objective at these estimates: q = 77.2")

    sims = {}
    for name, spec in runs:
        sims[name] = report(name, spec, cal.BENCHMARK_PREFS, args.pop, args.seed)

    if len(sims) > 1:
        base = sims[runs[-1][0]]
        print(f"\n=== grid-coarsening error vs '{runs[-1][0]}'")
        for name, sim in sims.items():
            if name == runs[-1][0]:
                continue
            rel = np.abs(sim - base) / np.maximum(np.abs(base), 1e-9)
            in_se = np.abs(sim - base) / cal.TARGET_MOMENT_SE
            print(f"  {name:26s} max rel {rel.max():6.2%}  "
                  f"median rel {np.median(rel):6.2%}  max |diff|/se {in_se.max():.2f}")


if __name__ == "__main__":
    main()
