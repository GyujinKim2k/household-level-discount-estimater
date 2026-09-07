"""Engel-curve diagnostics for the PSID consumption series.

The problem this addresses: PSID consumption sits ~34% below the simulated
level, and the gap is not uniform -- consumption/income runs 1.68 in the bottom
income decile to 0.40 in the top, against a flat ~0.99 in simulation, in every
age band. That gradient is exactly the pattern Aguiar & Bils (AER 2015)
document for consumption surveys: richer households under-report more, and the
under-reporting is concentrated in luxuries.

**What is and is not identified.** Budget shares are invariant to a
household-level multiplicative reporting error: if a household reports
``C_obs = C_true * exp(-phi)``, every category scales by the same factor and
the shares are unchanged. That makes shares informative about *relative*
position in a way the reported total is not. But fitting Engel curves on the
reported total and then inverting the same system returns the reported total
mechanically -- the level of ``phi`` is **not identified from a single
cross-section**. Aguiar & Bils recover it from the time dimension plus an
assumption that Engel curves are stable; a level correction otherwise needs an
anchor from outside PSID.

So this script does two separable things:

1. **Diagnostic (identified).** Estimate the Engel system and test whether the
   share structure moves with income in the way a necessity/luxury ordering
   predicts. This says whether the consumption gradient is consistent with
   differential under-reporting at all.

2. **Correction (assumption-dependent).** Food-based imputation in the tradition
   of Skinner (1987) and Blundell, Pistaferri & Preston (2008): food is the
   best-measured category, so its Engel curve maps food spending to total
   consumption. The elasticity is taken from the literature, not estimated
   here, and the level is anchored on the bottom income decile where
   under-reporting is assumed minimal. Both assumptions are stated in the
   output, because both are load-bearing.

Usage::

    uv run python scripts/engel_correction.py --out data/processed/psid_cons_corrected.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

WAVES = [2011, 2013, 2015, 2017, 2019, 2021, 2023]
MISSING_FROM = 9_999_998

#: Non-overlapping leaves that sum exactly to our consumption measure. PSID's
#: own aggregates cannot be used directly: `hous` nests utilities and
#: telephone, and `trans` nests the vehicle principal we strip as saving, so
#: summing the aggregates double-counts by ~17%.
LEAVES = ("food", "rent", "util", "tel", "trans", "health", "educ", "child",
          "cloth", "trips", "recr", "furn", "rep")

#: Total-expenditure elasticity of food. The literature is consistent and old:
#: food is a necessity with elasticity around 0.5-0.6. Used only by the
#: correction, never by the diagnostic.
FOOD_ELASTICITY = 0.55

#: `# IN FU`, per wave. Family size is not optional here. It rises
#: monotonically with income in this sample (2.20 members in the bottom decile
#: to 3.66 in the top), and food scales with members far more than luxuries do.
#: Omit it and the food equation attributes that variation to consumption,
#: which is how the first pass produced correction factors above 3x.
NFU = dict(zip(WAVES, ["ER47316", "ER53016", "ER60016", "ER66016",
                       "ER72016", "ER78016", "ER82017"]))


def load_panel(psid_dir: Path, x_path: Path) -> pd.DataFrame:
    """Long panel of category spending, deflated, one row per household-wave."""
    from scripts.build_psid_tensor import CPI, CPI_BASE, REF_YEAR

    fam = pd.read_pickle(psid_dir / "tax.pkl")
    emap = json.loads((psid_dir / "exp_map.json").read_text())
    d = torch.load(x_path, weights_only=False)
    rows = d["psid_row"].numpy()
    x = d["x"].numpy()

    def g(key: str, wave: int) -> np.ndarray:
        col = emap[str(wave)][key]
        if col is None:
            return np.zeros(len(rows))
        s = fam[col]
        return s.where(s < MISSING_FROM, 0.0).fillna(0.0).to_numpy()[rows]

    out = []
    for w, wave in enumerate(WAVES):
        defl = CPI_BASE / CPI[REF_YEAR[wave]]
        cat = {
            "food": g("food", wave), "rent": g("rent", wave),
            "util": g("util", wave), "tel": g("tel", wave),
            "trans": g("trans", wave) - g("vdown", wave) - g("vloan", wave),
            "health": g("health", wave), "educ": g("educ", wave),
            "child": g("child", wave), "cloth": g("cloth", wave),
            "trips": g("trips", wave), "recr": g("recr", wave),
            "furn": g("furn", wave), "rep": g("rep", wave),
        }
        df = pd.DataFrame({k: v * defl for k, v in cat.items()})
        df["hh"] = np.arange(len(rows))
        df["wave"] = wave
        df["cons"] = x[:, w, 1]      # already deflated in the tensor
        df["income"] = x[:, w, 0]
        df["age"] = x[:, w, 4]
        nfu = fam[NFU[wave]].to_numpy()[rows]
        df["nfu"] = np.clip(np.nan_to_num(nfu, nan=1.0), 1, 20)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def engel_system(p: pd.DataFrame) -> pd.DataFrame:
    """Share-on-log-total regressions, one per category, with age controls.

    The sign of each slope orders necessities against luxuries. Slopes sum to
    zero by construction because the shares sum to one, which is a useful
    check that the leaves really do partition consumption.
    """
    q = p[(p.cons > 1000) & (p.income > 1000)].copy()
    lc = np.log(q.cons.to_numpy())
    A = np.column_stack([np.ones_like(lc), lc, q.age.to_numpy() / 10.0,
                         np.log(q.nfu.to_numpy())])
    res = []
    for c in LEAVES:
        w = (q[c] / q.cons).to_numpy()
        beta, *_ = np.linalg.lstsq(A, w, rcond=None)
        # Expenditure elasticity of good j: 1 + (dw/dlogC)/w at the mean share.
        wbar = w.mean()
        res.append({"category": c, "mean_share": wbar,
                    "dshare_dlogC": beta[1],
                    "elasticity": 1.0 + beta[1] / wbar if wbar > 0 else np.nan})
    return pd.DataFrame(res).sort_values("elasticity")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--psid_dir", type=Path, default=Path("PSID-data"))
    ap.add_argument("--x", type=Path, default=Path("data/processed/psid_x.pt"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/psid_cons_corrected.csv"))
    args = ap.parse_args()

    p = load_panel(args.psid_dir, args.x)
    print(f"panel: {len(p)} household-waves")

    e = engel_system(p)
    print("\n=== Engel system: share on log total consumption (age controlled) ===")
    print(f"{'category':10s}{'mean share':>12s}{'dw/dlogC':>11s}{'elasticity':>12s}")
    for _, r in e.iterrows():
        print(f"{r.category:10s}{r.mean_share:12.3f}{r.dshare_dlogC:11.4f}"
              f"{r.elasticity:12.2f}")
    print(f"{'SUM':10s}{e.mean_share.sum():12.3f}{e.dshare_dlogC.sum():11.4f}"
          "        (slopes must sum to ~0)")

    # --- diagnostic: does the share structure disagree with reported total? ---
    # Necessity shares should fall with true consumption. If a household's
    # reported total is understated, its necessity shares look too HIGH for its
    # income. Comparing the food share's income gradient against its reported-
    # consumption gradient is the sharpest version of that comparison.
    q = p[(p.cons > 1000) & (p.income > 1000)].copy()
    q["food_share"] = q.food / q.cons
    print("\n=== food share by income decile vs by reported-consumption decile ===")
    for by in ("income", "cons"):
        q["d"] = pd.qcut(q[by], 10, labels=False)
        med = q.groupby("d")["food_share"].median()
        print(f"  by {by:7s} " + " ".join(f"{v:5.3f}" for v in med.values))
    print("  A necessity share falls in BOTH orderings. If it falls much more")
    print("  steeply against reported consumption than against income, the")
    print("  reported total is carrying measurement error that income is not.")

    # --- correction (assumption-dependent) --------------------------------
    # log(food) = a + b*log(C)  =>  log(C) = (log(food) - a)/b, with b taken
    # from the literature and `a` anchored so the bottom income decile is
    # unchanged -- that is where under-reporting is assumed smallest, and it is
    # an assumption, not a finding.
    q = q[q.food > 0].copy()
    lf, lc = np.log(q.food.to_numpy()), np.log(q.cons.to_numpy())
    ln, ag = np.log(q.nfu.to_numpy()), q.age.to_numpy() / 10.0

    # PSID's own food elasticity, with family size and age controlled. If this
    # lands near the literature's 0.5-0.6 the specification is sound; if it
    # stays near 1 the food equation is not measuring what it should and the
    # correction below is not usable.
    # Regress on consumption EXCLUDING food. Food is ~23% of the total, so
    # putting it on both sides biases the slope toward 1: the same regression
    # against total consumption returns 0.927, which would look like food is
    # nearly a luxury and is purely mechanical.
    lo = np.log((q.cons - q.food).clip(lower=500).to_numpy())
    Af = np.column_stack([np.ones_like(lf), lo, ln, ag])
    bf, *_ = np.linalg.lstsq(Af, lf, rcond=None)
    print(f"\n=== food equation (log food on log C, log N, age) ===")
    print(f"  elasticity w.r.t. consumption : {bf[1]:.3f}   "
          f"(literature 0.50-0.60)")
    print(f"  elasticity w.r.t. family size : {bf[2]:.3f}")

    # Impute with the literature elasticity, holding family size and age at
    # their household-specific values, anchored on the bottom income decile
    # where under-reporting is assumed smallest. That anchor is an assumption,
    # not a finding.
    dec = pd.qcut(q.income, 10, labels=False)
    anchor = dec == 0
    resid = lf - FOOD_ELASTICITY * lo - bf[2] * ln - bf[3] * ag
    a = np.median(resid[anchor])
    q["cons_imputed"] = np.exp((lf - a - bf[2] * ln - bf[3] * ag)
                               / FOOD_ELASTICITY)
    q["factor"] = q.cons_imputed / q.cons

    print(f"\n=== implied correction factor, food elasticity {FOOD_ELASTICITY} ===")
    q["d"] = dec
    f = q.groupby("d")["factor"].median()
    print("  by income decile: " + " ".join(f"{v:5.2f}" for v in f.values))
    print(f"  median overall: {q.factor.median():.2f}")
    print("\n  Rising factors = under-reporting grows with income, which is the")
    print("  Aguiar-Bils pattern. FLAT factors would mean the consumption")
    print("  gradient is behaviour, not measurement, and no correction is due.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    q[["hh", "wave", "income", "cons", "cons_imputed", "factor"]].to_csv(
        args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
