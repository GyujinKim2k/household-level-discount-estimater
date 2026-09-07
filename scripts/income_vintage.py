"""Has the age-income profile moved since Laibson et al. estimated it?

Their entire first stage -- and therefore every simulated panel -- rests on an
income process estimated on **PSID 1982-1991** for `comphs` households
(`laibson_calibration.YWORK_*`, frozen from `3_firststageregs_bs.do`)::

    log y(age) = cons + agecoeff*age + age2coeff*age^2/100 + age3coeff*age^3/10000
                      + spousecoeff*spouse + kidscoeff*kids + depadulcoeff*depadul

We apply that process to households observed **2011-2023**. Between those two
windows, real wages for high-school-educated workers stagnated. If the profile
has flattened or shifted down, the simulator generates a lifetime income path
those households never had, and no preference parameter can repair it -- which
would explain why the model over-predicts consumption at every (beta, delta,
rho) we have tried.

Direct evidence for this already exists: on the `comphs` sample the calibration
was built for, the simulator's median income at ages 35-44 is 49,000 against
PSID's 43,353. This re-estimates their equation on our own waves and compares
the implied profiles.

Comparison is of the **fitted profile**, not coefficient by coefficient. The
age cubic is badly collinear, so individual coefficients are not separately
interpretable; the curve they imply is.

Usage::

    uv run python scripts/income_vintage.py --x data/processed/psid_x_matched.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hh_npe.simulator import laibson_calibration as cal
from scripts.typical_household import (MARRIED, MSTAT, MISSING_FROM, NFU,
                                       NKIDS, WAVES, model_depadul, model_kids)


def their_log_income(age: np.ndarray, spouse, kids, depadul) -> np.ndarray:
    """Laibson et al.'s frozen 1982-91 profile (grids.py:53-60)."""
    return (cal.YWORK_CONS
            + cal.YWORK_AGECOEFF * age
            + cal.YWORK_AGE2COEFF * age ** 2 / 100.0
            + cal.YWORK_AGE3COEFF * age ** 3 / 10000.0
            + cal.YWORK_SPOUSECOEFF * spouse
            + cal.YWORK_KIDSCOEFF * kids
            + cal.YWORK_DEPADULCOEFF * depadul)


def design(age, spouse, kids, depadul):
    return np.column_stack([
        np.ones_like(age), age, age ** 2 / 100.0, age ** 3 / 10000.0,
        spouse, kids, depadul,
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", type=Path,
                    default=Path("data/processed/psid_x_matched.pt"))
    ap.add_argument("--psid_dir", type=Path, default=Path("PSID-data"))
    args = ap.parse_args()

    d = torch.load(args.x, weights_only=False)
    x = d["x"].numpy().astype(np.float64)
    rows = d["psid_row"].numpy()
    fam = pd.read_pickle(args.psid_dir / "tax.pkl")

    def col(c):
        s = fam[c]
        return s.where(s < MISSING_FROM, np.nan).to_numpy()[rows]

    nkids = np.column_stack([col(NKIDS[y]) for y in WAVES])
    nfu = np.column_stack([col(NFU[y]) for y in WAVES])
    married = np.column_stack([col(MSTAT[y]) for y in WAVES]) == MARRIED
    # grids.py:38 sets `spouse = 2.0` for every simulated household at every
    # age, so the model's "spouse" is the count of core adults, not a 0/1 flag.
    nhead = np.where(married, 2.0, 1.0)
    ndepad = np.clip(nfu - nkids - nhead, 0, None)

    # Their specification is `xtreg lny nkids nhead ndepadul urate cohd* age
    # agesq agecu, fe` -- household FIXED EFFECTS, so the age profile is
    # identified from households aging within the panel, not from comparing
    # different households. Pooled OLS instead conflates the age profile with
    # permanent differences between households, and gives a `nhead` coefficient
    # nearly 3x theirs because married households differ in more than headcount.
    from scripts.typical_household import STATE, state_unemployment
    ue = state_unemployment(args.psid_dir, Path(
        "replication-package-LLMRT/ParameterAndMoments/PSID/data"))
    st = np.column_stack([col(STATE[y_]) for y_ in WAVES])
    N, W = st.shape
    urate = np.array([[ue.get((int(st[i, w]) if np.isfinite(st[i, w]) else -1,
                               WAVES[w]), np.nan) for w in range(W)]
                      for i in range(N)])
    urate = np.where(np.isfinite(urate), urate, np.nanmean(urate))

    age = x[:, :, 4].ravel()
    inc = x[:, :, 0].ravel()
    hh = np.repeat(np.arange(N), W)
    # Cohort is time-invariant, so the household fixed effect absorbs it; the
    # cohd* terms in their code are collinear with the FE and drop out.
    cols = np.column_stack([nkids.ravel(), nhead.ravel(), ndepad.ravel(),
                            urate.ravel(), age, age ** 2 / 100.0,
                            age ** 3 / 10000.0])
    ok = np.isfinite(cols).all(axis=1) & (inc > 1000)
    y = np.log(inc[ok]); Xf = cols[ok]; g = hh[ok]

    def demean(v, grp):
        df = pd.DataFrame(v)
        return (df - df.groupby(grp).transform("mean")).to_numpy()

    bw, *_ = np.linalg.lstsq(demean(Xf, g), demean(y.reshape(-1, 1), g).ravel(),
                             rcond=None)
    # Map back to their term order; the constant is not identified under FE, so
    # it is recovered by matching the fitted mean to the sample mean.
    b = np.empty(7)
    b[1], b[2], b[3] = bw[4], bw[5], bw[6]          # age, agesq, agecu
    b[4], b[5], b[6] = bw[1], bw[0], bw[2]          # nhead, nkids, ndepadul
    fitted = design(Xf[:, 4], Xf[:, 1], Xf[:, 0], Xf[:, 2])[:, 1:] @ b[1:]
    b[0] = y.mean() - fitted.mean()

    theirs = np.array([cal.YWORK_CONS, cal.YWORK_AGECOEFF, cal.YWORK_AGE2COEFF,
                       cal.YWORK_AGE3COEFF, cal.YWORK_SPOUSECOEFF,
                       cal.YWORK_KIDSCOEFF, cal.YWORK_DEPADULCOEFF])
    names = ["const", "age", "age^2/100", "age^3/10000", "spouse", "kids",
             "depadul"]
    print(f"n = {int(ok.sum())} household-waves, comphs, 2011-2023\n")
    print(f"{'term':14s}{'theirs (1982-91)':>19s}{'ours (2011-23)':>17s}")
    for n, t, o in zip(names, theirs, b):
        print(f"{n:14s}{t:19.4f}{o:17.4f}")

    # The curve is what matters -- the cubic terms are collinear and not
    # separately interpretable. Evaluate both at the MODEL's own composition,
    # which is what the simulator actually feeds through.
    ages = np.arange(25, 61, 5).astype(float)
    sp = np.full_like(ages, 2.0)
    kk, dd = model_kids(ages), model_depadul(ages)
    lt = their_log_income(ages, sp, kk, dd)
    lo = design(ages, sp, kk, dd) @ b
    print(f"\nfitted income at the model's own composition (2010 USD)")
    print(f"{'age':>5s}{'theirs':>11s}{'ours':>11s}{'ratio':>9s}")
    for a_, t_, o_ in zip(ages, np.exp(lt), np.exp(lo)):
        print(f"{a_:5.0f}{t_:11.0f}{o_:11.0f}{o_ / t_:9.3f}")

    # Slope of the profile over the working years the window covers.
    def growth(logv):
        return float(np.exp(logv[ages == 55][0] - logv[ages == 25][0]))
    print(f"\nincome growth, age 25 -> 55")
    print(f"  theirs (1982-91): {growth(lt):.2f}x")
    print(f"  ours   (2011-23): {growth(lo):.2f}x")
    print("\nA flatter profile means the simulator hands households a lifetime")
    print("income path they never had. No (beta, delta, rho) can repair that.")


if __name__ == "__main__":
    main()
