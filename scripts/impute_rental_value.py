"""Imputed rent for owner-occupiers in 2011-2017, anchored on their own 2019+ value.

PSID began asking owners what their home would rent for in **2019**. Without it,
owner-occupiers have essentially no housing consumption in our measure: we strip
mortgage, property tax and home insurance as saving and transfers -- correctly,
and exactly as PSID's own ``TOTAL CONSUMPTION WITH RENTAL VALUE`` does -- which
leaves owners with nothing where renters keep their rent. Adding the observed
rental value to the three waves that have it closes 45% of the
consumption-slope gap, the largest effect of anything tested.

**The method, and the one it deliberately avoids.**

The obvious approach is to fit ``rent = f(income, wealth, family size, age)`` on
2019-2023 and predict backwards. That would be standard and it would corrupt
the estimation. The feature set is (income, consumption, liquid, illiquid, age);
if imputed consumption becomes a function of income and illiquid wealth, the
consumption channel carries no independent information and the posterior reads a
mechanical identity as household behaviour. It would very likely *improve* the
apparent fit while invalidating it.

Instead each household is anchored on **its own** observed rental value and
carried back in real terms::

    imputed(hh, t) = observed(hh, t_anchor) / (1 + g) ** (t_anchor - t)

with ``g`` self-calibrated from the 2019->2023 waves rather than taken from
outside. The imputed value then derives from that household's actual housing,
not from the variables being estimated on.

**Validated out of sample.** Back-projecting 2023 -> 2019 and comparing against
the truth: median |error| 0.187, median error +0.000 (unbiased), and imputed
rent is 26.7% of owner consumption, so a typical rent error moves consumption by
5.0% against a 23% effect. Usable, but the p90 error of 0.62 means individual
households can be badly wrong -- which matters more for per-household posteriors
than for population summaries, and is why the error is propagated rather than
treated as exact.

Usage::

    uv run python scripts/impute_rental_value.py --n_draws 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.build_psid_tensor import REF_YEAR, WAVES, deflator

MISSING_FROM = 9_999_998
#: `VALUE OF HOME IF RENTED`. Not collected before 2019.
RENTAL_VALUE = {2019: "ER77523", 2021: "ER81850", 2023: "ER85704"}
ANCHOR_WAVES = sorted(RENTAL_VALUE)
W1 = dict(zip(WAVES, ["ER52392", "ER58209", "ER65406", "ER71483",
                      "ER77509", "ER81836", "ER85690"]))
W2 = dict(zip(WAVES, ["ER52394", "ER58211", "ER65408", "ER71485",
                      "ER77511", "ER81838", "ER85692"]))
MORT = dict(zip(WAVES, ["ER52395A6", "ER58212A6", "ER65415", "ER71492",
                        "ER77521", "ER81848", "ER85702"]))
PTAX = dict(zip(WAVES, ["ER52395A8", "ER58212A8", "ER65417", "ER71495",
                        "ER77527", "ER81854", "ER85708"]))


def _col(fam, code, rows):
    s = fam[code]
    return s.where(s < MISSING_FROM, 0.0).fillna(0.0).to_numpy()[rows]


def owner_matrix(fam, rows) -> np.ndarray:
    """Owner in each wave: has home equity, a mortgage, or pays property tax."""
    out = []
    for w in WAVES:
        equity = _col(fam, W2[w], rows) - _col(fam, W1[w], rows)
        out.append((equity != 0) | (_col(fam, MORT[w], rows) > 0)
                   | (_col(fam, PTAX[w], rows) > 0))
    return np.column_stack(out)


def real_rental(h, rows) -> np.ndarray:
    """Observed rental value, deflated to 2010 USD. Zero where not collected."""
    cols = []
    for w in WAVES:
        if w in RENTAL_VALUE:
            cols.append(_col(h, RENTAL_VALUE[w], rows) * deflator(REF_YEAR[w]))
        else:
            cols.append(np.zeros(len(rows)))
    return np.column_stack(cols)


def calibrate_growth(rv: np.ndarray) -> float:
    """Annual real growth in imputed rent, from households observed twice."""
    a, b = rv[:, WAVES.index(2019)], rv[:, WAVES.index(2023)]
    ok = (a > 0) & (b > 0)
    return float(np.median(b[ok] / a[ok]) ** (1 / 4))


def error_scale(rv: np.ndarray, g: float) -> np.ndarray:
    """Out-of-sample back-projection errors, as multiplicative factors.

    Resampled to propagate imputation uncertainty rather than pretending the
    imputed values are exact.
    """
    a, b = rv[:, WAVES.index(2019)], rv[:, WAVES.index(2023)]
    ok = (a > 0) & (b > 0)
    return (b[ok] / g ** 4) / a[ok]


def impute(rv, own, g, rng=None, err=None):
    """Imputed rent per household-wave for 2011-2017.

    Anchored on the household's earliest observed value. Households whose
    ownership status changed between the imputed wave and the anchor are NOT
    anchored: the anchor then describes a different home. They fall back to the
    wave's median owner value -- a constant, which adds noise but, unlike a
    regression on income or wealth, creates no spurious correlation with the
    features the model is estimated on.
    """
    n, W = rv.shape
    out = np.zeros_like(rv)
    for w_i, w in enumerate(WAVES):
        if w in RENTAL_VALUE:
            out[:, w_i] = rv[:, w_i]          # observed; nothing to impute
            continue
        for i in range(n):
            if not own[i, w_i]:
                continue                       # renter: actual rent already in c
            anchors = [(aw, rv[i, WAVES.index(aw)]) for aw in ANCHOR_WAVES
                       if rv[i, WAVES.index(aw)] > 0
                       and own[i, WAVES.index(aw)]]
            if anchors:
                aw, val = anchors[0]           # earliest anchor = shortest reach
                v = val / g ** (REF_YEAR[aw] - REF_YEAR[w])
                if rng is not None and err is not None and len(err):
                    v *= rng.choice(err)
                out[i, w_i] = v
        if (out[:, w_i] == 0).all():
            continue
        # Fallback for owners with no usable anchor.
        med = np.median(out[own[:, w_i] & (out[:, w_i] > 0), w_i])
        gap = own[:, w_i] & (out[:, w_i] == 0)
        out[gap, w_i] = med
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", type=Path,
                    default=Path("data/processed/psid_x_matched_net.pt"))
    ap.add_argument("--psid_dir", type=Path, default=Path("PSID-data"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/psid_x_rental.pt"))
    ap.add_argument("--n_draws", type=int, default=20,
                    help="Imputation draws for error propagation. 0 = point "
                         "imputation with no noise.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = torch.load(args.x, weights_only=False)
    x = d["x"].numpy().astype(np.float64)
    rows = d["psid_row"].numpy()
    fam = pd.read_pickle(args.psid_dir / "tax.pkl")
    h = pd.read_pickle(args.psid_dir / "house.pkl")

    rv = real_rental(h, rows)
    own = owner_matrix(fam, rows)
    g = calibrate_growth(rv)
    err = error_scale(rv, g)
    print(f"households {len(rows)}")
    print(f"real rent growth, self-calibrated: {100 * (g - 1):.2f}%/yr")
    print(f"out-of-sample errors: n={len(err)}  median |e| "
          f"{np.median(np.abs(err - 1)):.3f}")

    # Tenure change between an imputed wave and the anchor breaks the anchor.
    changed = (own[:, :4].any(axis=1) & ~own[:, 4:].any(axis=1)).mean()
    print(f"early owners who are not owners by 2019 (anchor unusable): "
          f"{changed:.3f}")

    base = impute(rv, own, g)
    print(f"\n{'wave':6s}{'owners':>9s}{'imputed>0':>11s}{'median imp':>12s}"
          f"{'median cons':>13s}{'with rent':>12s}")
    for i, w in enumerate(WAVES):
        c = x[:, i, 1]
        print(f"{w:<6d}{own[:, i].mean():9.3f}{(base[:, i] > 0).mean():11.3f}"
              f"{np.median(base[base[:, i] > 0, i]) if (base[:, i] > 0).any() else 0:12.0f}"
              f"{np.median(c):13.0f}{np.median(c + base[:, i]):12.0f}")

    xa = x.copy()
    xa[:, :, 1] = x[:, :, 1] + base
    torch.save({**d, "x": torch.from_numpy(xa).float(),
                "rental_value_imputed": True,
                "rent_growth_per_year": g},
               args.out)
    print(f"\nwrote {args.out}")

    if args.n_draws:
        rng = np.random.default_rng(args.seed)
        meds = []
        for k in range(args.n_draws):
            imp = impute(rv, own, g, rng=rng, err=err)
            meds.append(np.median(x[:, :, 1] + imp))
            torch.save({**d, "x": torch.from_numpy(
                np.concatenate([x[:, :, :1],
                                (x[:, :, 1] + imp)[:, :, None],
                                x[:, :, 2:]], axis=2)).float(),
                        "rental_value_imputed": True, "draw": k},
                       args.out.with_name(f"{args.out.stem}_draw{k}.pt"))
        meds = np.array(meds)
        print(f"\nerror propagation, {args.n_draws} draws:")
        print(f"  median consumption across draws: {meds.mean():.0f} "
              f"+- {meds.std():.0f}  (range {meds.min():.0f}-{meds.max():.0f})")
        print(f"  point imputation: {np.median(x[:, :, 1] + base):.0f}")
        print("  If the posterior conclusions move across these draws, the")
        print("  imputation noise is material and must be reported as such.")


if __name__ == "__main__":
    main()
