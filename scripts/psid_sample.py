"""Realised N in the PSID extract, and the sample-selection ladder behind it.

Answers the last open question before Phase 4: how many households does PSID
actually supply for a 7-wave, ages-25-46-at-entry panel with a clean
credit-card measure (PSID_DATA.md)?

Reads a PSID Data Center ASCII extract by parsing column positions out of the
companion SAS layout, so the variable list is taken from the file rather than
hardcoded -- a re-pull with more variables still loads.

Each filter is reported with the count it removes. The selection is a modelling
decision, not data cleaning: "head in all 7 waves" is the strict reading of the
model's household (one decision unit for life), and it is where most of the
sample goes. Loosening it changes N by roughly a factor of two, so the ladder
is printed rather than the endpoint.

Usage::

    uv run python scripts/psid_sample.py --extract PSID-data/J364786
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

WAVES = [2011, 2013, 2015, 2017, 2019, 2021, 2023]

# Cross-year individual file: interview number, sequence, relation. Non-zero
# interview number means the individual was in a responding FU that wave.
IV = dict(zip(WAVES, ["ER34101", "ER34201", "ER34301", "ER34501",
                      "ER34701", "ER34901", "ER35101"]))
REL = dict(zip(WAVES, ["ER34103", "ER34203", "ER34303", "ER34503",
                       "ER34703", "ER34903", "ER35103"]))
# Family-level, attached to the individual. Labelled "AGE OF HEAD" through 2015
# and "AGE OF REFERENCE PERSON" from 2017 -- PSID renamed the role, same field.
AGE = dict(zip(WAVES, ["ER47317", "ER53017", "ER60017", "ER66017",
                       "ER72017", "ER78017", "ER82018"]))
CC = dict(zip(WAVES, ["ER48937", "ER54687", "ER61798", "ER67852",
                      "ER73880", "ER80002", "ER83972"]))
WEALTH2 = dict(zip(WAVES, ["ER52394", "ER58211", "ER65408", "ER71485",
                           "ER77511", "ER81838", "ER85692"]))
EXPEND = dict(zip(WAVES, ["ER52395E4", "ER58212E4", "ER65448B", "ER71527B",
                          "ER77587", "ER81914", "ER85768"]))

HEAD_CODE = 10          # relation-to-head/reference-person
MISSING_FROM = 9_999_998  # PSID DK/NA sentinels at the top of each field


def load_extract(stem: Path) -> pd.DataFrame:
    """Read `<stem>.txt` using the column positions in `<stem>.sas`."""
    sas = Path(f"{stem}.sas").read_text()
    block = sas[sas.index("INPUT"):]
    block = block[: block.index(";")]
    spec = re.findall(r"(\w+)\s+(\d+)\s*-\s*(\d+)", block)
    if not spec:
        raise SystemExit(f"no INPUT positions found in {stem}.sas")
    names = [s[0] for s in spec]
    colspecs = [(int(a) - 1, int(b)) for _, a, b in spec]
    return pd.read_fwf(f"{stem}.txt", colspecs=colspecs, names=names,
                       dtype="float64")


def selection_ladder(df: pd.DataFrame, age_low: int, age_high: int):
    """Successive filters with the count each one removes."""
    steps = [("rows in extract", pd.Series(True, index=df.index))]

    def add(label, mask):
        steps.append((label, steps[-1][1] & mask))

    add("interviewed in all 7 waves",
        np.logical_and.reduce([df[IV[y]] > 0 for y in WAVES]))
    add("head/reference in all 7",
        np.logical_and.reduce([df[REL[y]] == HEAD_CODE for y in WAVES]))
    add(f"age {age_low}-{age_high} at 2011",
        df[AGE[WAVES[0]]].between(age_low, age_high))
    # PSID ages are taken at interview date, so a 12-year span reads as 11-13
    # depending on the month. Anything outside that is an inconsistent record,
    # not a birthday: PSID documents age-reporting errors, and a household
    # whose head ages -24 years is not one household.
    add("age progresses 11-13 years",
        (df[AGE[WAVES[-1]]] - df[AGE[WAVES[0]]]).between(11, 13))
    add("credit card non-missing in all 7",
        np.logical_and.reduce([df[CC[y]] < MISSING_FROM for y in WAVES]))
    add("wealth non-missing in all 7",
        np.logical_and.reduce([df[WEALTH2[y]] < MISSING_FROM for y in WAVES]))
    add("expenditure > 0 in all 7",
        np.logical_and.reduce([df[EXPEND[y]] > 0 for y in WAVES]))
    return steps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extract", type=Path, default=Path("PSID-data/J364786"),
                   help="Path stem: expects <stem>.txt and <stem>.sas")
    p.add_argument("--age_low", type=int, default=25)
    p.add_argument("--age_high", type=int, default=46,
                   help="Matches window.start_age_high for n_waves=7 "
                        "(configs/npe/phase3.yaml).")
    args = p.parse_args()

    df = load_extract(args.extract)
    print(f"extract: {df.shape[0]} rows x {df.shape[1]} variables\n")

    steps = selection_ladder(df, args.age_low, args.age_high)
    prev = None
    for label, mask in steps:
        n = int(mask.sum())
        delta = "" if prev is None else f"  ({n - prev:+d})"
        print(f"{label:34s}{n:7d}{delta}")
        prev = n
    final = steps[-1][1]

    print(f"\nANALYSIS N = {int(final.sum())}")
    d = df.loc[final]
    print(f"age at 2011: {d[AGE[WAVES[0]]].min():.0f}-{d[AGE[WAVES[0]]].max():.0f}"
          f" (median {d[AGE[WAVES[0]]].median():.0f})")
    print(f"age at 2023: {d[AGE[WAVES[-1]]].min():.0f}-{d[AGE[WAVES[-1]]].max():.0f}")
    print("\ncredit-card debt, share > 0 by wave:")
    print("  " + "  ".join(f"{y}:{float((d[CC[y]] > 0).mean()):.3f}"
                           for y in WAVES))
    print("credit-card debt, median among holders:")
    print("  " + "  ".join(f"{y}:{d[CC[y]][d[CC[y]] > 0].median():.0f}"
                           for y in WAVES))


if __name__ == "__main__":
    main()
