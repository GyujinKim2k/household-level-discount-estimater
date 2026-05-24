"""Annual survival probabilities from SSA period life tables.

Wraps ``HARK.Calibration.life_tables.us_ssa`` (SSA Trustees Report 2020 tables;
last pre-pandemic historical year is 2018-2020 depending on Method).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd

Sex = Literal["M", "F", "average"]


@lru_cache(maxsize=1)
def _load_ssa() -> pd.DataFrame:
    from HARK.Calibration.life_tables.us_ssa.SSATools import get_ssa_life_tables

    return get_ssa_life_tables()


def survival_schedule(
    age_start: int = 20,
    age_end: int = 90,
    year: int = 2020,
    sex: Sex = "average",
) -> np.ndarray:
    """Annual survival probabilities for ages ``[age_start, age_end)``.

    ``LivPrb[i]`` is the probability of surviving from age ``age_start + i``
    to age ``age_start + i + 1``. Output has shape ``(age_end - age_start,)``.

    Parameters
    ----------
    age_start, age_end : int
        Inclusive start, exclusive end (in years). Default 20-90 (70 values).
    year : int
        SSA period life table year. Default 2020 (pre-pandemic).
    sex : "M", "F", or "average"
        ``"average"`` uses the unweighted mean of male and female schedules
        — appropriate for a sex-agnostic representative agent in the MVP.
    """
    df = _load_ssa()
    df_y = df[df["Year"] == year]
    if df_y.empty:
        raise ValueError(f"No SSA life table for year={year}")

    if sex == "average":
        q_m = _extract_q(df_y[df_y["Sex"] == "M"], age_start, age_end)
        q_f = _extract_q(df_y[df_y["Sex"] == "F"], age_start, age_end)
        q = 0.5 * (q_m + q_f)
    elif sex in ("M", "F"):
        q = _extract_q(df_y[df_y["Sex"] == sex], age_start, age_end)
    else:
        raise ValueError(f"sex must be 'M', 'F', or 'average'; got {sex!r}")

    return 1.0 - q


def _extract_q(df: pd.DataFrame, age_start: int, age_end: int) -> np.ndarray:
    df = df.sort_values("x")
    df = df[(df["x"] >= age_start) & (df["x"] < age_end)]
    expected = age_end - age_start
    if len(df) != expected:
        raise ValueError(
            f"Expected {expected} life-table rows for ages "
            f"[{age_start}, {age_end}); got {len(df)}"
        )
    return df["q(x)"].to_numpy()
