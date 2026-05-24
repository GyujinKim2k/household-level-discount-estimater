"""Tests for the SSA-based mortality schedule."""

import numpy as np
import pytest

from hh_npe.simulator.mortality import survival_schedule


def test_default_shape_and_range():
    s = survival_schedule()
    assert s.shape == (70,)  # ages 20..89 inclusive
    assert (s > 0).all()
    assert (s <= 1).all()


def test_survival_decreases_at_older_ages():
    """Annual survival probabilities decline with age over 60-89."""
    s = survival_schedule(age_start=60, age_end=90)
    # Average over first 10 years (60s) vs last 10 years (80s)
    assert s[:10].mean() > s[-10:].mean()


def test_average_is_mean_of_M_and_F():
    m = survival_schedule(sex="M")
    f = survival_schedule(sex="F")
    avg = survival_schedule(sex="average")
    np.testing.assert_allclose(avg, 0.5 * (m + f))


def test_female_survival_dominates_male():
    """Women have higher survival probabilities at almost every age."""
    m = survival_schedule(sex="M")
    f = survival_schedule(sex="F")
    # Allow a couple of ages where they cross; expect dominance in vast majority.
    assert (f >= m).mean() > 0.9


def test_invalid_sex_raises():
    with pytest.raises(ValueError, match="sex"):
        survival_schedule(sex="X")  # type: ignore[arg-type]


def test_unknown_year_raises():
    with pytest.raises(ValueError, match="No SSA"):
        survival_schedule(year=1850)


def test_custom_age_range():
    s = survival_schedule(age_start=30, age_end=50)
    assert s.shape == (20,)
