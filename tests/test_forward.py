"""Tests for forward simulation."""

import numpy as np
import pytest

from hh_npe.simulator.forward import simulate_households
from hh_npe.simulator.lifecycle import build_lifecycle_agent, solve_lifecycle


@pytest.fixture(scope="module")
def small_solved_agent():
    a = build_lifecycle_agent(
        delta=0.96,
        crra=2.0,
        n_income_states=5,
        age_start=20,
        age_end=30,
    )
    return solve_lifecycle(a)


def test_output_shapes(small_solved_agent):
    out = simulate_households(small_solved_agent, n_households=50, seed=0)
    T = small_solved_agent.T_cycle
    for key in ("income", "consumption", "liquid_assets", "cash_on_hand", "pLvl"):
        assert out[key].shape == (50, T), f"{key} has shape {out[key].shape}"


def test_income_positive(small_solved_agent):
    out = simulate_households(small_solved_agent, n_households=30, seed=0)
    assert (out["income"] > 0).all()


def test_consumption_positive(small_solved_agent):
    out = simulate_households(small_solved_agent, n_households=30, seed=0)
    assert (out["consumption"] > 0).all()


def test_liquid_assets_above_borrowing_constraint(small_solved_agent):
    """With BoroCnstArt=0, end-of-period assets should be >= 0."""
    out = simulate_households(small_solved_agent, n_households=30, seed=0)
    # Allow tiny numerical slack
    assert (out["liquid_assets"] >= -1e-8).all()


def test_mrkv_state_in_valid_range(small_solved_agent):
    out = simulate_households(small_solved_agent, n_households=30, seed=0)
    assert out["mrkv_state"].min() >= 0
    assert out["mrkv_state"].max() < 5  # n_income_states


def test_seed_reproducible(small_solved_agent):
    a = simulate_households(small_solved_agent, n_households=20, seed=123)
    b = simulate_households(small_solved_agent, n_households=20, seed=123)
    np.testing.assert_array_equal(a["income"], b["income"])
    np.testing.assert_array_equal(a["consumption"], b["consumption"])


def test_different_seeds_differ(small_solved_agent):
    a = simulate_households(small_solved_agent, n_households=20, seed=0)
    b = simulate_households(small_solved_agent, n_households=20, seed=1)
    assert not np.allclose(a["income"], b["income"])


def test_unsolved_agent_raises():
    a = build_lifecycle_agent(
        delta=0.96, crra=2.0, n_income_states=3, age_start=20, age_end=25
    )
    with pytest.raises(ValueError, match="solved"):
        simulate_households(a, n_households=5)
