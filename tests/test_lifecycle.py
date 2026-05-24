"""Smoke + economic-sanity tests for the lifecycle wrapper."""

import numpy as np
import pytest

from hh_npe.simulator.lifecycle import build_lifecycle_agent, solve_lifecycle


@pytest.fixture(scope="module")
def small_solved_agent():
    a = build_lifecycle_agent(
        delta=0.96,
        crra=2.0,
        n_income_states=5,
        age_start=20,
        age_end=30,  # 10-period lifecycle for speed
    )
    return solve_lifecycle(a)


def test_solve_returns_terminal_plus_T_periods(small_solved_agent):
    # HARK returns T_cycle + 1 solution objects (terminal + T regular periods)
    assert len(small_solved_agent.solution) == 11


def test_cFunc_one_per_markov_state(small_solved_agent):
    sol0 = small_solved_agent.solution[0]
    assert hasattr(sol0, "cFunc")
    assert len(sol0.cFunc) == 5


def test_consumption_increasing_in_m(small_solved_agent):
    """For each Markov state, c(m) must be (weakly) increasing in m."""
    sol0 = small_solved_agent.solution[0]
    ms = np.linspace(0.1, 10.0, 50)
    for k in range(5):
        cs = np.array([float(sol0.cFunc[k](m)) for m in ms])
        diffs = np.diff(cs)
        assert (diffs >= -1e-6).all(), f"State {k} not monotone in m: {cs}"


def test_consumption_monotone_in_income_state(small_solved_agent):
    """At a fixed m, higher income state → at least as much consumption."""
    sol0 = small_solved_agent.solution[0]
    m = 2.0
    cs = np.array([float(sol0.cFunc[k](m)) for k in range(5)])
    diffs = np.diff(cs)
    assert (diffs >= -1e-6).all(), f"Non-monotone in income state: {cs}"


def test_more_patient_agents_consume_less_when_young():
    """Higher delta → save more → lower c at young ages, same m and state."""
    a_lo = solve_lifecycle(
        build_lifecycle_agent(
            delta=0.90, crra=2.0, n_income_states=3, age_start=20, age_end=30
        )
    )
    a_hi = solve_lifecycle(
        build_lifecycle_agent(
            delta=0.99, crra=2.0, n_income_states=3, age_start=20, age_end=30
        )
    )
    c_lo = float(a_lo.solution[0].cFunc[1](2.0))
    c_hi = float(a_hi.solution[0].cFunc[1](2.0))
    assert c_lo > c_hi, f"Expected patient consumer to consume less: lo={c_lo}, hi={c_hi}"


def test_stashed_arrays_accessible(small_solved_agent):
    """grid_vals, P_matrix, LivPrb_path are stashed for downstream use."""
    assert small_solved_agent.grid_vals.shape == (5,)
    assert small_solved_agent.P_matrix.shape == (5, 5)
    assert small_solved_agent.LivPrb_path.shape == (10,)
