"""Tests for the biennial aggregation rule."""

import numpy as np
import pytest

from hh_npe.data.biennial import FEATURES, aggregate_biennial


def make_panel(N: int, T: int) -> dict[str, np.ndarray]:
    """Synthetic panel with predictable values for arithmetic checks."""
    return {
        "income":        np.arange(N * T, dtype=float).reshape(N, T),
        "consumption":   np.full((N, T), 0.5),
        "liquid_assets": (np.arange(N * T, dtype=float).reshape(N, T) + 1000.0),
        "t_age":         np.tile(np.arange(T), (N, 1)),
    }


def test_output_shape_and_dtype():
    panel = make_panel(N=4, T=20)
    x, alive = aggregate_biennial(panel, age_start_sim=20, start_age=30, n_waves=5)
    assert x.shape == (4, 5, 3)
    assert x.dtype == np.float32
    assert alive.shape == (4, 5)
    assert alive.dtype == bool


def test_features_ordering_constant():
    assert FEATURES == ("income", "consumption", "liquid_assets")


def test_flows_summed_over_two_year_window():
    panel = make_panel(N=2, T=20)
    x, _ = aggregate_biennial(panel, age_start_sim=20, start_age=30, n_waves=2)
    # Wave 0 = ages 30-31 = annual indices 10, 11.
    expected_income_h0 = panel["income"][0, 10] + panel["income"][0, 11]
    expected_cons_h0 = panel["consumption"][0, 10] + panel["consumption"][0, 11]
    assert x[0, 0, 0] == pytest.approx(expected_income_h0)
    assert x[0, 0, 1] == pytest.approx(expected_cons_h0)


def test_stock_at_end_of_window():
    panel = make_panel(N=2, T=20)
    x, _ = aggregate_biennial(panel, age_start_sim=20, start_age=30, n_waves=2)
    # Wave 0 end = annual index 11.
    assert x[0, 0, 2] == pytest.approx(panel["liquid_assets"][0, 11])
    # Wave 1 = ages 32-33 = indices 12, 13; end stock at 13.
    assert x[0, 1, 2] == pytest.approx(panel["liquid_assets"][0, 13])


def test_alive_all_true_when_no_reset():
    panel = make_panel(N=3, T=20)
    _, alive = aggregate_biennial(panel, age_start_sim=20, start_age=30, n_waves=5)
    assert alive.all()


def test_alive_false_when_t_age_resets_in_window():
    panel = make_panel(N=2, T=20)
    # Household 0 dies after index 14 and is replaced at 15.
    panel["t_age"][0, 15] = 0
    panel["t_age"][0, 16] = 1
    panel["t_age"][0, 17] = 2
    _, alive = aggregate_biennial(panel, age_start_sim=20, start_age=30, n_waves=5)
    # Wave 2 covers ages 34-35 (indices 14-15) — reset at 15 → dead.
    assert alive[0, 2] is np.bool_(False) or not alive[0, 2]
    # Wave 1 (indices 12-13) unaffected.
    assert alive[0, 1]
    # Wave 3 covers ages 36-37 (indices 16-17). The pre-window check spans
    # index 15→16: 0→1 is non-decreasing → counts as alive.
    assert alive[0, 3]
    # Household 1 completely unaffected.
    assert alive[1].all()


def test_window_exceeds_simulator_range_raises():
    panel = make_panel(N=2, T=10)  # only 10 annual periods
    with pytest.raises(ValueError, match="exceeds"):
        aggregate_biennial(panel, age_start_sim=20, start_age=30, n_waves=5)


def test_works_with_simulator_panel():
    """End-to-end sanity check with the real simulator."""
    from hh_npe.simulator.forward import simulate_households
    from hh_npe.simulator.lifecycle import build_lifecycle_agent, solve_lifecycle

    agent = solve_lifecycle(build_lifecycle_agent(
        delta=0.95, crra=2.0, n_income_states=5, age_start=20, age_end=40,
    ))
    panel = simulate_households(agent, n_households=8, seed=0)
    x, alive = aggregate_biennial(panel, age_start_sim=20, start_age=30, n_waves=5)
    assert x.shape == (8, 5, 3)
    # With mortality between ages 30-39 around 0.001-0.002 annual,
    # most/all of 8 households should be fully alive.
    assert alive.any()
