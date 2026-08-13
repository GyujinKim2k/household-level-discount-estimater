"""Tests for the observation-wave aggregation rule."""

import numpy as np
import pytest

from hh_npe.data.waves import FEATURES, aggregate_waves


def make_panel(N: int, T: int) -> dict[str, np.ndarray]:
    """Synthetic panel with predictable values for arithmetic checks."""
    return {
        "income":        np.arange(N * T, dtype=float).reshape(N, T),
        "consumption":   np.full((N, T), 0.5),
        "liquid_assets": (np.arange(N * T, dtype=float).reshape(N, T) + 1000.0),
        "t_age":         np.tile(np.arange(T), (N, 1)),
    }


def test_output_shape_and_dtype():
    """Defaults are the pre-registered choice: biennial, 5 waves."""
    panel = make_panel(N=4, T=20)
    x, alive = aggregate_waves(panel, age_start_sim=20, start_age=30)
    assert x.shape == (4, 5, 3)
    assert x.dtype == np.float32
    assert alive.shape == (4, 5)
    assert alive.dtype == bool


def test_features_ordering_constant():
    assert FEATURES == ("income", "consumption", "liquid_assets")


def test_annual_waves_read_panel_directly():
    """wave_years=1: flows are the single year, stock is that year's value."""
    panel = make_panel(N=2, T=20)
    x, _ = aggregate_waves(
        panel, age_start_sim=20, start_age=30, n_waves=10, wave_years=1
    )
    for w in range(10):
        t = 10 + w  # age 30+w
        assert x[0, w, 0] == pytest.approx(panel["income"][0, t])
        assert x[0, w, 1] == pytest.approx(panel["consumption"][0, t])
        assert x[0, w, 2] == pytest.approx(panel["liquid_assets"][0, t])


def test_biennial_flows_summed_over_two_year_window():
    panel = make_panel(N=2, T=20)
    x, _ = aggregate_waves(panel, age_start_sim=20, start_age=30, n_waves=2)
    # Wave 0 = ages 30-31 = annual indices 10, 11.
    expected_income_h0 = panel["income"][0, 10] + panel["income"][0, 11]
    expected_cons_h0 = panel["consumption"][0, 10] + panel["consumption"][0, 11]
    assert x[0, 0, 0] == pytest.approx(expected_income_h0)
    assert x[0, 0, 1] == pytest.approx(expected_cons_h0)


def test_biennial_stock_at_end_of_window():
    panel = make_panel(N=2, T=20)
    x, _ = aggregate_waves(panel, age_start_sim=20, start_age=30, n_waves=2)
    # Wave 0 end = annual index 11.
    assert x[0, 0, 2] == pytest.approx(panel["liquid_assets"][0, 11])
    # Wave 1 = ages 32-33 = indices 12, 13; end stock at 13.
    assert x[0, 1, 2] == pytest.approx(panel["liquid_assets"][0, 13])


def test_alive_all_true_when_no_reset():
    panel = make_panel(N=3, T=20)
    _, alive = aggregate_waves(panel, age_start_sim=20, start_age=30)
    assert alive.all()


def test_alive_false_when_t_age_resets_annual():
    panel = make_panel(N=2, T=20)
    # Household 0 dies after index 14 and is replaced at 15.
    panel["t_age"][0, 15] = 0
    panel["t_age"][0, 16] = 1
    panel["t_age"][0, 17] = 2
    _, alive = aggregate_waves(
        panel, age_start_sim=20, start_age=30, n_waves=10, wave_years=1
    )
    # Wave 5 = index 15; the 14→15 boundary check catches the reset.
    assert not alive[0, 5]
    # Wave 4 (index 14) and wave 6 (index 16, boundary 15→16 rises) unaffected.
    assert alive[0, 4]
    assert alive[0, 6]
    assert alive[1].all()


def test_alive_false_when_t_age_resets_biennial():
    panel = make_panel(N=2, T=20)
    panel["t_age"][0, 15] = 0
    panel["t_age"][0, 16] = 1
    panel["t_age"][0, 17] = 2
    _, alive = aggregate_waves(panel, age_start_sim=20, start_age=30, n_waves=5)
    # Wave 2 covers ages 34-35 (indices 14-15) — reset at 15 → dead.
    assert not alive[0, 2]
    assert alive[0, 1]
    # Wave 3 covers indices 16-17; pre-window check 15→16 is non-decreasing.
    assert alive[0, 3]
    assert alive[1].all()


def test_window_exceeds_simulator_range_raises():
    panel = make_panel(N=2, T=10)  # only 10 annual periods
    with pytest.raises(ValueError, match="exceeds"):
        aggregate_waves(panel, age_start_sim=20, start_age=30, n_waves=5)


def test_bad_wave_years_raises():
    panel = make_panel(N=2, T=20)
    with pytest.raises(ValueError, match="wave_years"):
        aggregate_waves(panel, age_start_sim=20, start_age=30, n_waves=2, wave_years=0)


def test_works_with_simulator_panel():
    """End-to-end sanity check with the real simulator."""
    from hh_npe.simulator.forward import simulate_households
    from hh_npe.simulator.lifecycle import build_lifecycle_agent, solve_lifecycle

    agent = solve_lifecycle(build_lifecycle_agent(
        delta=0.95, crra=2.0, n_income_states=5, age_start=20, age_end=40,
    ))
    panel = simulate_households(agent, n_households=8, seed=0)
    x, alive = aggregate_waves(panel, age_start_sim=20, start_age=30, n_waves=5)
    assert x.shape == (8, 5, 3)
    # With mortality between ages 30-39 around 0.001-0.002 annual,
    # most/all of 8 households should be fully alive.
    assert alive.any()


def test_twoasset_feature_set_adds_illiquid():
    from hh_npe.data.waves import FEATURES_MVP, FEATURES_TWOASSET

    assert FEATURES_MVP == ("income", "consumption", "liquid_assets")
    assert FEATURES_TWOASSET == FEATURES_MVP + ("illiquid_assets",)

    panel = make_panel(N=3, T=20)
    panel["illiquid_assets"] = np.full((3, 20), 7.0)
    x, _ = aggregate_waves(
        panel, age_start_sim=20, start_age=30, features=FEATURES_TWOASSET
    )
    assert x.shape == (3, 5, 4)
    # Illiquid balance is a stock: read at the end of the window, not summed.
    assert np.all(x[:, :, 3] == 7.0)


def test_stocks_are_not_summed_flows_are():
    """A 2-year window must sum flows and read stocks once."""
    from hh_npe.data.waves import FEATURES_TWOASSET

    panel = make_panel(N=1, T=20)
    panel["illiquid_assets"] = np.full((1, 20), 5.0)
    panel["consumption"] = np.full((1, 20), 3.0)
    x, _ = aggregate_waves(
        panel, age_start_sim=20, start_age=30, features=FEATURES_TWOASSET
    )
    assert x[0, 0, 1] == pytest.approx(6.0)  # consumption: flow, 2 x 3.0
    assert x[0, 0, 3] == pytest.approx(5.0)  # illiquid: stock, read once


def test_missing_feature_raises():
    from hh_npe.data.waves import FEATURES_TWOASSET

    panel = make_panel(N=2, T=20)  # no illiquid_assets key
    with pytest.raises(KeyError, match="illiquid_assets"):
        aggregate_waves(
            panel, age_start_sim=20, start_age=30, features=FEATURES_TWOASSET
        )
