"""Tests for Rouwenhorst AR(1) discretization."""

import numpy as np
import pytest

from hh_npe.simulator.income import rouwenhorst, stationary_distribution


def test_row_stochastic():
    _, P = rouwenhorst(n_states=7, rho=0.95, sigma=0.15)
    assert P.shape == (7, 7)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert (P >= 0).all()


def test_grid_symmetric_about_mu():
    grid, _ = rouwenhorst(n_states=7, rho=0.95, sigma=0.15, mu=2.0)
    np.testing.assert_allclose(grid.mean(), 2.0)
    np.testing.assert_allclose(grid[0] + grid[-1], 2.0 * 2.0)


def test_stationary_moments_match_ar1():
    """Stationary mean = mu and variance = sigma^2 / (1 - rho^2).

    Rouwenhorst matches both exactly (this is its defining property —
    unlike Tauchen, which only matches them asymptotically in n_states).
    """
    n, rho, sigma, mu = 11, 0.95, 0.15, 0.5
    grid, P = rouwenhorst(n, rho, sigma, mu=mu)
    pi = stationary_distribution(P)
    mean = (pi * grid).sum()
    var = (pi * (grid - mean) ** 2).sum()
    np.testing.assert_allclose(mean, mu, atol=1e-10)
    np.testing.assert_allclose(var, sigma ** 2 / (1.0 - rho ** 2), rtol=1e-8)


def test_n_states_2_works():
    grid, P = rouwenhorst(n_states=2, rho=0.5, sigma=0.2)
    assert grid.shape == (2,)
    assert P.shape == (2, 2)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_high_persistence_diagonal_dominant():
    _, P = rouwenhorst(n_states=5, rho=0.99, sigma=0.05)
    assert (np.diag(P) > 0.9).all()


def test_invalid_inputs():
    with pytest.raises(ValueError, match="n_states"):
        rouwenhorst(n_states=1, rho=0.5, sigma=0.1)
    with pytest.raises(ValueError, match="rho"):
        rouwenhorst(n_states=5, rho=1.0, sigma=0.1)
    with pytest.raises(ValueError, match="sigma"):
        rouwenhorst(n_states=5, rho=0.5, sigma=0.0)


def test_stationary_is_a_fixed_point():
    """Verify pi @ P == pi numerically."""
    _, P = rouwenhorst(n_states=9, rho=0.97, sigma=0.12)
    pi = stationary_distribution(P)
    np.testing.assert_allclose(pi @ P, pi, atol=1e-10)
    np.testing.assert_allclose(pi.sum(), 1.0)
