"""Tests for the Sobol-based prior sampler."""

import numpy as np
import pytest

from hh_npe.npe.prior import PriorBox, make_sbi_prior, sample_sobol


def test_sample_shape_and_bounds():
    box = PriorBox()
    s = sample_sobol(64, box)
    assert s.shape == (64, 2)
    assert (s[:, 0] >= box.delta_low).all() and (s[:, 0] <= box.delta_high).all()
    assert (s[:, 1] >= box.crra_low).all() and (s[:, 1] <= box.crra_high).all()


def test_sobol_beats_iid_uniform_in_discrepancy():
    """Sobol's 1D marginal empirical CDF should be closer to uniform than IID."""
    box = PriorBox()
    n = 256
    sobol_s = sample_sobol(n, box, seed=0)
    rng = np.random.default_rng(0)
    iid_s = np.column_stack([
        rng.uniform(box.delta_low, box.delta_high, n),
        rng.uniform(box.crra_low, box.crra_high, n),
    ])

    def cdf_l1(samples, lo, hi):
        u = (np.sort(samples) - lo) / (hi - lo)
        return np.mean(np.abs(u - np.arange(1, n + 1) / n))

    sobol_err = cdf_l1(sobol_s[:, 0], box.delta_low, box.delta_high)
    iid_err = cdf_l1(iid_s[:, 0], box.delta_low, box.delta_high)
    assert sobol_err < iid_err, f"sobol={sobol_err:.5f}, iid={iid_err:.5f}"


def test_reproducible_with_seed():
    a = sample_sobol(32, seed=42)
    b = sample_sobol(32, seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_diverge():
    a = sample_sobol(32, seed=0)
    b = sample_sobol(32, seed=1)
    assert not np.allclose(a, b)


def test_invalid_n_raises():
    with pytest.raises(ValueError, match="n_samples"):
        sample_sobol(0)


def test_custom_box_respected():
    box = PriorBox(delta_low=0.5, delta_high=0.6, crra_low=1.0, crra_high=2.0)
    s = sample_sobol(64, box)
    assert (s[:, 0] >= 0.5).all() and (s[:, 0] <= 0.6).all()
    assert (s[:, 1] >= 1.0).all() and (s[:, 1] <= 2.0).all()


def test_sbi_prior_constructs_and_evaluates():
    import torch

    p = make_sbi_prior()
    s = p.sample((10,))
    assert s.shape == (10, 2)
    lp = p.log_prob(s)
    assert torch.isfinite(lp).all()


def test_two_param_box_is_the_default():
    box = PriorBox()
    assert box.names == ("delta", "crra")
    assert box.n_params == 2
    assert not box.estimates_beta


def test_beta_box_puts_beta_first():
    """Laibson et al. order prefs as [beta delta rho]; we match that."""
    box = PriorBox(beta_low=0.30, beta_high=1.00)
    assert box.names == ("beta", "delta", "crra")
    assert box.n_params == 3
    np.testing.assert_allclose(box.low, [0.30, 0.85, 0.5])
    np.testing.assert_allclose(box.high, [1.00, 1.00, 5.0])


def test_sample_three_params_within_bounds():
    from hh_npe.npe.prior import PHASE3

    s = sample_sobol(128, PHASE3)
    assert s.shape == (128, 3)
    assert (s >= PHASE3.low).all() and (s <= PHASE3.high).all()


def test_sbi_prior_three_params():
    import torch

    from hh_npe.npe.prior import PHASE3

    p = make_sbi_prior(PHASE3)
    s = p.sample((10,))
    assert s.shape == (10, 3)
    assert torch.isfinite(p.log_prob(s)).all()


def test_half_specified_beta_raises():
    with pytest.raises(ValueError, match="both be set"):
        PriorBox(beta_low=0.3)
    with pytest.raises(ValueError, match="both be set"):
        PriorBox(beta_high=1.0)


def test_inverted_bounds_raise():
    with pytest.raises(ValueError, match="low < high"):
        PriorBox(delta_low=1.0, delta_high=0.85)
    with pytest.raises(ValueError, match="low < high"):
        PriorBox(beta_low=1.0, beta_high=0.3)
