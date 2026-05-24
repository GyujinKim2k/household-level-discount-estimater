"""Tests for the centralized seeding utility."""

import random

import numpy as np
import torch

from hh_npe.utils.seeding import seed_all


def test_numpy_reproducible():
    seed_all(42)
    a = np.random.randn(5)
    seed_all(42)
    b = np.random.randn(5)
    np.testing.assert_array_equal(a, b)


def test_torch_reproducible():
    seed_all(42)
    a = torch.randn(5)
    seed_all(42)
    b = torch.randn(5)
    torch.testing.assert_close(a, b)


def test_python_random_reproducible():
    seed_all(42)
    a = [random.random() for _ in range(5)]
    seed_all(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_different_seeds_diverge():
    seed_all(42)
    a = np.random.randn(5)
    seed_all(43)
    b = np.random.randn(5)
    assert not np.allclose(a, b)
