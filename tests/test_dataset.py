"""Tests for dataset save/load helpers."""

import pytest
import torch

from hh_npe.data.dataset import load_dataset, save_dataset


def test_roundtrip(tmp_path):
    theta = torch.randn(10, 2)
    x = torch.randn(10, 5, 3)
    p = tmp_path / "d.pt"
    save_dataset(theta, x, p)
    t2, x2 = load_dataset(p)
    torch.testing.assert_close(theta.float(), t2)
    torch.testing.assert_close(x.float(), x2)


def test_creates_parent_dirs(tmp_path):
    theta = torch.randn(3, 2)
    x = torch.randn(3, 5, 3)
    p = tmp_path / "a" / "b" / "d.pt"
    save_dataset(theta, x, p)
    assert p.exists()


def test_mismatched_n_samples_raises(tmp_path):
    theta = torch.randn(5, 2)
    x = torch.randn(6, 5, 3)
    with pytest.raises(ValueError, match="n_samples"):
        save_dataset(theta, x, tmp_path / "d.pt")


def test_dtype_coerced_to_float32(tmp_path):
    theta = torch.randn(4, 2, dtype=torch.float64)
    x = torch.randn(4, 5, 3, dtype=torch.float64)
    p = tmp_path / "d.pt"
    save_dataset(theta, x, p)
    t2, x2 = load_dataset(p)
    assert t2.dtype == torch.float32
    assert x2.dtype == torch.float32
