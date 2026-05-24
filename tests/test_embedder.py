"""Tests for the Transformer trajectory embedder."""

import pytest
import torch

from hh_npe.npe.embedder import TrajectoryTransformer


def test_default_output_shape():
    m = TrajectoryTransformer()
    x = torch.randn(8, 5, 3)
    assert m(x).shape == (8, 32)


def test_default_param_count_in_target_range():
    """``small`` for our purposes: 2 layers, d=64, 4 heads → ~100k params.

    The dominant cost is dim_feedforward = 4*d_model = 256, giving ~50k per
    encoder layer. Bound chosen to flag accidental architecture inflation.
    """
    m = TrajectoryTransformer()
    n = sum(p.numel() for p in m.parameters())
    assert 50_000 < n < 200_000, f"unexpected param count: {n}"


def test_eval_mode_is_batch_invariant():
    m = TrajectoryTransformer()
    m.eval()
    x = torch.randn(4, 5, 3)
    with torch.no_grad():
        full = m(x)
        for i in range(4):
            single = m(x[i : i + 1])
            torch.testing.assert_close(full[i : i + 1], single, rtol=1e-5, atol=1e-6)


def test_gradients_flow_to_all_parameters():
    m = TrajectoryTransformer()
    x = torch.randn(4, 5, 3)
    m(x).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"


def test_wrong_seq_len_raises():
    m = TrajectoryTransformer()
    with pytest.raises(ValueError, match="Expected"):
        m(torch.randn(8, 4, 3))


def test_wrong_n_features_raises():
    m = TrajectoryTransformer()
    with pytest.raises(ValueError, match="Expected"):
        m(torch.randn(8, 5, 4))


def test_2d_input_raises():
    m = TrajectoryTransformer()
    with pytest.raises(ValueError, match="3D"):
        m(torch.randn(5, 3))


def test_custom_hyperparams():
    m = TrajectoryTransformer(
        n_features=4, seq_len=7, d_model=32, n_heads=2, n_layers=1, output_dim=16
    )
    out = m(torch.randn(2, 7, 4))
    assert out.shape == (2, 16)
