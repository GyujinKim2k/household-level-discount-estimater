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


def test_standardization_survives_scale_that_layernorm_squashes():
    """A small signed feature beside a six-figure one must stay visible.

    Phase 3's ``liquid_assets`` (mean -$181) sits next to ``income`` ($101k).
    LayerNorm rescales across features within each wave, so the small feature's
    variation is divided by the large feature's magnitude. Fixed per-feature
    statistics keep the two on comparable footing.
    """
    torch.manual_seed(0)
    x = torch.stack([
        torch.full((16, 5), 100_000.0) + torch.randn(16, 5) * 1_000,  # income
        torch.randn(16, 5) * 200,                                      # liquid
    ], dim=-1)
    mean, std = x.mean(dim=(0, 1)), x.std(dim=(0, 1))

    m = TrajectoryTransformer(n_features=2, feature_mean=mean, feature_std=std)
    m.eval()
    with torch.no_grad():
        h = (x - m.feature_mean) / m.feature_std
    # Both features arrive at the projection with comparable spread.
    assert 0.5 < h[..., 1].std() / h[..., 0].std() < 2.0

    ln = TrajectoryTransformer(n_features=2)
    with torch.no_grad():
        h_ln = ln.input_norm(x)
    # LayerNorm collapses each wave to the same two values regardless of the
    # small feature's own variation, so it carries far less of it.
    assert h_ln[..., 1].std() < 0.1 * h[..., 1].std()


def test_standardization_buffers_round_trip_through_state_dict():
    """Stats must travel with the checkpoint, or inference normalizes differently."""
    mean, std = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    m = TrajectoryTransformer(n_features=2, feature_mean=mean, feature_std=std)
    clone = TrajectoryTransformer(
        n_features=2, feature_mean=torch.zeros(2), feature_std=torch.ones(2)
    )
    clone.load_state_dict(m.state_dict())
    torch.testing.assert_close(clone.feature_mean, mean)
    torch.testing.assert_close(clone.feature_std, std)

    m.eval(), clone.eval()
    x = torch.randn(3, 5, 2)
    with torch.no_grad():
        torch.testing.assert_close(m(x), clone(x))


def test_zero_variance_feature_does_not_divide_by_zero():
    m = TrajectoryTransformer(
        n_features=2, feature_mean=torch.zeros(2), feature_std=torch.tensor([1.0, 0.0])
    )
    assert torch.isfinite(m(torch.randn(4, 5, 2))).all()


def test_half_specified_stats_raise():
    with pytest.raises(ValueError, match="both be given"):
        TrajectoryTransformer(n_features=2, feature_mean=torch.zeros(2))


def test_wrong_stat_length_raises():
    with pytest.raises(ValueError, match="must have 2 entries"):
        TrajectoryTransformer(
            n_features=2, feature_mean=torch.zeros(3), feature_std=torch.ones(3)
        )


def test_custom_hyperparams():
    m = TrajectoryTransformer(
        n_features=4, seq_len=7, d_model=32, n_heads=2, n_layers=1, output_dim=16
    )
    out = m(torch.randn(2, 7, 4))
    assert out.shape == (2, 16)
