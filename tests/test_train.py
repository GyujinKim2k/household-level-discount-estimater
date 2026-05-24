"""Smoke tests for npe.train — verify the sbi integration runs end-to-end."""

import pytest
import torch

from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PriorBox, sample_sobol
from hh_npe.npe.train import load_posterior, save_posterior, train_npe


@pytest.fixture(scope="module")
def synthetic_dataset():
    """Tiny synthetic (theta, x) pair: x weakly correlated with theta so NPE has signal."""
    torch.manual_seed(0)
    box = PriorBox()
    theta_np = sample_sobol(128, box, seed=0)  # power of 2
    theta = torch.from_numpy(theta_np).float()
    n = theta.shape[0]
    x = torch.randn(n, 5, 3)
    # Inject signal so the NPE has something to fit
    x[:, :, 0] += theta[:, 0:1] * 5.0       # income correlates with delta
    x[:, :, 1] += theta[:, 1:2] * 0.5       # consumption correlates with crra
    return theta, x


@pytest.fixture(scope="module")
def tiny_trained(synthetic_dataset):
    theta, x = synthetic_dataset
    embedder = TrajectoryTransformer(d_model=16, n_heads=2, n_layers=1, output_dim=8)
    posterior, _, _ = train_npe(
        theta, x,
        embedder=embedder,
        hidden_features=16,
        num_transforms=2,
        max_num_epochs=5,
        stop_after_epochs=5,
        batch_size=32,
        show_progress=False,
    )
    return posterior, embedder


def test_train_returns_samplable_posterior(tiny_trained, synthetic_dataset):
    posterior, _ = tiny_trained
    _, x = synthetic_dataset
    samples = posterior.sample((50,), x=x[0], show_progress_bars=False)
    assert samples.shape == (50, 2)


def test_posterior_samples_within_prior_box(tiny_trained, synthetic_dataset):
    posterior, _ = tiny_trained
    _, x = synthetic_dataset
    samples = posterior.sample((100,), x=x[0], show_progress_bars=False)
    box = PriorBox()
    # Allow small slack for flow-density support tail
    assert (samples[:, 0] >= box.delta_low - 0.01).all()
    assert (samples[:, 0] <= box.delta_high + 0.01).all()
    assert (samples[:, 1] >= box.crra_low - 0.05).all()
    assert (samples[:, 1] <= box.crra_high + 0.05).all()


def test_save_load_roundtrip(tiny_trained, tmp_path):
    posterior, embedder = tiny_trained
    p = tmp_path / "posterior.pt"
    save_posterior(posterior, embedder, PriorBox(), p)
    loaded = load_posterior(p)
    assert "posterior" in loaded
    assert "embedder_state_dict" in loaded
    assert "box" in loaded
    assert isinstance(loaded["box"], PriorBox)
