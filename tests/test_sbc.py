"""Tests for SBC diagnostics."""

import numpy as np
import torch

from hh_npe.evaluation.sbc import compute_ranks, coverage_at_level, plot_sbc_ranks


class FakeUniformPosterior:
    """Posterior that ignores ``x`` and samples from a fixed uniform box.

    With theta_true also drawn from the same uniform, ``compute_ranks`` should
    produce uniformly distributed ranks — a sanity check that the rank
    calculation is correct.
    """

    def __init__(self, low, high, seed: int = 0) -> None:
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        self.gen = torch.Generator().manual_seed(seed)

    def sample(self, shape, x=None, show_progress_bars=False):
        (n,) = shape
        d = self.low.shape[0]
        u = torch.rand(n, d, generator=self.gen)
        return self.low + u * (self.high - self.low)


class DeviceCheckingPosterior(FakeUniformPosterior):
    """Refuses a tensor that is not on ``_device``, the way sbi's net does.

    sbi does not move inputs; a CPU ``x`` against a CUDA net raises deep inside
    nflows. Every scoring path had assumed CPU, because until the dataset run
    finished every job was pinned to CPU with ``CUDA_VISIBLE_DEVICES=""``.
    """

    def __init__(self, low, high, device="cpu", **kw):
        super().__init__(low, high, **kw)
        self._device = device

    def sample(self, shape, x=None, show_progress_bars=False):
        if x is not None and x.device.type != torch.device(self._device).type:
            raise RuntimeError(
                f"Expected all tensors on {self._device}, got x on {x.device}"
            )
        return super().sample(shape, x, show_progress_bars)

    def log_prob(self, theta, x=None):
        for t in (theta, x):
            if t is not None and t.device.type != torch.device(self._device).type:
                raise RuntimeError(f"tensor on {t.device}, net on {self._device}")
        return torch.zeros(len(theta))


def test_posterior_device_defaults_to_cpu():
    from hh_npe.evaluation.sbc import posterior_device

    assert posterior_device(FakeUniformPosterior([0.0], [1.0])).type == "cpu"
    assert posterior_device(
        DeviceCheckingPosterior([0.0], [1.0], device="cuda")).type == "cuda"


def test_compute_ranks_moves_inputs_to_the_posterior_device():
    post = DeviceCheckingPosterior(low=[0.0, 0.0], high=[1.0, 1.0], device="cpu")
    ranks = compute_ranks(post, torch.rand(8, 2), torch.zeros(8, 5, 3),
                          n_posterior_samples=20)
    assert ranks.shape == (8, 2)


def test_scoring_moves_inputs_to_the_posterior_device():
    """The crash was here: CPU held-out tensors into a CUDA-trained posterior."""
    from hh_npe.evaluation.scoring import estimation_scores
    from hh_npe.npe.prior import PriorBox

    box = PriorBox()
    post = DeviceCheckingPosterior(low=box.low, high=box.high, device="cpu")
    per_param, log_q = estimation_scores(
        post, box, torch.rand(6, box.n_params), torch.zeros(6, 5, 4), n_post=20
    )
    assert set(per_param) == set(box.names)
    assert np.isfinite(log_q)


def test_compute_ranks_shape_and_bounds():
    post = FakeUniformPosterior(low=[0.0, 0.0], high=[1.0, 1.0])
    thetas = torch.rand(20, 2)
    xs = torch.zeros(20, 5, 3)
    ranks = compute_ranks(post, thetas, xs, n_posterior_samples=50)
    assert ranks.shape == (20, 2)
    assert (ranks >= 0).all()
    assert (ranks <= 50).all()


def test_ranks_uniform_when_posterior_matches_prior():
    """If both true and posterior are Uniform(0,1), ranks should be ~ uniform."""
    post = FakeUniformPosterior(low=[0.0], high=[1.0], seed=42)
    rng = np.random.default_rng(42)
    n_sim, n_post = 500, 100
    thetas = torch.from_numpy(rng.uniform(0, 1, (n_sim, 1))).float()
    xs = torch.zeros(n_sim, 5, 3)
    ranks = compute_ranks(post, thetas, xs, n_posterior_samples=n_post)
    assert abs(ranks.mean() - n_post / 2) < 5
    hist, _ = np.histogram(ranks, bins=10)
    expected = n_sim / 10
    assert (hist > expected / 2).all()
    assert (hist < expected * 2).all()


def test_coverage_at_90pct_near_90():
    post = FakeUniformPosterior(low=[0.0], high=[1.0], seed=42)
    rng = np.random.default_rng(42)
    n_sim, n_post = 1000, 100
    thetas = torch.from_numpy(rng.uniform(0, 1, (n_sim, 1))).float()
    xs = torch.zeros(n_sim, 5, 3)
    ranks = compute_ranks(post, thetas, xs, n_posterior_samples=n_post)
    cov = coverage_at_level(ranks, n_post, level=0.9)
    assert abs(cov[0] - 0.9) < 0.05


def test_overconfident_posterior_fails_coverage():
    """Posterior concentrated near 0.5 won't cover the unit interval uniformly."""

    class Overconfident:
        def sample(self, shape, x=None, show_progress_bars=False):
            (n,) = shape
            return 0.5 + 0.01 * torch.randn(n, 1)

    rng = np.random.default_rng(0)
    n_sim, n_post = 500, 100
    thetas = torch.from_numpy(rng.uniform(0, 1, (n_sim, 1))).float()
    xs = torch.zeros(n_sim, 5, 3)
    ranks = compute_ranks(Overconfident(), thetas, xs, n_posterior_samples=n_post)
    cov = coverage_at_level(ranks, n_post, level=0.9)
    assert cov[0] < 0.5, f"overconfident posterior should miscover: cov={cov[0]}"


def test_plot_creates_file(tmp_path):
    ranks = np.random.randint(0, 100, size=(80, 2))
    p = tmp_path / "sbc.png"
    plot_sbc_ranks(ranks, 100, ["delta", "crra"], p)
    assert p.exists()
    assert p.stat().st_size > 0
