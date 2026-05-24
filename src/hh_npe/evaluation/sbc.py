"""Simulation-Based Calibration (Talts et al. 2018).

For an amortized NPE posterior to be well-calibrated:

1. Draw ``theta_i`` from the prior.
2. Simulate ``x_i`` from ``theta_i``.
3. Draw ``N`` posterior samples ``theta^*`` from ``p(theta | x_i)``.
4. Compute the rank of ``theta_i`` among the ``N`` posterior samples
   (per parameter dimension).

Across many SBC simulations the ranks must be **uniform on [0, N]**. Bumps
near the edges indicate posterior overconfidence; bumps in the middle
indicate underconfidence. Talts et al. recommend ``n_simulations >= 1000``
for crisp diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch


def compute_ranks(
    posterior,
    thetas: torch.Tensor,
    xs: torch.Tensor,
    n_posterior_samples: int = 1000,
) -> np.ndarray:
    """Per ``(theta_i, x_i)``, compute the rank of ``theta_i`` in posterior samples.

    Returns an array of shape ``(n_sim, n_params)`` with integer values in
    ``[0, n_posterior_samples]``.
    """
    if thetas.shape[0] != xs.shape[0]:
        raise ValueError(
            f"thetas and xs must have the same n_sim; got {thetas.shape[0]} vs {xs.shape[0]}"
        )
    n_sim, n_params = thetas.shape
    ranks = np.zeros((n_sim, n_params), dtype=int)
    for i in range(n_sim):
        samples = posterior.sample(
            (n_posterior_samples,), x=xs[i], show_progress_bars=False
        )
        for d in range(n_params):
            ranks[i, d] = int((samples[:, d] < thetas[i, d]).sum())
    return ranks


def run_sbc(
    posterior,
    prior,
    simulator: Callable[[torch.Tensor], torch.Tensor],
    n_simulations: int = 1000,
    n_posterior_samples: int = 1000,
    seed: int = 0,
) -> dict:
    """Draw ``thetas`` from ``prior``, simulate ``xs``, compute ranks.

    Returns a dict with ``thetas``, ``xs``, ``ranks``, and ``n_posterior_samples``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    thetas = prior.sample((n_simulations,))
    xs = simulator(thetas)
    ranks = compute_ranks(posterior, thetas, xs, n_posterior_samples)
    return {
        "thetas": thetas,
        "xs": xs,
        "ranks": ranks,
        "n_posterior_samples": n_posterior_samples,
    }


def coverage_at_level(
    ranks: np.ndarray,
    n_posterior_samples: int,
    level: float = 0.9,
) -> np.ndarray:
    """Empirical coverage of the level-credible interval (per parameter).

    Should equal ``level`` within sampling tolerance when the posterior is
    calibrated. Returns shape ``(n_params,)``.
    """
    alpha = (1.0 - level) / 2.0
    low = int(alpha * n_posterior_samples)
    high = int((1.0 - alpha) * n_posterior_samples)
    return ((ranks >= low) & (ranks <= high)).mean(axis=0)


def plot_sbc_ranks(
    ranks: np.ndarray,
    n_posterior_samples: int,
    param_names: list[str],
    output_path: str | Path,
    n_bins: int = 20,
) -> None:
    """Save a 2-row figure per parameter: rank histogram + ECDF vs. uniform."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sim, n_params = ranks.shape
    if len(param_names) != n_params:
        raise ValueError(
            f"param_names length {len(param_names)} != ranks n_params {n_params}"
        )

    fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 6), squeeze=False)

    for d in range(n_params):
        ax = axes[0, d]
        ax.hist(ranks[:, d], bins=n_bins, edgecolor="k")
        ax.axhline(n_sim / n_bins, color="r", linestyle="--", label="uniform")
        ax.set_title(f"{param_names[d]} rank histogram")
        ax.set_xlabel("rank")
        ax.legend(fontsize=8)

        ax = axes[1, d]
        sorted_u = np.sort(ranks[:, d]) / n_posterior_samples
        emp_quantiles = np.linspace(0, 1, n_sim, endpoint=True)
        ax.plot(emp_quantiles, sorted_u, label="empirical")
        ax.plot([0, 1], [0, 1], "r--", label="uniform")
        ax.set_title(f"{param_names[d]} ECDF")
        ax.set_xlabel("quantile")
        ax.legend(fontsize=8)

    fig.tight_layout()
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=100)
    plt.close(fig)
