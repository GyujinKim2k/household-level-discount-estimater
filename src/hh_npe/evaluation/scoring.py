"""Posterior scoring: how sharp it is, and whether the sharpness is earned.

Two questions that must not be confused.

**Estimation** -- contraction, correlation between truth and posterior mean,
mean absolute error, and held-out ``log q(theta_true | x)``. These say how much
the posterior narrows the prior and how well its centre tracks the truth.

**Calibration** -- SBC rank uniformity (Talts et al. 2018) and empirical
coverage of the 90% credible interval. These say whether the stated uncertainty
is honest. A posterior can contract beautifully and be confidently wrong;
contraction alone cannot tell the difference, which is why both live here and
get reported together.

Shared by ``scripts/compare_windows.py`` and ``scripts/pilot_random_age.py`` so
the two report the same numbers computed the same way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import stats

from hh_npe.evaluation.sbc import compute_ranks, coverage_at_level, plot_sbc_ranks
from hh_npe.npe.prior import PriorBox


def estimation_scores(
    post, box: PriorBox, theta: torch.Tensor, x: torch.Tensor, n_post: int = 400
) -> tuple[dict, float]:
    """Per-parameter sharpness and accuracy, plus mean held-out log density."""
    prior_var = ((box.high - box.low) ** 2) / 12.0
    means, sds, lps = [], [], []
    for i in range(len(theta)):
        s = post.sample((n_post,), x=x[i], show_progress_bars=False)
        means.append(s.mean(0))
        sds.append(s.std(0))
        lps.append(post.log_prob(theta[i][None], x=x[i]).item())
    means, sds = torch.stack(means).numpy(), torch.stack(sds).numpy()
    truth = theta.numpy()
    per_param = {
        n: {
            "contraction": float(1 - (sds[:, j] ** 2).mean() / prior_var[j]),
            "corr": float(np.corrcoef(truth[:, j], means[:, j])[0, 1]),
            "mae": float(np.abs(means[:, j] - truth[:, j]).mean()),
        }
        for j, n in enumerate(box.names)
    }
    return per_param, float(np.mean(lps))


def calibration_scores(
    post, box: PriorBox, theta: torch.Tensor, x: torch.Tensor,
    n_post: int = 1000, out_dir: Path | None = None, tag: str = "",
) -> dict:
    """SBC ranks -> uniformity p-value and 90% credible-interval coverage."""
    ranks = compute_ranks(post, theta, x, n_posterior_samples=n_post)
    cov = coverage_at_level(ranks, n_post, level=0.9)
    per_param = {}
    for j, n in enumerate(box.names):
        # Under calibration the ranks are uniform on {0..n_post}; compare the
        # normalized ranks to Uniform(0,1). A small p-value is miscalibration.
        u = (ranks[:, j] + 0.5) / (n_post + 1)
        ks = stats.kstest(u, "uniform")
        per_param[n] = {"coverage_90": float(cov[j]),
                        "ks_p": float(ks.pvalue),
                        "ks_stat": float(ks.statistic)}
    if out_dir is not None:
        plot_sbc_ranks(ranks, n_post, list(box.names), out_dir / f"sbc_ranks_{tag}.png")
        np.savez(out_dir / f"sbc_ranks_{tag}.npz", ranks=ranks, coverage_90=cov)
    return per_param
