"""Sobol-based prior sampler for (delta, crra), plus an sbi-compatible BoxUniform.

The Sobol sequence is used to draw parameter samples that we'll feed through
the simulator to generate ``(theta, x)`` training pairs. The companion
``BoxUniform`` provides the log-density that sbi needs for NPE training. Both
correspond to the uniform distribution on the same box, so density evaluations
remain valid even though sampling is quasi-random rather than IID.

Power-of-2 sample sizes (``n_samples = 2**m``) preserve Sobol's low-discrepancy
guarantees most cleanly; non-power-of-2 sizes work but lose some of the
sequence's balance properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import qmc

if TYPE_CHECKING:
    from sbi.utils import BoxUniform


@dataclass(frozen=True)
class PriorBox:
    """Box bounds on (delta, crra). Pinned values match SIMULATOR_SPEC.md."""

    delta_low: float = 0.85
    delta_high: float = 1.00
    crra_low: float = 0.5
    crra_high: float = 5.0

    @property
    def low(self) -> np.ndarray:
        return np.array([self.delta_low, self.crra_low])

    @property
    def high(self) -> np.ndarray:
        return np.array([self.delta_high, self.crra_high])

    @property
    def names(self) -> tuple[str, str]:
        return ("delta", "crra")


def sample_sobol(
    n_samples: int,
    box: PriorBox = PriorBox(),
    seed: int = 0,
) -> np.ndarray:
    """Draw ``n_samples`` quasi-random points from the uniform prior on ``box``.

    Uses a scrambled Sobol sequence (``scipy.stats.qmc.Sobol``); returns an
    array of shape ``(n_samples, 2)`` with columns ``[delta, crra]``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    engine = qmc.Sobol(d=2, scramble=True, seed=seed)
    u = engine.random(n_samples)
    return qmc.scale(u, box.low, box.high)


def make_sbi_prior(box: PriorBox = PriorBox()) -> "BoxUniform":
    """Return an sbi-compatible ``BoxUniform`` prior on the same ``box``.

    Used by sbi during NPE training for log-density evaluation. Sampling from
    this object is pseudo-random (not Sobol); use :func:`sample_sobol` to
    generate training points and pass them via sbi's pre-existing-samples
    interface.
    """
    import torch
    from sbi.utils import BoxUniform

    return BoxUniform(
        low=torch.tensor(box.low, dtype=torch.float32),
        high=torch.tensor(box.high, dtype=torch.float32),
    )
