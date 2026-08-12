"""Annual simulator output → biennial 5-wave trajectories.

Aggregation rule (per SIMULATOR_SPEC.md §6, pinned 2026-08-09):
- **Flows** (income, consumption): summed across the 2-year window.
- **Stocks** (liquid_assets): value at the end of the second year of the window.

A 5-wave biennial trajectory starting at age 30 covers ages 30-39 (annual
indices 10-19 relative to ``age_start_sim=20``). Wave w spans annual indices
``[t_start + 2*w, t_start + 2*w + 1]``.

The function also returns an ``alive`` mask flagging waves during which HARK
replaced the household with a newborn (``t_age`` non-monotone within the
window). Callers typically discard households whose ``alive`` is False at any
wave.
"""

from __future__ import annotations

import numpy as np

FEATURES: tuple[str, str, str] = ("income", "consumption", "liquid_assets")


def aggregate_biennial(
    panel: dict[str, np.ndarray],
    age_start_sim: int,
    start_age: int,
    n_waves: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse an annual simulation panel to a biennial ``(N, n_waves, 3)`` tensor.

    Parameters
    ----------
    panel
        Output of :func:`hh_npe.simulator.forward.simulate_households`. Must
        contain keys ``income``, ``consumption``, ``liquid_assets``, ``t_age``.
    age_start_sim
        The simulator's first-period age (e.g. 20).
    start_age
        Age at which the first biennial wave begins.
    n_waves
        Number of consecutive biennial waves to extract.

    Returns
    -------
    x : ndarray, shape ``(N, n_waves, 3)``, dtype float32
        Feature order matches :data:`FEATURES`.
    alive : ndarray, shape ``(N, n_waves)``, dtype bool
        True if no ``t_age`` reset occurred in the window or in the preceding
        annual step (so the wave fully reflects one continuous household).
    """
    income = panel["income"]
    cons = panel["consumption"]
    liq = panel["liquid_assets"]
    t_age = panel["t_age"]

    N, T = income.shape
    t_start = start_age - age_start_sim
    t_end = t_start + 2 * n_waves
    if t_end > T:
        raise ValueError(
            f"Biennial window ages {start_age}..{start_age + 2 * n_waves - 1} "
            f"exceeds simulator range; have {T} annual periods, need {t_end}."
        )

    x = np.zeros((N, n_waves, 3), dtype=np.float32)
    alive = np.ones((N, n_waves), dtype=bool)

    for w in range(n_waves):
        t0 = t_start + 2 * w
        t1 = t0 + 1
        x[:, w, 0] = income[:, t0] + income[:, t1]
        x[:, w, 1] = cons[:, t0] + cons[:, t1]
        x[:, w, 2] = liq[:, t1]

        # Rebirth detection: t_age must be monotonically non-decreasing across
        # the window (and across the boundary from the prior period if any).
        check_start = max(t0 - 1, 0)
        window_t_age = t_age[:, check_start : t1 + 1]
        diffs = np.diff(window_t_age, axis=1)
        alive[:, w] = (diffs >= 0).all(axis=1)

    return x, alive
