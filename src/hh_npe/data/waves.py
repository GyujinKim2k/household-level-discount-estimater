"""Annual simulator output → fixed-length observation-wave trajectories.

Aggregation rule (per SIMULATOR_SPEC.md §6, pinned 2026-08-09):
- **Flows** (income, consumption): summed across the ``wave_years`` window.
- **Stocks** (liquid_assets): value at the end of the final year of the window.

``wave_years`` sets the observation frequency. All phases use
``wave_years=2`` (biennial), matching PSID's post-1997 observation schedule --
the empirical target in Phase 4. ``wave_years=1`` (annual) is supported and
tested, since Laibson et al.'s own moments are annual, but is not the
pre-registered choice.

Wave ``w`` spans annual indices ``[t_start + wave_years*w, t_start +
wave_years*(w+1))`` where ``t_start = start_age - age_start_sim``.

The function also returns an ``alive`` mask flagging waves during which HARK
replaced the household with a newborn (``t_age`` non-monotone within the
window). Callers typically discard households whose ``alive`` is False at any
wave.
"""

from __future__ import annotations

import numpy as np

#: Phase 1-2 feature set (HARK MVP: single liquid asset, no credit cards).
FEATURES_MVP: tuple[str, ...] = ("income", "consumption", "liquid_assets")

#: Phase 3 feature set. ``liquid_assets`` now goes negative (credit-card debt)
#: and ``illiquid_assets`` is observed. Eight of Laibson et al.'s sixteen target
#: moments are wealth *conditional on debt status*, so both asset series carry
#: identifying information -- dropping the illiquid balance would discard the
#: signal that present bias rides on.
FEATURES_TWOASSET: tuple[str, ...] = FEATURES_MVP + ("illiquid_assets",)

#: Phase 3 with the household's age attached to each wave. Needed once the
#: observation window stops being fixed at ages 30-39: consumption and asset
#: levels are strongly age-dependent, so a model shown a window without knowing
#: *when* it is looking has to marginalize over age rather than condition on
#: it. PSID always records the head's age, so conditioning is the honest
#: choice. ``age`` is not a flow, so it is read at the end of each wave like
#: any other stock.
FEATURES_TWOASSET_AGE: tuple[str, ...] = FEATURES_TWOASSET + ("age",)

#: Default remains the MVP set so Phase 1-2 behaviour is unchanged.
FEATURES = FEATURES_MVP

#: Variables summed over the window; everything else is a stock read at its end.
FLOWS = frozenset({"income", "consumption"})


def aggregate_waves(
    panel: dict[str, np.ndarray],
    age_start_sim: int,
    start_age: int,
    n_waves: int = 5,
    wave_years: int = 2,
    features: tuple[str, ...] = FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse an annual simulation panel to an ``(N, n_waves, n_features)`` tensor.

    Parameters
    ----------
    panel
        Output of :func:`hh_npe.simulator.forward.simulate_households`. Must
        contain keys ``income``, ``consumption``, ``liquid_assets``, ``t_age``.
    age_start_sim
        The simulator's first-period age (e.g. 20).
    start_age
        Age at which the first wave begins.
    n_waves
        Number of consecutive waves to extract.
    wave_years
        Years spanned by each wave. 2 = biennial (all phases, matches PSID);
        1 = annual (supported, not pre-registered).
    features
        Panel keys to extract, in order. :data:`FEATURES_MVP` (Phases 1-2) or
        :data:`FEATURES_TWOASSET` (Phase 3). Order is load-bearing: the
        embedder's input dimension is positional.

    Returns
    -------
    x : ndarray, shape ``(N, n_waves, len(features))``, dtype float32
        Feature order matches ``features``.
    alive : ndarray, shape ``(N, n_waves)``, dtype bool
        True if no ``t_age`` reset occurred in the window or in the preceding
        annual step (so the wave fully reflects one continuous household).
    """
    if wave_years < 1:
        raise ValueError(f"wave_years must be >= 1; got {wave_years}")

    missing = [f for f in features if f not in panel]
    if missing:
        raise KeyError(f"panel is missing feature(s) {missing}; has {sorted(panel)}")
    t_age = panel["t_age"]

    N, T = panel[features[0]].shape
    t_start = start_age - age_start_sim
    t_end = t_start + wave_years * n_waves
    if t_end > T:
        raise ValueError(
            f"Observation window ages {start_age}..{start_age + t_end - t_start - 1} "
            f"exceeds simulator range; have {T} annual periods, need {t_end}."
        )

    x = np.zeros((N, n_waves, len(features)), dtype=np.float32)
    alive = np.ones((N, n_waves), dtype=bool)

    for w in range(n_waves):
        t0 = t_start + wave_years * w
        t1 = t0 + wave_years  # exclusive
        for k, name in enumerate(features):
            series = panel[name]
            x[:, w, k] = (
                series[:, t0:t1].sum(axis=1) if name in FLOWS else series[:, t1 - 1]
            )

        # Rebirth detection: t_age must be monotonically non-decreasing across
        # the window (and across the boundary from the prior period if any).
        check_start = max(t0 - 1, 0)
        window_t_age = t_age[:, check_start:t1]
        diffs = np.diff(window_t_age, axis=1)
        alive[:, w] = (diffs >= 0).all(axis=1)

    return x, alive
