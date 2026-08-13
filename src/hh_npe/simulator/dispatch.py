"""One-household trajectory generation, shared by dataset generation and SBC.

Both ``scripts/generate_dataset.py`` and ``scripts/run_sbc.py`` need the exact
same "theta in, wave tensor out" mapping -- SBC is only meaningful if its
simulator is bit-identical to the one that produced the training set. Keeping
the dispatch here rather than in either script guarantees that.
"""

from __future__ import annotations

import numpy as np

from hh_npe.data.waves import FEATURES_MVP, FEATURES_TWOASSET, aggregate_waves

AGE_START_SIM = 20
AGE_END_SIM = 90


def simulate_one_hark(
    theta: np.ndarray, sim_seed: int, start_age: int, n_waves: int,
    wave_years: int, grid: str = "coarse",
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 1-2 path: HARK ``MarkovConsumerType``, no credit cards, beta = 1."""
    from hh_npe.simulator.forward import simulate_households
    from hh_npe.simulator.lifecycle import build_lifecycle_agent, solve_lifecycle

    delta, crra = float(theta[0]), float(theta[1])
    agent = build_lifecycle_agent(
        delta=delta, crra=crra,
        age_start=AGE_START_SIM, age_end=AGE_END_SIM, n_agents=1,
    )
    solve_lifecycle(agent)
    panel = simulate_households(agent, n_households=1, seed=sim_seed)
    x, alive = aggregate_waves(
        panel, age_start_sim=AGE_START_SIM, start_age=start_age,
        n_waves=n_waves, wave_years=wave_years, features=FEATURES_MVP,
    )
    return x[0], alive[0]


def simulate_one_twoasset(
    theta: np.ndarray, sim_seed: int, start_age: int, n_waves: int,
    wave_years: int, grid: str = "mid",
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 3 path: credit cards, illiquid asset, naive quasi-hyperbolic beta."""
    from hh_npe.simulator.twoasset import GRIDS, simulate, solve

    beta, delta, crra = (float(v) for v in theta)
    if grid not in GRIDS:
        raise ValueError(f"unknown grid {grid!r}; expected one of {sorted(GRIDS)}")
    spec = GRIDS[grid]
    sol = solve(beta, delta, crra, spec)
    panel = simulate(sol, n_households=1, seed=sim_seed)
    x, alive = aggregate_waves(
        panel, age_start_sim=AGE_START_SIM, start_age=start_age,
        n_waves=n_waves, wave_years=wave_years, features=FEATURES_TWOASSET,
    )
    return x[0], alive[0]


SIMULATORS = {"hark": simulate_one_hark, "twoasset": simulate_one_twoasset}

#: Feature set each simulator emits, for embedder sizing and sanity checks.
FEATURES_FOR = {"hark": FEATURES_MVP, "twoasset": FEATURES_TWOASSET}
