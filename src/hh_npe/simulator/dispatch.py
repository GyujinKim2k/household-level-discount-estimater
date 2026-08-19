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


def simulate_batch_twoasset_gpu(
    thetas: np.ndarray, seed_base: int, start_age: int, n_waves: int,
    wave_years: int, grid: str = "full", theta_batch: int = 16, chunk: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 3 on the GPU: many draws per backward induction, one panel each.

    Lives here, beside the single-draw CPU paths, for the reason in the module
    docstring: SBC is only meaningful when its simulator is the one that made
    the training set. The GPU solve is reproducible at a fixed
    ``(device, theta_batch, chunk)`` but not across them -- ~99% of states hold
    two exactly-tied choices, and cuBLAS picks its summation order by problem
    size -- so those settings are arguments rather than defaults to be guessed,
    and callers should take them from the dataset's recorded configuration.

    The forward pass and wave aggregation stay on the CPU and are shared with
    :func:`simulate_one_twoasset` verbatim; only the solver differs.
    """
    from hh_npe.simulator.twoasset import GRIDS, simulate
    from hh_npe.simulator.twoasset_gpu import solve_batch

    if grid not in GRIDS:
        raise ValueError(f"unknown grid {grid!r}; expected one of {sorted(GRIDS)}")

    # Consume each sub-batch before solving the next: a Solution holds ~58 MB of
    # policy arrays, so accumulating a whole large block would exhaust host RAM.
    xs, alives = [], []
    for s0 in range(0, len(thetas), theta_batch):
        s1 = min(s0 + theta_batch, len(thetas))
        sols = solve_batch(thetas[s0:s1], GRIDS[grid],
                           theta_batch=theta_batch, chunk=chunk)
        for i, sol in enumerate(sols):
            panel = simulate(sol, n_households=1, seed=seed_base + s0 + i)
            x, alive = aggregate_waves(
                panel, age_start_sim=AGE_START_SIM, start_age=start_age,
                n_waves=n_waves, wave_years=wave_years, features=FEATURES_TWOASSET,
            )
            xs.append(x[0])
            alives.append(alive[0])
        del sols
    return np.stack(xs), np.stack(alives)


SIMULATORS = {"hark": simulate_one_hark, "twoasset": simulate_one_twoasset}

#: Feature set each simulator emits, for embedder sizing and sanity checks.
FEATURES_FOR = {"hark": FEATURES_MVP, "twoasset": FEATURES_TWOASSET}
