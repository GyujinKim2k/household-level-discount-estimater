"""Forward simulation from a solved lifecycle policy.

Returns raw panel arrays of shape ``(n_households, T_cycle)``. HARK's default
behavior is to replace dead agents with newborns (so panel size stays
constant). We expose ``t_age`` in the output so downstream code can detect
rebirth events (``t_age`` resets to 0); filtering to single-lineage
trajectories is the caller's responsibility (typically done at the
data-preparation stage for NPE training).
"""

from __future__ import annotations

import numpy as np
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType


def simulate_households(
    agent: MarkovConsumerType,
    n_households: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Simulate ``n_households`` for the full ``agent.T_cycle`` lifecycle.

    Returns a dict of arrays with shape ``(n_households, T_cycle)``:

    - ``income``         — actual income ``p_t * exp(grid[Mrkv_t])``
    - ``consumption``    — level consumption ``cNrm * pLvl``
    - ``liquid_assets``  — end-of-period assets ``aNrm * pLvl``
    - ``cash_on_hand``   — beginning-of-period cash ``mNrm * pLvl``
    - ``pLvl``           — permanent income (constant per agent in MVP setup)
    - ``mrkv_state``     — integer index into ``agent.grid_vals``
    - ``t_age``          — integer age within the lifecycle; resets to 0 on rebirth
    """
    if not getattr(agent, "solution", None):
        raise ValueError("Agent must be solved before simulating")

    T = agent.T_cycle
    agent.AgentCount = n_households
    agent.T_sim = T
    agent.seed = seed

    # Ensure t_age is tracked so callers can detect HARK's rebirth replacements.
    if "t_age" not in agent.track_vars:
        agent.track_vars = list(agent.track_vars) + ["t_age"]

    agent.initialize_sim()
    agent.simulate()

    h = agent.history
    pLvl = h["pLvl"]  # HARK convention: (T, N)
    cNrm = h["cNrm"]
    aNrm = h["aNrm"]
    mNrm = h["mNrm"]
    mrkv = h["Mrkv"].astype(int)
    t_age = h["t_age"].astype(int)

    grid = agent.grid_vals
    income = pLvl * np.exp(grid[mrkv])

    # Transpose to (n_households, T_cycle) — household-as-row is the natural
    # orientation for downstream NPE feature engineering.
    return {
        "income": income.T,
        "consumption": (cNrm * pLvl).T,
        "liquid_assets": (aNrm * pLvl).T,
        "cash_on_hand": (mNrm * pLvl).T,
        "pLvl": pLvl.T,
        "mrkv_state": mrkv.T,
        "t_age": t_age.T,
    }
