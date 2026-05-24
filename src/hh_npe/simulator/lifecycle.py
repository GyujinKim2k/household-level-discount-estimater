"""HARK MarkovConsumerType wrapper for the Phase 1 MVP lifecycle model.

Encodes AR(1) log income as a Rouwenhorst Markov chain. Within each Markov
state ``k``, the transitory shock is degenerate at ``exp(grid[k])`` and the
permanent shock is degenerate at 1, so the only income variation comes from
the Markov state itself. Permanent income ``p_t`` stays at its initial value
throughout life (``PermGroFac=1``, ``psi=1``).

HARK 0.17 builds its income shock distributions via a constructor declared in
``MarkovConsumerType.default_['params']['constructors']``. We override the
``IncShkDstn`` and ``MrkvArray`` constructors with closures that read our
Rouwenhorst grid + transition matrix from attributes set during ``__init__``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.distributions import DiscreteDistributionLabeled

from hh_npe.simulator.income import rouwenhorst, stationary_distribution
from hh_npe.simulator.mortality import survival_schedule


def _make_inc_shk_dstn_constructor():
    def constructor(T_cycle, _grid_vals, **_):
        gv = np.asarray(_grid_vals)
        K = len(gv)
        out = []
        for _t in range(T_cycle):
            row = [
                DiscreteDistributionLabeled(
                    pmv=np.array([1.0]),
                    atoms=np.array([[1.0], [float(np.exp(gv[k]))]]),
                    var_names=["PermShk", "TranShk"],
                )
                for k in range(K)
            ]
            out.append(row)
        return out

    return constructor


def _make_markov_array_constructor():
    def constructor(T_cycle, _P_matrix, **_):
        return [np.asarray(_P_matrix)] * T_cycle

    return constructor


def build_lifecycle_agent(
    *,
    delta: float,
    crra: float,
    n_income_states: int = 9,
    rho_y: float = 0.97,
    sigma_eps: float = 0.15,
    mu_y: float = 0.0,
    Rfree: float = 1.03,
    age_start: int = 20,
    age_end: int = 90,
    mortality_year: int = 2020,
    mortality_sex: Literal["M", "F", "average"] = "average",
    BoroCnstArt: float = 0.0,
    n_agents: int = 1000,
) -> MarkovConsumerType:
    """Construct an *unsolved* lifecycle ``MarkovConsumerType``.

    Parameters
    ----------
    delta, crra
        Structural parameters (to be estimated by NPE in Phase 2).
    n_income_states, rho_y, sigma_eps, mu_y
        AR(1) income process configuration (Rouwenhorst).
    Rfree
        Gross risk-free rate (constant across age and Markov state).
    age_start, age_end
        Inclusive start, exclusive end. Default 20-90 → 70 periods.
    mortality_year, mortality_sex
        Inputs to :func:`hh_npe.simulator.mortality.survival_schedule`.
    BoroCnstArt
        Artificial borrowing constraint (default 0 = no borrowing).
    n_agents
        ``AgentCount`` for later simulation; does not affect the solve.
    """
    T_cycle = age_end - age_start
    grid_logy, Pmat = rouwenhorst(n_income_states, rho_y, sigma_eps, mu=mu_y)
    livprb = survival_schedule(
        age_start, age_end, year=mortality_year, sex=mortality_sex
    )
    K = n_income_states

    # Initial distribution over Markov states = stationary dist of the chain.
    init_mrkv = stationary_distribution(Pmat)

    defaults_ctors = MarkovConsumerType.default_["params"]["constructors"]
    ctors = dict(defaults_ctors)
    ctors["IncShkDstn"] = _make_inc_shk_dstn_constructor()
    ctors["MrkvArray"] = _make_markov_array_constructor()

    agent = MarkovConsumerType(
        cycles=1,
        T_cycle=T_cycle,
        CRRA=crra,
        DiscFac=delta,
        Rfree=[np.full(K, Rfree) for _ in range(T_cycle)],
        PermGroFac=[np.full(K, 1.0) for _ in range(T_cycle)],
        LivPrb=[np.full(K, p) for p in livprb],
        BoroCnstArt=BoroCnstArt,
        AgentCount=n_agents,
        MrkvPrbsInit=init_mrkv,
        constructors=ctors,
        _grid_vals=grid_logy,
        _P_matrix=Pmat,
    )
    # Stash useful arrays for the simulator and tests
    agent.grid_vals = grid_logy
    agent.P_matrix = Pmat
    agent.LivPrb_path = livprb
    return agent


def solve_lifecycle(agent: MarkovConsumerType) -> MarkovConsumerType:
    """Solve ``agent`` in place and return it."""
    agent.solve()
    return agent
