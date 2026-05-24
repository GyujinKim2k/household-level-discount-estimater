"""AR(1) income process discretized via Rouwenhorst.

Rouwenhorst is preferred over Tauchen for highly-persistent processes
(rho close to 1), which is the empirically relevant regime for annual
log earnings in PSID (rho_y typically 0.95-0.99). See Kopecky & Suen
(Review of Economic Dynamics 2010) for the comparison.
"""

from __future__ import annotations

import numpy as np


def rouwenhorst(
    n_states: int,
    rho: float,
    sigma: float,
    mu: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize the AR(1) ``y_t = (1 - rho) * mu + rho * y_{t-1} + eps_t``,
    ``eps_t ~ N(0, sigma**2)``, via the Rouwenhorst method.

    Parameters
    ----------
    n_states : int
        Number of grid points (>= 2).
    rho : float
        Persistence in (-1, 1).
    sigma : float
        Std dev of the innovation ``eps_t`` (> 0).
    mu : float
        Unconditional mean of ``y_t``. Default 0.

    Returns
    -------
    grid : ndarray of shape (n_states,)
        Evenly-spaced states, symmetric about ``mu``.
    P : ndarray of shape (n_states, n_states)
        Row-stochastic transition matrix; ``P[i, j] = Pr(y_{t+1} = grid[j] | y_t = grid[i])``.
    """
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states}")
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    # Unconditional std: sigma_y = sigma / sqrt(1 - rho^2)
    sigma_y = sigma / np.sqrt(1.0 - rho * rho)
    psi = sigma_y * np.sqrt(n_states - 1.0)
    grid = np.linspace(mu - psi, mu + psi, n_states)

    p = 0.5 * (1.0 + rho)
    q = p
    P = np.array([[p, 1.0 - p], [1.0 - q, q]])

    for n in range(3, n_states + 1):
        P_new = np.zeros((n, n))
        P_new[:-1, :-1] += p * P
        P_new[:-1, 1:] += (1.0 - p) * P
        P_new[1:, :-1] += (1.0 - q) * P
        P_new[1:, 1:] += q * P
        # Halve interior rows so all rows sum to 1.
        P_new[1:-1, :] *= 0.5
        P = P_new

    return grid, P


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic matrix ``P``.

    Solves ``pi @ P = pi`` subject to ``sum(pi) = 1`` via a constrained
    least-squares system. Assumes ``P`` is irreducible (true for Rouwenhorst).
    """
    n = P.shape[0]
    A = np.vstack([P.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi
