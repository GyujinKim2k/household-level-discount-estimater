# Simulator Specification — Household-Level NPE Project

**Status:** DRAFT — not yet frozen.
**Phase:** Phase 1 (MVP).
**Last updated:** 2026-05-24.

This document specifies the lifecycle consumption-saving simulator used to
generate training data for the household-level NPE. It must be **frozen
(timestamped via a tagged git commit)** before any PSID empirical work begins.

## 1. Model overview

Phase 1 MVP: single-asset lifecycle consumption-saving with CRRA utility,
exponential discounting (β=1), AR(1) labor income, age-specific mortality, no
illiquid assets, no credit cards, no present-bias. Annual frequency. Ages 20–90.

Phases 2–4 are out of scope for this version of the spec — they will produce
their own tagged versions (`simulator-v2`, ...).

## 2. Parameters

### 2.1 Structural parameters (estimated)
- **δ** — long-run discount factor.
- **ρ** (`crra` in code) — coefficient of relative risk aversion.
- **β** — present-bias parameter. **Locked at β=1 for Phase 1–2.**

### 2.2 Calibrated parameters (Phase 1 pinned values)
- Survival probabilities `LivPrb_t` — **SSA Trustees Report 2020 period life table, year=2020, sex="average"** (50/50 male/female mean). Source: HARK's bundled `HARK.Calibration.life_tables.us_ssa`.
- AR(1) income persistence `rho_y` = **0.97** (typical PSID-literature value).
- AR(1) innovation std `sigma_eps` = **0.15**.
- AR(1) mean of log income at entry `mu_y` = **0.0** (income is measured relative to mean).
- Rouwenhorst Markov chain size `n_income_states` = **9**.
- Risk-free rate `Rfree` = **1.03** (3% real; HARK / Carroll buffer-stock baseline).
- Initial liquid wealth at age 20 = **0** (HARK default `kLogInitMean = -12`; SCF-based distribution deferred to Phase 3).
- Borrowing constraint `BoroCnstArt` = **0.0** (no borrowing; credit cards deferred to Phase 3).
- Age range: **20–90, annual** → `T_cycle = 70` periods.
- Initial Markov state distribution = **stationary distribution of the Rouwenhorst chain**.

### 2.3 Prior on structural parameters (Phase 2 NPE training)
- `delta` ~ Uniform[**0.85, 1.00**]
- `crra` ~ Uniform[**0.5, 5.0**]
- `beta` = 1.0 (locked)
- **Sampling**: scrambled Sobol sequence (`scipy.stats.qmc.Sobol`) for low-discrepancy coverage; the companion sbi `BoxUniform` provides the log-density used during NPE training. Prefer power-of-2 sample sizes.

## 3. Income process

AR(1) in log income: `y_t = (1 - rho_y) * mu_y + rho_y * y_{t-1} + eps_t`, `eps_t ~ N(0, sigma_eps^2)`. Approximated as a discrete Markov chain via the **Rouwenhorst** method (preferred over Tauchen at high persistence; matches the unconditional mean and variance of the AR(1) exactly). Implemented in `src/hh_npe/simulator/income.py`.

Naming note: Laibson uses ρ for CRRA risk aversion; AR(1) persistence is also conventionally written ρ. In code we use `crra` for the structural parameter and `rho_y` for income persistence.

## 4. Solution method

HARK 0.17 `MarkovConsumerType` backward induction. We override HARK's default `IncShkDstn` and `MrkvArray` constructors (`MarkovConsumerType.default_['params']['constructors']`) with closures that read our Rouwenhorst grid and transition matrix from instance attributes. Within each Markov state `k`, the transitory shock distribution is degenerate at `exp(grid[k])` and the permanent shock distribution is degenerate at 1, so the only income variation comes from the Markov state transitions. `PermGroFac = 1` everywhere — permanent income stays at its initial value throughout life.

## 5. Forward simulation

`src/hh_npe/simulator/forward.py` calls HARK's built-in `.simulate()` and returns panel arrays of shape `(n_households, T_cycle)` for income, consumption, liquid assets, cash-on-hand, permanent income, Markov state, and `t_age`. HARK's default behavior replaces dead agents with newborns; we expose `t_age` so callers can detect rebirth events (`t_age` resets to 0). Filtering to single-lineage trajectories is performed at the NPE data-prep stage, not the simulator stage.

## 6. Annual → biennial aggregation rule

To be drafted. NPE inputs are **5-wave biennial trajectories** matching the
PSID observation schedule. The aggregation rule (which annual variable
collapses to each biennial wave: end-of-period stock? two-year average flow?
mid-period snapshot?) is itself a pre-registered modeling choice and will be
specified here before Phase 4.

## 7. Validation targets

- Reproduce HARK / Carroll buffer-stock standard lifecycle profiles
  (income / consumption / wealth means by age). **Phase 1 gate.**
- Reproduce Laibson et al. published moments at their reported θ̂ ≈
  (β=0.50, δ=0.99, ρ=1.3). **Phase 1 sanity check** per user decision
  2026-05-24.

## 8. Pre-registration commitments

- The simulator specified here is frozen at tag `simulator-v1` before any
  PSID data is loaded.
- Deviations from this spec require a new tagged version and a documented
  reason. The current document is replaced; the prior version remains in
  git history under its tag.

---

*Drafted: 2026-05-24. Author: Gyujin Kim.*
