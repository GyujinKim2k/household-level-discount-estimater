# Simulator Specification — Household-Level NPE Project

**Status:** DRAFT — not yet frozen.
**Phase:** Phase 1 (MVP).
**Last updated:** 2026-08-09.

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

NPE inputs are **5-wave biennial trajectories** matching the PSID observation
schedule. The aggregation rule is itself a pre-registered modeling choice and
is fixed as follows. Implemented in `src/hh_npe/data/biennial.py`.

### 6.1 Window definition
With `age_start_sim = 20` (§2.2) and a first-wave age `start_age`, wave `w`
(0-indexed) spans annual indices `[t_start + 2w, t_start + 2w + 1]` where
`t_start = start_age - age_start_sim`. Phase 2 pins `start_age = 30` and
`n_waves = 5`, so the observation window is **ages 30–39**. A window extending
past the simulator's final period raises `ValueError` rather than truncating.

### 6.2 Aggregation by variable type
- **Flows** (`income`, `consumption`) — **summed** over the two annual periods
  in the window. Matches PSID's retrospective annual-flow questions, which
  cover the full period between waves rather than a snapshot.
- **Stocks** (`liquid_assets`) — value at the **end of the second year** of the
  window. Matches PSID's point-in-time wealth questions.

Feature order is fixed by `biennial.FEATURES = ("income", "consumption",
"liquid_assets")` and must not be reordered — the embedder's input dimension is
positional. Output is `(n_households, n_waves, 3)`, dtype `float32`.

### 6.3 Rebirth (single-lineage) filter
HARK replaces dead agents with newborns (§5), so a raw window can splice two
different households together. `aggregate_biennial` returns an `alive` mask
that is False for any wave in which `t_age` is not monotonically non-decreasing
across the window **or across the boundary from the immediately preceding
annual period**. Dataset generation keeps only households with `alive` True at
every wave; at the Phase 2 pilot this retained 8,092 of 8,192 draws (98.8%).

## 7. Validation targets

- Reproduce HARK / Carroll buffer-stock standard lifecycle profiles
  (income / consumption / wealth means by age). **Phase 1 gate.**
- Laibson et al. comparison. Their MSM estimates, read from the replication
  package (`LifecycleSimulation/output/simulations/table3/EDFbatch_table3*.mat`,
  `MSMout.optprefs` and `MSMout.optprefs_stderr.full`):

  | Specification | β | δ | ρ | q |
  |---|---|---|---|---|
  | Benchmark (naive quasi-hyperbolic) | 0.5305 (0.114) | 0.9891 (0.0051) | 1.9355 (0.435) | 77.2 |
  | Exponential restriction (β ≡ 1) | 1 (imposed) | 0.9600 (0.0053) | 1.4663 (0.226) | 759.6 |

  **The exponential row is the comparable one for Phases 1–2**, which lock β=1
  (§2.1) and expose no β parameter at all. Both δ=0.9600 and ρ=1.4663 fall
  inside the Phase 2 prior box (§2.3), so the pilot posterior can be checked
  against them directly.

  Note that this is a **parameter-level** comparison only. Their target moments
  (%Visa, meanVisa, wealth | debt, wealth | no debt) are credit-card moments,
  and the Phase 1 MVP has no credit cards (`BoroCnstArt = 0`, §2.2). A
  moment-level replication is therefore **deferred to Phase 3**, when credit
  cards and the illiquid asset enter.

  Supersedes the earlier target of θ̂ ≈ (β=0.50, δ=0.99, ρ=1.3) recorded
  2026-05-24; the ρ value there was wrong by ~1.5 SE and the β value is not
  representable in a Phase 1–2 model.

## 8. Pre-registration commitments

- The simulator specified here is frozen at tag `simulator-v1` before any
  PSID data is loaded.
- Deviations from this spec require a new tagged version and a documented
  reason. The current document is replaced; the prior version remains in
  git history under its tag.

---

*Drafted: 2026-05-24. Author: Gyujin Kim.*
