# Simulator Specification — Household-Level NPE Project

**Status:** DRAFT — not yet frozen.
**Phase:** Phase 3 (two-asset, credit cards, present bias). Port **validated**
against Laibson et al.'s published simulation (§7.1).
**Last updated:** 2026-08-13.

This document specifies the lifecycle consumption-saving simulator used to
generate training data for the household-level NPE. It must be **frozen
(timestamped via a tagged git commit)** before any PSID empirical work begins.

Two simulators exist in the tree. **Phase 3 is the `twoasset` simulator** and is
what this spec describes. The Phase 1–2 HARK MVP (`hark`) is retained only so
those results stay reproducible; §§1–5 below describe the Phase 3 model, and
the Phase 1–2 model is summarised in §9.

## 1. Model overview

Phase 3: **two-asset lifecycle consumption-saving with credit cards and naive
quasi-hyperbolic discounting**, ported from Laibson, Maxted, Repetto and
Tobacman's replication package (`replication-package-LLMRT/`). Annual
frequency, ages 20–90 (`T = 71`).

Per period the household holds liquid assets `X` and illiquid assets `Z`, and
chooses next-period `(X', Z')` jointly:

- `X` may go **negative** down to an age-varying credit limit, borrowed at
  `R_CC`; positive balances earn `R`. This is the credit-card margin.
- `Z ≥ 0` earns `R_gamma > R`, but withdrawals pay a proportional, age-varying
  liquidation penalty. Its return is received as a consumable flow
  `(R_gamma − 1)·Z`.
- Utility is CRRA over consumption scaled by effective household size.
- A bequest motive with weight `alpha` values the annuitized estate.

The illiquid asset is what makes present bias identifiable: a present-biased
household simultaneously borrows expensively on `X` and holds illiquid `Z`, a
combination an exponential household would not choose.

Only their **baseline** configuration is ported. Skipped robustness branches:
`sqrtscale`, `mortalityn`, `split_income_at_retirement`,
`income_var_multiplier`, `income_auto9`, `zilliq`, the `R_gamma_expectation`
perceived-return case, and the sophisticated (`betahat = beta`) agent.

## 2. Parameters

### 2.1 Structural parameters (estimated)
- **β** — present-bias parameter. **Unlocked in Phase 3** (locked at 1 for
  Phases 1–2).
- **δ** — long-run discount factor.
- **ρ** (`crra` in code) — coefficient of relative risk aversion.

Ordering is `(β, δ, ρ)`, matching their `prefs` vector. `PriorBox.names`
returns this order and it is load-bearing for the posterior's column meaning.

**Naive, not sophisticated.** `betahat = 1`: the current self acts on `β·δ` but
believes every future self will discount exponentially. Backward induction
therefore takes **two** argmaxes per state — one at `β·δ` giving the simulated
policy, one at `betahat·δ` giving the value propagated to the previous self.
This matches their benchmark (`naif = 1`, `betahat = 1`).

### 2.2 Calibrated parameters
All first-stage values are theirs, education group **`comphs`** (completed high
school — their benchmark, `EDFbatch_baseline.m:22`). They are **frozen as
literals** in `src/hh_npe/simulator/laibson_calibration.py` so the 653 MB
replication package is not a runtime dependency;
`python -m hh_npe.simulator.laibson_calibration` re-extracts all 126 values
from the package and asserts bit-exact equality.

| Group | Source | Values |
|---|---|---|
| Demographics (kids, dependent adults) | IPUMS-USA | 6 coefficients |
| Income profile (age cubic, spouse, kids, dependents) | PSID | 7 coefficients |
| Income AR(1) + transitory | PSID | `psi = 0.8400`, `vareps = 0.05708`, `varnu = 0.04509` |
| Credit limit (quadratic in age, as a multiple of mean income) | SCF | 3 coefficients |
| Initial wealth (median wealth-to-income) | SCF | total `1.4696`, liquid `0.05486` |
| Mortality | SSA TR2023 historical, M/F average over 2000–2004 | 71 annual death probabilities |

Returns (their `benchmark` case): `R = 1.0203`, `R_gamma = 1.0500`,
`R_CC = 1.1059`. Bequest weight `alpha = 0.5`. Persistent income states
`nS = 3`, Tauchen span `m = 1.5`.

**Deviation from Phases 1–2:** mortality is now their SSA TR2023 schedule, not
HARK's bundled SSA 2020 period life table. The two simulators are calibrated to
different sources by design — the Phase 3 model is a replication target.

### 2.3 Prior on structural parameters (Phase 3 NPE training)
- `beta`  ~ Uniform[**0.30, 1.00**]
- `delta` ~ Uniform[**0.85, 1.00**]
- `crra`  ~ Uniform[**0.5, 5.0**]

All three of their benchmark estimates (β=0.5305, δ=0.9891, ρ=1.9355) fall
inside this box, as do the exponential-restriction estimates (δ=0.9600,
ρ=1.4663). `delta < 1` is required — the bequest term divides by `1 − δ`.

**Sampling**: scrambled Sobol sequence (`scipy.stats.qmc.Sobol`); the companion
sbi `BoxUniform` provides the log-density used during NPE training. Prefer
power-of-2 sample sizes.

## 3. Income process

Log income has a deterministic age/demographic profile plus a persistent AR(1)
component and an iid transitory shock:

- **Profile** `ymean_(age)`: cubic in age plus spouse, kids and dependent-adult
  terms, with household composition itself an exponential-quadratic function of
  age (§2.2).
- **Persistent**: AR(1) with `psi = 0.8400`, discretized to `nS = 3` states by
  **Tauchen (1986)** on a grid spanning `±1.5` unconditional standard
  deviations. Ported from their `discretizeAR1`.
- **Transitory**: iid lognormal, discretized onto the *liquid-grid lattice in
  levels* so realized income always lands on a representable cash-on-hand
  shift, with middle-Riemann-sum width weighting. Ported from their
  `discretizeIID`.

**Deviation from Phases 1–2, deliberate:** Phases 1–2 used Rouwenhorst, which
§3 of the earlier spec preferred on the grounds that it dominates Tauchen at
high persistence. At `nS = 3` the discretization choice materially changes the
income process, and Phase 3's purpose is comparison against their published
estimates, so fidelity to their method wins. Their `psi = 0.84` is only
moderately persistent, so Rouwenhorst's advantage does not apply here anyway.

Naming note: Laibson uses ρ for CRRA risk aversion; AR(1) persistence is also
conventionally written ρ. In code we use `crra` for the structural parameter
and `psi`/`YWORK_AUTO` for income persistence.

## 4. Solution method

Brute-force discrete dynamic programming — no first-order conditions, no
policy interpolation, matching their approach. For every state `(X, Z, S)` at
every age, the joint argmax is taken over all `(X', Z')` grid pairs.

Grids (`src/hh_npe/simulator/grids.py`) reproduce theirs exactly at default
granularity: **nX = 190, nZ = 84**. The liquid grid is uniform at `xjump` below
zero and coarsens geometrically above it; the illiquid grid coarsens
geometrically from zero.

**One structural deviation, for vectorization.** They store a *separate* liquid
grid per age because the credit limit varies with age. We build a single grid
wide enough for the largest limit over the lifecycle and carry a per-age
feasibility mask. Since the negative segment is uniform at `xjump` and every
limit is rounded to that lattice, each age's grid is an **exact subset** of the
common grid — no interpolation, no approximation. This is asserted in
`tests/test_twoasset.py::test_credit_limit_is_on_the_grid`.

**Correctness gate.** Their dynamic budget identity
(`LifecycleSim_BackwardInduct.m:169`, `assert(abs(check) < 1e-5)`) is
re-derived independently from the stored policy indices and checked over the
**entire** grid, not a sample, in float64. In float32 the identity holds to
~1e-9 relative, which is ample for an argmax; both are tested.

**Unsolvable states.** Some states satisfy the credit limit but admit no
affordable action (maximum debt, no illiquid buffer, near the terminal age).
Their code lets `max` return an arbitrary index there. We expose a `solvable`
mask instead, and assert the forward pass never reaches such a state.

**Floating-point precision is a modeling requirement, not an implementation
detail.** The utility gap between adjacent liquid-grid points is
`(C/hhs)^(−ρ)·xjump ≈ 1.1e−5` at typical consumption, while the continuation
value accumulates over 71 periods to `|EV| ≈ 1e2`. float32's resolution at that
magnitude is `1.5e−5` — *larger than the gap the argmax must resolve*. Running
the DP in float32 therefore selects quantization noise, and because ties break
toward the lowest liquid index it does so with a systematic bias: it inflated
simulated credit-card borrowing by roughly 50% and drove the MSM objective from
70.6 to 352.2.

Two mitigations, both required:
1. `solve` runs in **float64 by default** (`ModelSpec.dtype`). Their MATLAB is
   double precision throughout, so this matches them.
2. `_age_step` **centres `EV`** before it enters the large tensor. `argmax` is
   invariant to adding a constant, so this is exact, and it recovers most of the
   lost headroom — centred float32 tracks float64 to within ~5% on the target
   moments, versus a ~50% error uncentred.

float32 remains selectable for quick iteration but must **never** be used for a
dataset or a validation run.

**Grid snapping ties round up.** Income realizations land on the `xjump`
lattice by construction (§3), but the liquid grid coarsens to 2000/4000/6000
above \$50k, so **26.9%** of cash-on-hand queries at the full grid fall exactly
midway between two grid points. `_nearest_index` resolves these to the *higher*
index, matching MATLAB's `griddedInterpolant(..., 'nearest')`. Rounding down
instead discards up to \$6,000 of liquid wealth per snap — only for wealthier
households, compounding with age, and in the expectation step as well as the
forward pass. It inflated simulated credit-card borrowing by ~10%.

**Grid coarsening.** NPE needs 10³–10⁵ solves, so dataset generation coarsens
the grids. Coarsening is an approximation and its cost is *measured*, not
assumed — `scripts/validate_twoasset.py`:

| Grid | solve | max \|diff\|/se vs full | median relative |
|---|---|---|---|
| coarse 81×46 | 72 s | 3.15 | 5.4% |
| **mid 107×56** | **234 s** | **0.94** | **2.2%** |
| full 190×84 (theirs) | 1678 s | — | — |

Phase 3 dataset generation uses the **mid** grid, which sits under one standard
error of discretization error. The full grid remains the reference for
validation runs.

## 5. Forward simulation

`twoasset.simulate` ports their `LifecycleSim_ForwardIter.m`. Households are
seeded at the SCF median liquid/illiquid wealth-to-income ratios at age 20 and
followed for the full lifecycle. Returned panel arrays, shape
`(n_households, T)`:

`income`, `consumption` (including the illiquid dividend), `liquid_assets`
(their `simL = simX − simY`, the position *before* income arrives, negative =
credit-card debt), `illiquid_assets`, `cash_on_hand`, `income_state`, `t_age`.

**No mortality in the forward pass.** This is their design, not an omission:
death enters only through the backward induction and through the survival
weights used when averaging moments. No household is ever replaced, so the
rebirth filter of §6.3 is a no-op for Phase 3 and retains 100% of draws. The
filter is kept because the Phase 1–2 HARK path does replace dead agents.

## 6. Annual → observation-wave aggregation rule

NPE inputs are fixed-length trajectories of `n_waves` observation waves, each
spanning `wave_years` annual simulator periods. The rule is itself a
pre-registered modeling choice and is fixed as follows. Implemented in
`src/hh_npe/data/waves.py`.

### 6.1 Window definition
With `age_start_sim = 20` (§2.2) and a first-wave age `start_age`, wave `w`
(0-indexed) spans annual indices `[t_start + wave_years·w, t_start +
wave_years·(w+1))` where `t_start = start_age - age_start_sim`. A window
extending past the simulator's final period raises `ValueError` rather than
truncating.

All phases pin `wave_years = 2`, `start_age = 30`, `n_waves = 5` → observation
window **ages 30–39**.

`wave_years = 1` (annual) is implemented and tested but is **not** the
pre-registered choice. It was considered for Phase 3 on the grounds that
Laibson et al.'s own moments are annual, and rejected: PSID has been biennial
since 1997, and Phase 4's empirical target is PSID. The simulator runs at
annual frequency internally either way (§2.2); only the observation window
differs, and it must match the data the posterior will eventually be applied
to.

### 6.2 Aggregation by variable type
- **Flows** (`income`, `consumption`) — **summed** over the annual periods in
  the window. With `wave_years = 1` this is the single year's value.
- **Stocks** (`liquid_assets`) — value at the **end of the final year** of the
  window.

Feature order is load-bearing — the embedder's input dimension is positional.
Two sets are defined in `src/hh_npe/data/waves.py`:

| Set | Features | Used by |
|---|---|---|
| `FEATURES_MVP` | income, consumption, liquid_assets | Phases 1–2 (`hark`) |
| `FEATURES_TWOASSET` | income, consumption, liquid_assets, **illiquid_assets** | Phase 3 (`twoasset`) |

Phase 3 observes the illiquid balance because **eight of Laibson et al.'s
sixteen target moments are wealth conditional on debt status** (§7). Dropping
`illiquid_assets` would discard exactly the signal present bias rides on, since
the identifying pattern is expensive borrowing held *simultaneously* with
illiquid wealth. Output is `(n_households, n_waves, len(features))`, `float32`.

`income` and `consumption` are flows (summed); every other feature is a stock
(read at the end of the window).

### 6.3 Rebirth (single-lineage) filter
HARK replaces dead agents with newborns (§5), so a raw window can splice two
different households together. `aggregate_waves` returns an `alive` mask that
is False for any wave in which `t_age` is not monotonically non-decreasing
across the window **or across the boundary from the immediately preceding
annual period**. Dataset generation keeps only households with `alive` True at
every wave; at the Phase 2 pilot this retained 8,092 of 8,192 draws (98.8%).

## 7. Validation targets

### 7.1 Port fidelity (Phase 3 gate — **passed**)

At their exact 190×84 grid and their published benchmark estimates
(β=0.5305, δ=0.9891, ρ=1.9355), our 16 simulated moments match their stored
`MSMout.optMoments` to **mean |log(ours/theirs)| = 0.0102**, every moment
within 4%. MSM objective q = 85.3 against their 77.2, the gap being Monte Carlo
noise at `pop = 10000` under independent RNG streams.

Comparison is against **their simulated moments**, not the data. Comparing to
the data conflates port error with their model's own misfit — their objective
at the optimum is 77.2, so a perfect port still sits far from the data. The
moment code is separately verified by applying it to their own stored panel,
which reproduces all 16 of their values exactly.

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

## 9. Phase 1–2 model (retained for reproducibility)

The `hark` simulator, superseded by §§1–5 but kept so the Phase 2 results
remain reproducible. Single-asset lifecycle consumption-saving with CRRA
utility, exponential discounting (β locked at 1), AR(1) labor income, no
illiquid assets, no credit cards. Annual, ages 20–90 (`T_cycle = 70`).

- Solver: HARK 0.17 `MarkovConsumerType` backward induction, with the default
  `IncShkDstn` and `MrkvArray` constructors overridden by closures reading our
  Rouwenhorst grid and transition matrix. Within Markov state `k` the
  transitory shock is degenerate at `exp(grid[k])` and the permanent shock at
  1; `PermGroFac = 1` throughout.
- Income: AR(1) in logs, `rho_y = 0.97`, `sigma_eps = 0.15`, `mu_y = 0.0`,
  Rouwenhorst with `n_income_states = 9`, initial state drawn from the chain's
  stationary distribution.
- Calibration: survival from HARK's bundled SSA 2020 period life table
  (`sex="average"`), `Rfree = 1.03`, initial liquid wealth 0,
  `BoroCnstArt = 0.0`.
- Prior: `delta ~ U[0.85, 1.00]`, `crra ~ U[0.5, 5.0]`, `beta = 1` locked.
- Features: `FEATURES_MVP` (3), biennial waves, ages 30–39.

HARK's forward pass replaces dead agents with newborns, which is why §6.3's
rebirth filter exists; at the Phase 2 pilot it retained 8,092 of 8,192 draws
(98.8%).
