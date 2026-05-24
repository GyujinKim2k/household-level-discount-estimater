# Household-Level NPE for Lifecycle Preference Parameters

Per-household Neural Posterior Estimation (NPE) over (β, δ, ρ) for the
Laibson, Lee, Maxted, Repetto, Tobacman (NBER WP 13314 / RFS 2026) lifecycle
consumption-saving model. The **headline contribution is a posterior
distribution per household**, not a population point estimate.

## Status

**Phase 1 (current): MVP simulator.** Single liquid asset, CRRA, exponential
discounting (β=1), AR(1) income (Rouwenhorst-discretized, via HARK's
`MarkovConsumerType`), mortality, age 20–90, annual frequency.

Subsequent phases:
- **Phase 2** — NPE proof of concept on the MVP simulator (β=1, learn δ, ρ);
  SBC must pass before moving on.
- **Phase 3** — Add illiquid asset with age-declining adjustment cost,
  credit cards with income-based credit limit, naive β-δ quasi-hyperbolic
  discounting, medical / employment / family shocks.
- **Phase 4** — PSID 2005–2013 empirics, RNPE for misspecification, MMD
  checks, out-of-sample horse race, external validation against unused PSID
  variables.

## Setup

Requires Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync --extra dev
uv run pytest tests/test_env.py
```

The env test imports `torch`, `sbi`, `HARK`, `hydra`, `wandb`, `polars`, and
`numba`, solves a trivial HARK model, and reports CUDA availability. **No GPU
required** for this test to pass; the project is designed to run on CPU and
transparently use GPU when one is attached.

## Layout

```
configs/        Hydra configs for simulator / NPE / evaluation
src/hh_npe/     Package
  simulator/    HARK-based lifecycle model + forward simulation
  data/         Annual → biennial aggregation; PSID loaders (Phase 4)
  npe/          Embedder + SNPE-C training
  evaluation/   SBC, coverage tests
  utils/        Logging, seeding
scripts/        CLI entry points (populated in Phase 1+)
tests/          pytest
SIMULATOR_SPEC.md   Pre-registration spec; frozen before Phase 4
```

## Reproducibility

- Seeds set centrally (`src/hh_npe/utils/seeding.py`, populated in Phase 1).
- Hydra dumps resolved config to `outputs/<run>/.hydra/config.yaml`.
- W&B runs default to `mode="offline"`; run `wandb login` to enable cloud sync.

## References

- Laibson, Lee, Maxted, Repetto, Tobacman — NBER WP 13314, RFS 2026.
- Cranmer, Brehmer, Louppe — PNAS 2020 (SBI review).
- Greenberg, Nonnenmacher, Macke — 2019 (SNPE-C / APT).
- Talts et al. — 2018 (simulation-based calibration).
- Ward et al. — NeurIPS 2022 (Robust NPE under misspecification).
- Wei & Jiang — Marketing Science 2025 (population-level NNE; the contrast).

## Pre-registration

See [SIMULATOR_SPEC.md](SIMULATOR_SPEC.md) for the simulator specification.
This document must be **frozen via a tagged git commit before any PSID work
begins**.
