# Phase 4 — PSID empirical results

**Status:** in progress. Estimates below are conditional on a simulator whose
training set has a known limitation (§7.1), so they are reported as the current
state, not as final.
**Last updated:** 2026-09-07.

Per-household posteriors over (β, δ, ρ) for 889 PSID households observed in
seven biennial waves, 2011–2023, against Laibson, Lee, Maxted, Repetto &
Tobacman's single population MSM estimate.

---

## 1. Headline

```
                          beta       delta        crra
median of means         0.8465      0.9898      4.5002
mean of means           0.8158      0.9802      4.2455
sd across households    0.1029      0.0231      0.8077
median posterior sd     0.1288      0.0130      0.1109

Laibson et al. MSM      0.5305      0.9891      1.9355
  their std error       0.1140      0.0051      0.4350

share of households whose 90% CI covers their estimate:
  beta 0.397    delta 0.735    crra 0.199
```

**δ replicates almost exactly** — 0.9898 against their 0.9891, with 73.5% of
households covering their point. **β and ρ are both substantially higher.**

### Is the heterogeneity real?

```
            between-hh sd   within-hh sd   ratio
beta               0.1029         0.1294    0.79
delta              0.0231         0.0198    1.17
crra               0.8077         0.1154    7.00
```

A ratio below 1 means households differ by *less* than any one of them is
uncertain — the apparent spread is estimation noise, and a single population
estimate would serve as well.

- **ρ heterogeneity is real** (7.0×).
- **δ is marginal** (1.2×).
- **β heterogeneity is not demonstrated** (0.79). This is a caution about the
  project's own premise and belongs in any writeup.

Within households the parameters are close to orthogonal: median
`corr(β,ρ) = +0.017`, `corr(β,δ) = −0.166`, `corr(δ,ρ) = −0.076`. There is no
strong β–ρ ridge, so the estimates are not trading off against each other.

Figures in `figures/`.

---

## 2. Sample

```
rows in the extract                 15044
interviewed in all 7 waves           8472
+ head/reference throughout          4470
+ aged 25-46 at 2011                 2149
+ age progresses 11-13 years         2119
+ non-missing on all features        2021
--- then Laibson et al.'s own filter ---
+ comphs education (12-15 yrs)       1117
+ not self-employed                  1012
+ no business or farm income          889
```

Headship, not attrition, is where the sample goes: requiring the *same person*
as head throughout halves it. Their education restriction removes another 47%.

**Matched to `3_scfanalysis_withCondMoments_bs.do:301–323`**, which is not
optional: their entire first stage — income profile, credit limits, initial
wealth — is estimated on `comphs` households only.

---

## 3. Feature construction

`(N, 7, 5)` = income, consumption, liquid, illiquid, age, in **2010 USD**
(their `cpibaseyear`). Flows deflated at the income reference year, stocks at
the interview year.

| feature | construction |
|---|---|
| income | after-tax **non-asset** income. `nasry = total family income − asset income`, then TAXSIM federal + state + employee FICA, **pro-rated by the non-asset share** exactly as `2_builddata.do:96` |
| consumption | `TOTAL EXPENDITURE − mortgage − property tax − home insurance − vehicle principal` **+ imputed rent for owners** (§5) |
| liquid | checking/saving + CD/bonds − credit-card debt |
| illiquid | stocks + IRA + vehicles + other + other real estate + farm/business + home equity, **net of all debts**, floored at 0 |
| age | head's age |

**Independent validation:** TAXSIM after-tax non-asset income for tax year 2010
has median 40,593 against a simulated 40,000 at age 30, and 51,000 vs 50,838 at
ages 35–44. Nothing was tuned to make that hold.

---

## 4. The consumption investigation

PSID consumption initially sat ~34% below simulated, with a gradient: c/y ran
1.68 in the bottom income decile to 0.40 in the top, against a flat ~0.99 in
simulation, **in every age band**. Five candidates were tested.

| candidate | verdict | slope gap closed |
|---|---|---|
| Differential measurement error | Engel: PSID's food elasticity is 0.575, inside the literature's 0.50–0.60, bounding differential under-reporting at 1.06–1.20× against the 2.2× observed | ~0 |
| Sample mismatch | applied their filter, 2119 → 889 | 3% |
| Household composition | typical-household adjustment (§6) | 2% |
| Calibration vintage | **refuted** — re-estimated their income process on 2011–2023 their way (`xtreg … , fe`): growth over ages 25–55 is 1.53× in 1982–91 and 2.03× now (steeper, not flatter), and `nhead` (0.319 → 0.356) and `kids` (0.013 → 0.021) replicate | ~0 |
| **Owner-occupied housing services** | **confirmed and fixed** (§5) | **42%** |

```
ages 35-44, all 7 waves    slope   med cons     c/y
matched, net               0.368      29660   0.684
+ rental value             0.554      35621   0.822
simulated                  0.807      51276   1.046
```

**A residual gap remains** (0.554 vs 0.807). It is no longer attributable to any
data-handling defect we have been able to find, and is most plausibly the
model's structural limits (§7.1).

---

## 5. Rental equivalence — the largest single correction

PSID reports owners' housing as mortgage + property tax + insurance. We strip
those as saving and transfers — correctly, and identically to PSID's own
`TOTAL CONSUMPTION WITH RENTAL VALUE`, which is
`TOTEXP − (mortgage + property tax + home insurance) + rental value`
(correlation 0.9936 with that formula, median absolute difference 0). That
leaves **owners with no housing consumption at all**, while renters keep their
rent. Owners are 42% of the sample in 2011 rising to 56% by 2023.

PSID began collecting `VALUE OF HOME IF RENTED` in **2019**. For 2011–2017 it is
imputed by **anchoring each household on its own later observed value** and
carrying it back in real terms at 2.49%/yr, self-calibrated from the 2019→2023
waves.

**A cross-sectional model `rent = f(income, wealth, size, age)` was deliberately
rejected.** It would make imputed consumption a function of income and illiquid
wealth — two other features — so the consumption channel would carry no
independent information and the posterior would read a mechanical identity as
behaviour. It would likely *improve* the apparent fit while invalidating it.

Validation and coverage:

```
out-of-sample (2023 -> 2019): median |error| 0.187, median error +0.000
imputed rent = 26.7% of owner consumption -> 5.0% consumption error
early owners with unusable anchor: 5.7% (fall back to the wave median,
  a constant, which adds noise but no spurious correlation)
error propagation, 20 draws: median consumption 35053 +- 56  (0.16%)
```

Effect on model fit:

```
in-box posterior mass: median 0.858 -> 0.986,  p10 0.523 -> 0.645
households the model cannot represent: 0 of 889
```

**This is the number that changed most.** The boundary pinning that made the
earlier estimates artifacts — 92% of δ intervals reaching 1.0 — largely
dissolved, because it was substantially caused by a missing consumption
component rather than by the model.

---

## 6. Matched methodology

Implemented from their code:

- **Typical-household adjustment** (`3_scfanalysis:52–56`). Each moment is
  regressed on `nhead, ndepad, und18, unemprate` plus cohort and age dummies,
  then re-centred on the **model's own** demographic profile
  (`b0·exp(b1·AGE − b2·AGE²)` from `laibson_calibration`). Necessary because the
  model has composition as a deterministic function of age and `grids.py:38`
  sets `spouse = 2.0` for every household — unmarried PSID households are
  otherwise not comparable to anything it generates. Worth 2%.
- **Pro-rated taxes** (`2_builddata.do:96`), removing 3.5% of federal liability
  attributable to asset income.
- **State unemployment**, with 2016+ imputed as the state's 2015 position scaled
  to the national level (the shipped BLS file stops in 2015; BLS blocks scripted
  download).

### Known deviations that cannot be closed

1. **`hasVisa` is not in PSID.** Their sample is conditional on *holding* a
   credit card; PSID only asks about card *debt*. Using debt as a possession
   proxy would select on the outcome that identifies β and bias it downward —
   toward their number, for the wrong reason. "Ever borrowed in 7 waves" gives
   66.6% against a true US holding rate of ~70% and is the best available bound
   (585 households), but it is a bound, not a fix.
2. **α = 2.02 is assumed, not estimated.** `generateAlphaBootstrap.m` is
   literally `2 + 0.5*randn(2000,1)`; the bootstrap only propagates uncertainty
   about the assumed value. Their parameters were fitted against doubled card
   debt, so not applying it is a units mismatch on the margin β rides on;
   applying it assumes PSID's under-reporting matches SCF's.
3. **Cross-section vs panel.** They match four age-band moments from repeated
   cross-sections; we condition on seven-wave trajectories. This is the
   contribution, not a defect, but "same information set" is never literally
   true.

---

## 7. Open issues

### 7.1 The simulator has almost no heterogeneity

The forward pass has **two** stochastic elements: a 3-state persistent income
Markov chain and a transitory income shock. Everything else is deterministic and
**identical across households**:

```python
xind = np.full(N, ix0, dtype=np.int64)   # every household: same initial liquid
zind = np.full(N, iz,  dtype=np.int64)   # every household: same initial illiquid
```

No unemployment spells, health or medical shocks, divorce or marriage, births
beyond the age profile, inheritances, or return heterogeneity. All
cross-sectional dispersion at any age comes from accumulated income shocks
starting from one common initial condition — the SCF median.

Measured income volatility is roughly right in the middle and thin in the tails:
the model implies var 0.2043 for the 2-year change in log income against an
actual 0.2632 (1.29×), but the robust IQR-based sd is 0.275 against the model's
0.452, so the typical household moves *less* than assumed and the excess is
tail events the model has no mechanism for.

**This is the leading explanation for ρ ≈ 4.5.** High ρ is the only channel this
model has for generating precautionary saving, so missing precautionary motives
are routed through it. The ρ between/within ratio of 7.0 says households
genuinely differ — but possibly in exposure to shocks the model cannot see
rather than in risk aversion.

### 7.2 Other

- Consumption slope still 0.554 against 0.807 simulated.
- 9.3% of households have posterior-mean ρ within 5% of the 5.0 ceiling. Down
  sharply from before the rental fix; widening the prior is **no longer the
  obvious next step**, and would move along a ridge the data do not resolve.
- β heterogeneity remains within estimation noise (§1).
- `SIMULATOR_SPEC` §7's Phase 1 gate — reproducing HARK/Carroll buffer-stock
  profiles — still has no test.

---

## 8. Reproducing

```bash
uv run python scripts/taxsim_run.py                       # after-tax income
uv run python scripts/build_psid_tensor.py --match_laibson \
    --out data/processed/psid_x_matched_net.pt            # x tensor, net wealth
uv run python scripts/impute_rental_value.py --n_draws 20 # + imputed rent
uv run python scripts/psid_posterior.py \
    --x data/processed/psid_x_rental.pt --out outputs/psid_rental
uv run python scripts/plot_psid_population.py \
    --posterior_npz outputs/psid_rental/posterior_uncorrected.npz \
    --x data/processed/psid_x_rental.pt --out outputs/psid_rental
```

PSID microdata is not in the repository — see `PSID_DATA.md` for the variable
codes needed to re-pull it. Six Data Center extracts are used: `J364786` (IND
linkage), `J364817` (FAM superset), `J364913` (education), `J364914` (rental
value). `J364812`, `J364814` and `J364816` are strict subsets of `J364817`.
