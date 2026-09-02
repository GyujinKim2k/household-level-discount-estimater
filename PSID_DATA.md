# PSID data constraints for Phase 4

What a PSID extract can and cannot supply for household-level NPE. Verified
against PSID's own family-file codebooks, not inferred from the literature —
the finding below overturns the wave count the simulator spec was frozen on,
so it is recorded with its evidence.

**Status:** verified 2026-08-31 (wave-count consequence updated 2026-09-02) against `FAM{2005,2007,2009,2011,2013,2015,2017,2019,2021,2023}ER_codebook.pdf`,
fetched from `psidtest.isr.umich.edu/documents/psid/codebook/`. No PSID
microdata exists in this tree; the replication package's `.dta` files are
10-row schema stubs and `1_mergepsid.do` downloads the real files via
`psid install`, which requires registration.

## 1. Laibson et al. do not use a PSID panel for the moments

From `replication-package-LLMRT/ParameterAndMoments/main.do`:

| source | years | role |
|---|---|---|
| IPUMS Census/ACS | 1980, 1990, 2000, 2001–2014 | household demographics |
| **PSID** | **1982–1991** | **income process only** (first stage) |
| **SCF** | **1995–2013** | **the 16 target moments** (second stage) |

The moments the MSM point estimate matches — credit card balances, credit
limits, wealth conditional on debt status — are built in
`SCF/code/2_buildmoments.do`. **SCF is a repeated cross-section with no panel
dimension**, so per-household posteriors are impossible on their data by
construction. This project's use of a PSID panel is therefore a departure from
the replication, not a reproduction of it, and any comparison against their
point estimate is across data sources. Say so in the writeup.

## 2. Wave schedule

PSID is annual 1968–1997 and **biennial from 1999**: 1999, 2001, … , 2023 =
13 waves. The 2023 family file was released May 2025.

Four runs of 10 consecutive biennial waves exist in calendar terms
(1999–2017, 2001–2019, 2003–2021, 2005–2023). Calendar availability is *not*
the binding constraint.

## 3. The binding constraint: credit-card debt is separable only from 2011

`FEATURES_TWOASSET` requires `liquid_assets` to go negative as credit-card
debt (`SIMULATOR_SPEC.md` §6). PSID reports that separately only from 2011.

**Before 2011** there is one lumped total. The 2009 question reads:

> W38. Aside from the debts that we have already talked about, like any
> mortgage on your main home or vehicle loans — do you (or anyone in your
> family living there) currently have any other debts such as **credit card
> charges, student loans, medical or legal bills, or loans from relatives**?
>
> W39. If you added up **all of these debts**, about how much would they
> amount to right now?

Credit cards cannot be recovered from that sum.

**From 2011** the module splits into `W38A/W39A` (credit/store card) plus
separate student-loan, medical, legal, relative-loan and residual items.

| wave | has credit/store card debt? | indicator | amount |
|---|---|---|---|
| 2005 | no — lumped | `ER26602` W38 WTR OTHER DEBTS | `ER26603` W39 VALUE ALL DEBTS |
| 2007 | no — lumped | `ER37620` | `ER37621` |
| 2009 | no — lumped | `ER43611` | `ER43612` |
| 2011 | **yes** | `ER48936` | `ER48937` |
| 2013 | **yes** | `ER54686` | `ER54687` |
| 2015 | **yes** | `ER61797` | `ER61798` |
| 2017 | **yes** | `ER67851` | `ER67852` |
| 2019 | **yes** | `ER73879` | `ER73880` |
| 2021 | **yes** | `ER80001` | `ER80002` |
| 2023 | **yes** | `ER83971` | `ER83972` |

Labels are `W38A WTR HAVE CREDIT/STORE CARD DEBT` and `W39A AMOUNT
CREDIT/STORE CARD DEBT` throughout. Note the measure includes **store cards**
alongside bank cards, which suits the model's revolving-debt margin.

Imputation and accuracy flags (`IMP VAL CREDIT CARD DEBT (W39A)`,
`ACC VAL …`) accompany the amount; decide explicitly whether to use reported
or imputed values, and record which.

### Consequence

```
2011, 2013, 2015, 2017, 2019, 2021, 2023  =  7 biennial waves
```

**Seven is the ceiling with a clean debt measure.** This is now the project
default (`SIMULATOR_SPEC.md` §6.1.2, `configs/npe/phase3.yaml`), and it is not
a compromise forced on a broken model — it is calibrated:

```
7 waves, k=8, 5-member ensemble, start age U{25..46}
  held-out log q   5.007
  coverage_90      beta 0.916   delta 0.902   crra 0.905   (nominal 0.900)
  KS               all three pass
```

Ten waves is genuinely sharper — 5.446 nats, a real 0.44 gap — so if a source
offering ten clean waves ever appears it is the better window. PSID is not that
source, and seven costs 0.44 nats while keeping calibration.

The alternative of stretching to **10 waves over 2005–2023** is rejected: the
first three waves' credit cards would be inseparable from medical, legal,
student and family debt, and those are exactly the waves carrying the
borrow-repay arc that identifies `beta`. Trading 0.44 nats for a mismeasured
headline parameter is the wrong direction.

> **Retracted.** An earlier version of this section called the seven-wave
> ceiling "a real obstacle", on the grounds that a 7-wave posterior failed SBC.
> That finding did not survive replication: it rested on one KS p-value from
> one training seed, and five seeds moved those p-values over two to three
> orders of magnitude. Single models at any wave count undercover (~0.87);
> ensembling fixes it at 7 waves as well as at 10. See `SIMULATOR_SPEC.md`
> §6.1.2–6.1.3. The obstacle was in the estimator, not in PSID.

## 4. Still unverified

- **Realised N.** Reinterview response is 96–98% per wave, so 7 waves (6
  transitions) retains roughly 0.97⁶ ≈ 0.83. PSID 2011 had ~8,900 families;
  heads aged 25–40 are perhaps a quarter. That suggests order 1,700–1,900
  households — arithmetic from published rates, **not a count**. Needs the
  extract.
- **Headship continuity.** A 7-wave window spans 12 years, over which families
  split, merge and change head. The model's household is one decision unit for
  life; PSID's is not. Define the linkage rule before counting.
- **Consumption.** PSID's expenditure module expanded in 1999 and again in
  2005. Check series consistency across 2011–2023 before use.
- **Illiquid assets.** The wealth module runs every wave since 1999, so this is
  expected to be fine, but the component mapping to the model's `Z` has not
  been fixed.

## Reproducing the check

```bash
curl -sL -A "Mozilla/5.0" -o fam2011.pdf \
  https://psidtest.isr.umich.edu/documents/psid/codebook/FAM2011ER_codebook.pdf
uv run --no-project --with pypdf python -c "
from pypdf import PdfReader; import re
r = PdfReader('fam2011.pdf')
pat = re.compile(r'^(ER\d+)\s+\"(W3[89][^\"]*)\"')
for p in r.pages:
    for line in (p.extract_text() or '').split('\n'):
        m = pat.match(line.strip())
        if m: print(m.group(1), m.group(2))
"
```

`psidonline.isr.umich.edu` returns 403 to scripted requests;
`psidtest.isr.umich.edu` serves the same codebooks and does not.
