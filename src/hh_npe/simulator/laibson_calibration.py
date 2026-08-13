"""Frozen first-stage calibration from Laibson et al.'s replication package.

Education group ``comphs`` (their benchmark, ``EDFbatch_baseline.m:22``).
Values are inlined so the 653 MB replication package is **not** a runtime
dependency; ``python -m hh_npe.simulator.laibson_calibration`` re-extracts them
from the package and asserts they still match.

Sources (their ``FirstStageParams.m``):
- demographics — IPUMS-USA
- income process — PSID
- credit limit — SCF
- mortality — SSA TR2023 historical death probabilities, male/female average
  over calendar years 2000-2004, ages 20-90
- second-stage target moments — their ``est_secondstage.mat``
"""

from __future__ import annotations

import numpy as np

EDUC = "comphs"
AGE_START = 20
AGE_END = 90
AGE_RETIRE = 64  # their ``retireage``; only used by the income-split robustness

# --- demographics: effective household size ------------------------------
# kids_(age)    = a0kids * exp(a1kids*age - a2kids*age^2)
# depadul_(age) = a0depadul * exp(a1depadul*age - a2depadul*age^2)
A0_KIDS = np.float64(0.003410572104586154)
A1_KIDS = np.float64(0.35821261723801895)
A2_KIDS = np.float64(0.005081298245188375)
A0_DEPADUL = np.float64(4.585428728590593e-06)
A1_DEPADUL = np.float64(0.45178921907870556)
A2_DEPADUL = np.float64(0.004382787905395041)

# Household-member weights [spouse, dependent adult, kid] (baseline, not sqrt scale).
HH_WEIGHT = (1.0, 1.0, 0.4)

# --- income: deterministic log profile ------------------------------------
# ymean_(age) = cons + agecoeff*age + age2coeff*age^2/100 + age3coeff*age^3/10000
#               + spousecoeff*spouse + kidscoeff*kids + depadulcoeff*depadul
YWORK_KIDSCOEFF = np.float64(0.013492077455287167)
YWORK_SPOUSECOEFF = np.float64(0.31911520596187626)
YWORK_DEPADULCOEFF = np.float64(0.23651044041995328)
YWORK_AGECOEFF = np.float64(0.13502511541998957)
YWORK_AGE2COEFF = np.float64(-0.2221586353815037)
YWORK_AGE3COEFF = np.float64(0.10646200599193933)
YWORK_CONS = np.float64(7.563421249389648)

# --- income: stochastic component ----------------------------------------
# AR(1) persistent component (variance ``vareps``) + iid transitory (``varnu``).
YWORK_AUTO = np.float64(0.8400135500431678)
YWORK_VAREPS = np.float64(0.05707941151455871)
YWORK_VARNU = np.float64(0.04508554448217923)
YWORK_SIGMAEPS = float(np.sqrt(YWORK_VAREPS))
YWORK_SIGMANU = float(np.sqrt(YWORK_VARNU))

# --- credit limit ---------------------------------------------------------
# creditline_(age) = c0 + c1*age + c2*age^2   (as a multiple of mean income)
C0_CREDIT = np.float64(0.16721227922042575)
C1_CREDIT = np.float64(-0.001869365470594639)
C2_CREDIT = np.float64(0.00013566344085291099)

# --- initial wealth (median wealth-to-average-income ratios) --------------
MED_TOTAL_WEALTH = np.float64(1.4695795059204104)
MED_LIQ_WEALTH = np.float64(0.054860156774520885)

# --- rates of return (their 'benchmark' case) -----------------------------
R_FREE = 1.0203      # liquid saving
R_GAMMA = 1.0500     # illiquid asset
R_CC = 1.1059        # credit-card borrowing

# --- other structural constants (their baseline flags) --------------------
ALPHA_BEQUEST = 0.5  # weight on the bequest motive
N_INCOME_STATES = 3  # their ``nS``
AR1_GRID_SPAN = 1.5  # their ``m``: grid half-width in multiples of the sd

# --- mortality: P(die between age i and i+1), ages 20..90 ----------------
DEATH_PROB = np.array([
    np.float64(0.0009253000000000001), np.float64(0.0009497), np.float64(0.0009505), np.float64(0.0009434),
    np.float64(0.0009381000000000001), np.float64(0.0009358), np.float64(0.0009422999999999999), np.float64(0.0009505),
    np.float64(0.0009629), np.float64(0.0009841), np.float64(0.0010098), np.float64(0.001051),
    np.float64(0.0011065000000000003), np.float64(0.0011768), np.float64(0.0012594), np.float64(0.0013555),
    np.float64(0.0014624), np.float64(0.0015855), np.float64(0.0017194), np.float64(0.0018652000000000002),
    np.float64(0.0020253), np.float64(0.0021964000000000003), np.float64(0.002382), np.float64(0.0025872),
    np.float64(0.0028059), np.float64(0.0030424), np.float64(0.0032958999999999996), np.float64(0.0035637000000000004),
    np.float64(0.003844), np.float64(0.0041329999999999995), np.float64(0.0044346), np.float64(0.004750599999999999),
    np.float64(0.0050941), np.float64(0.0054748), np.float64(0.0059254), np.float64(0.0064429),
    np.float64(0.0070162), np.float64(0.007663499999999999), np.float64(0.0083956), np.float64(0.009182599999999999),
    np.float64(0.010024700000000001), np.float64(0.010931199999999999), np.float64(0.011941499999999999), np.float64(0.0130477),
    np.float64(0.014299800000000001), np.float64(0.0157005), np.float64(0.0171709), np.float64(0.0187691),
    np.float64(0.0204987), np.float64(0.0223924), np.float64(0.0244593), np.float64(0.0267129),
    np.float64(0.029215599999999998), np.float64(0.0319821), np.float64(0.035022300000000006), np.float64(0.038369),
    np.float64(0.0420458), np.float64(0.04609339999999999), np.float64(0.050580099999999996), np.float64(0.055610799999999995),
    np.float64(0.06129949999999999), np.float64(0.0677416), np.float64(0.0749908), np.float64(0.0830679),
    np.float64(0.0920392), np.float64(0.10198700000000001), np.float64(0.11297109999999999), np.float64(0.1250501),
    np.float64(0.1382246), np.float64(0.1523771), np.float64(0.1675718),
])

# --- second-stage target moments -----------------------------------------
# 16 rows = 4 moment types x 4 age bands (21-30, 31-40, 41-50, 51-60), in the
# order [%Visa, meanVisa, wealth|debt, wealth|no debt]. Columns: value, se, N.
MOMENT_NAMES = ("pct_visa", "mean_visa", "wealth_debt", "wealth_nodebt")
AGE_BANDS = ((21, 30), (31, 40), (41, 50), (51, 60))
TARGET_MOMENTS = np.array([
    np.float64(0.6395306417155814), np.float64(0.6292221575171006), np.float64(0.5883615818911393), np.float64(0.5026888776318379),
    np.float64(0.11140327191032179), np.float64(0.0963981110101683), np.float64(0.1096009531059808), np.float64(0.10443145018365),
    np.float64(1.221574513866596), np.float64(1.8679500796161603), np.float64(3.377182686700301), np.float64(4.64977713548991),
    np.float64(1.6585798850310285), np.float64(2.8002288033170393), np.float64(4.612995320394674), np.float64(8.070565796020265),
])
TARGET_MOMENT_SE = np.array([
    np.float64(0.02148365664375412), np.float64(0.02589845728101662), np.float64(0.029116165470790097), np.float64(0.03689864131366247),
    np.float64(0.012358889193796187), np.float64(0.013861757011603465), np.float64(0.017146113661819503), np.float64(0.020241294491796045),
    np.float64(0.11657422380175385), np.float64(0.1667293412294971), np.float64(0.2268413032747909), np.float64(0.34046862967675323),
    np.float64(0.12918191063990117), np.float64(0.1542058567916562), np.float64(0.25180196072952105), np.float64(0.3926143586496589),
])
TARGET_MOMENT_N = np.array([
    np.float64(1097.0), np.float64(1504.0), np.float64(1698.0), np.float64(1411.0),
    np.float64(1097.0), np.float64(1504.0), np.float64(1698.0), np.float64(1411.0),
    np.float64(1097.0), np.float64(1504.0), np.float64(1698.0), np.float64(1411.0),
    np.float64(1097.0), np.float64(1504.0), np.float64(1698.0), np.float64(1411.0),
])

# --- Laibson et al. MSM estimates, table 3 (order: beta, delta, rho) ------
BENCHMARK_PREFS = (0.5305, 0.9891, 1.9355)     # naive quasi-hyperbolic
EXPONENTIAL_PREFS = (1.0, 0.9600, 1.4663)      # beta identically 1


def survival_share() -> np.ndarray:
    """``alive_``: fraction of the age-20 cohort still alive at each age."""
    return np.concatenate([[1.0], np.cumprod(1.0 - DEATH_PROB[:-1])])


if __name__ == "__main__":  # re-extract from the replication package and verify
    from pathlib import Path

    from scipy.io import loadmat

    root = Path(__file__).resolve().parents[3] / "replication-package-LLMRT"
    b = root / "LifecycleSimulation" / "input" / EDUC
    d = root / "ParameterAndMoments" / "DeathProbs"

    dem = loadmat(b / "est_firststage_demographics.mat")["est_demographics"].squeeze()
    inc = loadmat(b / "est_firststage_income.mat")["est_income"].squeeze()
    ar1 = loadmat(b / "est_firststage_income.mat")["est_ar1"].squeeze()
    cred = loadmat(b / "est_firststage_creditlim.mat")["est_creditlim"].squeeze()
    iw = loadmat(b / "est_firststage_initwealth.mat")["est_initwealth"].squeeze()
    ss = loadmat(b / "est_secondstage.mat")["est_secondstage"]

    m = np.genfromtxt(d / "DeathProbsE_M_Hist_TR2023.csv", delimiter=",", skip_header=2)[:, 1:]
    f = np.genfromtxt(d / "DeathProbsE_F_Hist_TR2023.csv", delimiter=",", skip_header=2)[:, 1:]
    death = ((m + f) / 2)[100:105, 20:91].mean(axis=0)

    checks = [
        ("demographics", dem, [A0_KIDS, A1_KIDS, A2_KIDS, A0_DEPADUL, A1_DEPADUL, A2_DEPADUL]),
        ("income", inc, [YWORK_KIDSCOEFF, YWORK_SPOUSECOEFF, YWORK_DEPADULCOEFF,
                         YWORK_AGECOEFF, YWORK_AGE2COEFF, YWORK_AGE3COEFF, YWORK_CONS]),
        ("ar1", ar1, [YWORK_AUTO, YWORK_VAREPS, YWORK_VARNU]),
        ("creditlim", cred, [C0_CREDIT, C1_CREDIT, C2_CREDIT]),
        ("initwealth", iw, [MED_TOTAL_WEALTH, MED_LIQ_WEALTH]),
        ("death", death, DEATH_PROB),
        ("moments", ss[:, 0], TARGET_MOMENTS),
        ("moment_se", ss[:, 1], TARGET_MOMENT_SE),
        ("moment_N", ss[:, 2], TARGET_MOMENT_N),
    ]
    for name, fresh, frozen in checks:
        np.testing.assert_allclose(np.ravel(fresh), np.ravel(frozen), rtol=0, atol=0)
        print(f"{name:14s} ok ({np.size(fresh)} values)")
    print("all frozen calibration values match the replication package")
