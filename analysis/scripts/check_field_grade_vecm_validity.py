"""
Check if Field_Grade_Officers_Z (borderline I(0)/I(1)) is causing problems in VECM
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("DIAGNOSTIC: Is Field_Grade_Officers_Z Causing VECM Problems?")
print("=" * 100)

# Load data
df = pd.read_excel('analysis/complete_normalized_dataset_v12.3.xlsx')

SELECTED_VARS = [
    'Junior_Enlisted_Z',
    'Company_Grade_Officers_Z',
    'Field_Grade_Officers_Z',  # <-- BORDERLINE VARIABLE
    'GOFOs_Z',
    'Warrant_Officers_Z',
    'Policy_Count_Log',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z'
]

data = df[SELECTED_VARS].dropna()

# ============================================================================
# TEST 1: Estimate AR(1) coefficient for Field_Grade
# ============================================================================
print("\n" + "=" * 100)
print("TEST 1: Is Field_Grade a Near-Unit Root Process?")
print("=" * 100)

from statsmodels.tsa.ar_model import AutoReg

fg = data['Field_Grade_Officers_Z']

# Estimate AR(1): y_t = c + rho * y_{t-1} + e_t
ar_model = AutoReg(fg, lags=1, trend='c')
ar_result = ar_model.fit()

rho = ar_result.params[1]  # AR(1) coefficient
print(f"\nAR(1) coefficient (rho): {rho:.6f}")
print(f"\nInterpretation:")
if rho >= 0.99:
    print(f"  rho >= 0.99: Very close to unit root (essentially I(1))")
    print(f"  -> Safe to treat as I(1) in VECM")
elif rho >= 0.95:
    print(f"  0.95 <= rho < 0.99: Near-unit root")
    print(f"  -> Borderline I(0)/I(1), approximately valid in VECM")
elif rho >= 0.90:
    print(f"  0.90 <= rho < 0.95: Highly persistent I(0)")
    print(f"  -> Questionable to treat as I(1)")
else:
    print(f"  rho < 0.90: Clearly stationary I(0)")
    print(f"  -> Should NOT be in VECM with I(1) variables")

# ============================================================================
# TEST 2: Compare VECM with vs without Field_Grade
# ============================================================================
print("\n" + "=" * 100)
print("TEST 2: Does Removing Field_Grade Change Cointegration Rank?")
print("=" * 100)

# With Field_Grade (8 variables)
print("\nWith Field_Grade_Officers_Z (8 variables):")
joh_with = coint_johansen(data, det_order=0, k_ar_diff=1)
trace_with = joh_with.trace_stat
crit_with = joh_with.trace_stat_crit_vals[:, 1]
rank_with = sum(trace_with > crit_with)
print(f"  Cointegration rank: {rank_with}")

# Without Field_Grade (7 variables)
data_without = data.drop(columns=['Field_Grade_Officers_Z'])
print("\nWithout Field_Grade_Officers_Z (7 variables):")
joh_without = coint_johansen(data_without, det_order=0, k_ar_diff=1)
trace_without = joh_without.trace_stat
crit_without = joh_without.trace_stat_crit_vals[:, 1]
rank_without = sum(trace_without > crit_without)
print(f"  Cointegration rank: {rank_without}")

print(f"\nCOMPARISON:")
print(f"  With Field_Grade: rank = {rank_with} (out of {len(SELECTED_VARS)} variables)")
print(f"  Without Field_Grade: rank = {rank_without} (out of {len(SELECTED_VARS)-1} variables)")

if rank_with == len(SELECTED_VARS):
    print(f"\n  WARNING: Full rank detected WITH Field_Grade")
    print(f"  This suggests Field_Grade may be I(0), causing spurious cointegration")
elif rank_without == len(SELECTED_VARS) - 1:
    print(f"\n  WARNING: Full rank detected WITHOUT Field_Grade too")
    print(f"  The full rank issue is not caused by Field_Grade")
else:
    print(f"\n  Both tests show less than full rank -> OK")

# ============================================================================
# TEST 3: Check if Field_Grade Loads Significantly on Error Correction
# ============================================================================
print("\n" + "=" * 100)
print("TEST 3: Does Field_Grade Participate Meaningfully in Cointegration?")
print("=" * 100)

# Fit VECM with rank=6
vecm = VECM(data, k_ar_diff=1, coint_rank=6, deterministic='nc')
vecm_result = vecm.fit()

# Extract beta (cointegration vectors) and alpha (error correction)
beta = vecm_result.beta
alpha = vecm_result.alpha

# Field_Grade is variable index 2
fg_index = 2

print(f"\nField_Grade_Officers_Z in Cointegration Vectors (beta):")
print("-" * 100)
for r in range(6):
    beta_coef = beta[fg_index, r]
    print(f"  Coint Vector {r+1}: beta = {beta_coef:8.4f}")

avg_beta = np.mean(np.abs(beta[fg_index, :]))
print(f"\nAverage |beta| for Field_Grade: {avg_beta:.4f}")

print(f"\nField_Grade_Officers_Z Error Correction Speeds (alpha):")
print("-" * 100)
for r in range(6):
    alpha_coef = alpha[fg_index, r]
    # Get p-value from model summary
    print(f"  EC {r+1}: alpha = {alpha_coef:8.4f}")

avg_alpha = np.mean(np.abs(alpha[fg_index, :]))
print(f"\nAverage |alpha| for Field_Grade: {avg_alpha:.4f}")

print(f"\nINTERPRETATION:")
if avg_beta < 0.1 and avg_alpha < 0.05:
    print(f"  Field_Grade has very weak participation in cointegration")
    print(f"  -> May not truly be I(1), could be causing issues")
elif avg_beta < 0.5 and avg_alpha < 0.1:
    print(f"  Field_Grade has moderate participation")
    print(f"  -> Acceptable but not strong")
else:
    print(f"  Field_Grade participates meaningfully in cointegration")
    print(f"  -> Valid to include in VECM")

# ============================================================================
# TEST 4: Residual Analysis for Field_Grade Equation
# ============================================================================
print("\n" + "=" * 100)
print("TEST 4: Are Field_Grade Residuals Well-Behaved?")
print("=" * 100)

# Get residuals for Field_Grade equation
fg_resid = vecm_result.resid[:, fg_index]

# ADF test on residuals (should be stationary)
adf_resid = adfuller(fg_resid, maxlag=5)

print(f"\nADF test on Field_Grade residuals:")
print(f"  ADF statistic: {adf_resid[0]:.4f}")
print(f"  p-value: {adf_resid[1]:.4f}")

if adf_resid[1] < 0.01:
    print(f"\n  Residuals are stationary (p < 0.01) -> GOOD")
    print(f"  Model properly captured non-stationarity")
else:
    print(f"\n  WARNING: Residuals may not be stationary")
    print(f"  Model may not have properly captured dynamics")

# Check for autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(fg_resid, lags=[5], return_df=False)

print(f"\nLjung-Box test for autocorrelation (lag 5):")
print(f"  p-value: {lb[1][0]:.4f}")

if lb[1][0] > 0.05:
    print(f"  No autocorrelation detected (p > 0.05) -> GOOD")
else:
    print(f"  WARNING: Autocorrelation detected")
    print(f"  Model may be misspecified")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "=" * 100)
print("FINAL VERDICT: Is Field_Grade_Officers_Z Causing Problems?")
print("=" * 100)

print(f"""
DIAGNOSTICS SUMMARY:

1. AR(1) coefficient: rho = {rho:.4f}
   {'[GOOD]' if rho >= 0.95 else '[CONCERN]'} {'Near-unit root' if rho >= 0.95 else 'Clearly I(0)'}

2. Cointegration rank:
   With Field_Grade: {rank_with}, Without: {rank_without}
   {'[GOOD]' if rank_with < len(SELECTED_VARS) else '[WARNING]'} {'Not full rank' if rank_with < len(SELECTED_VARS) else 'Full rank detected'}

3. Participation in cointegration:
   Avg |beta| = {avg_beta:.4f}, Avg |alpha| = {avg_alpha:.4f}
   {'[GOOD]' if avg_beta > 0.3 or avg_alpha > 0.05 else '[CONCERN]'} {'Meaningful participation' if avg_beta > 0.3 or avg_alpha > 0.05 else 'Weak participation'}

4. Residual diagnostics:
   ADF p-value: {adf_resid[1]:.4f}, LB p-value: {lb[1][0]:.4f}
   {'[GOOD]' if adf_resid[1] < 0.05 and lb[1][0] > 0.05 else '[CONCERN]'} {'Well-behaved residuals' if adf_resid[1] < 0.05 and lb[1][0] > 0.05 else 'Residual issues'}

RECOMMENDATION:
""")

# Count how many tests passed
tests_passed = sum([
    rho >= 0.95,
    rank_with < len(SELECTED_VARS),
    avg_beta > 0.3 or avg_alpha > 0.05,
    adf_resid[1] < 0.05 and lb[1][0] > 0.05
])

if tests_passed >= 3:
    print("Field_Grade_Officers_Z appears valid for VECM (3+ tests passed)")
    print("The borderline stationarity is likely near-unit root, not true I(0)")
    print("-> PROCEED with current VECM")
elif tests_passed >= 2:
    print("Field_Grade_Officers_Z is borderline (2 tests passed)")
    print("Consider robustness check: re-run VECM without Field_Grade")
    print("-> CAUTIOUS PROCEED with sensitivity analysis")
else:
    print("WARNING: Field_Grade_Officers_Z may be causing issues (< 2 tests passed)")
    print("Consider excluding it or using deterministic trend in VECM")
    print("-> REVISE model specification")

print("\n" + "=" * 100)
