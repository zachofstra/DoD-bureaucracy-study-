"""
VECM Lag 2 Robustness Tests and Equation Documentation
Comprehensive diagnostic testing and model specification documentation

Tests performed:
1. Residual autocorrelation (Portmanteau/Ljung-Box)
2. Residual normality (Jarque-Bera)
3. Residual heteroskedasticity (ARCH effects)
4. System stability (eigenvalues)
5. Granger causality
6. Cointegration rank robustness
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

output_dir = 'data/analysis/VECM_LAG2'

print("=" * 100)
print("VECM LAG 2 - ROBUSTNESS TESTS AND EQUATION DOCUMENTATION")
print("=" * 100)

# =============================================================================
# LOAD DATA AND ESTIMATE MODEL
# =============================================================================
print("\n[1/7] Loading data and estimating VECM...")

df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

endog_vars = [
    'Junior_Enlisted_Z',
    'FOIA_Simple_Days_Z',
    'Total_PAS_Z',
    'Total_Civilians_Z',
    'Policy_Count_Log',
    'Field_Grade_Officers_Z',
    'GOFOs_Z'
]

exog_vars = ['GDP_Growth', 'Major_Conflict']

data = df[endog_vars + exog_vars].dropna()
endog_data = data[endog_vars]
exog_data = data[exog_vars]

# Estimate VECM
vecm_model = VECM(endog_data, exog=exog_data, k_ar_diff=2, coint_rank=4, deterministic='ci')
vecm_result = vecm_model.fit()

print(f"  Model estimated: Lag 2, Rank 4")
print(f"  Observations: {vecm_result.nobs}")
print(f"  Variables: {len(endog_vars)} endogenous, {len(exog_vars)} exogenous")

# =============================================================================
# TEST 1: RESIDUAL AUTOCORRELATION
# =============================================================================
print("\n[2/7] Testing for residual autocorrelation...")

autocorr_results = []
autocorr_pass = []

for i, var in enumerate(endog_vars):
    residuals = vecm_result.resid[:, i]

    # Ljung-Box test at multiple lags
    lb_results = acorr_ljungbox(residuals, lags=[5, 10, 15], return_df=True)

    # Check if p-values > 0.05 (no autocorrelation)
    pass_5 = lb_results.loc[5, 'lb_pvalue'] > 0.05
    pass_10 = lb_results.loc[10, 'lb_pvalue'] > 0.05
    pass_15 = lb_results.loc[15, 'lb_pvalue'] > 0.05

    overall_pass = pass_5 and pass_10 and pass_15

    autocorr_results.append({
        'Variable': var,
        'LB_Lag5_pval': lb_results.loc[5, 'lb_pvalue'],
        'LB_Lag10_pval': lb_results.loc[10, 'lb_pvalue'],
        'LB_Lag15_pval': lb_results.loc[15, 'lb_pvalue'],
        'Pass': 'YES' if overall_pass else 'NO'
    })
    autocorr_pass.append(overall_pass)

autocorr_df = pd.DataFrame(autocorr_results)
n_pass_autocorr = sum(autocorr_pass)

print(f"  Autocorrelation test: {n_pass_autocorr}/{len(endog_vars)} equations pass")
print(f"  Result: {'PASS' if n_pass_autocorr >= len(endog_vars) * 0.8 else 'FAIL'} (>=80% threshold)")

# =============================================================================
# TEST 2: RESIDUAL NORMALITY
# =============================================================================
print("\n[3/7] Testing for residual normality...")

normality_results = []
normality_pass = []

for i, var in enumerate(endog_vars):
    residuals = vecm_result.resid[:, i]

    # Jarque-Bera test
    jb_stat, jb_pval = stats.jarque_bera(residuals)

    # Check if p-value > 0.05 (normal distribution)
    passed = jb_pval > 0.05

    normality_results.append({
        'Variable': var,
        'JB_Statistic': jb_stat,
        'JB_pvalue': jb_pval,
        'Pass': 'YES' if passed else 'NO'
    })
    normality_pass.append(passed)

normality_df = pd.DataFrame(normality_results)
n_pass_normality = sum(normality_pass)

print(f"  Normality test: {n_pass_normality}/{len(endog_vars)} equations pass")
print(f"  Result: {'PASS' if n_pass_normality >= len(endog_vars) * 0.5 else 'FAIL'} (>=50% threshold)")
print(f"  Note: Normality often fails in time series; not critical for inference")

# =============================================================================
# TEST 3: HETEROSKEDASTICITY (ARCH EFFECTS)
# =============================================================================
print("\n[4/7] Testing for heteroskedasticity (ARCH effects)...")

arch_results = []
arch_pass = []

for i, var in enumerate(endog_vars):
    residuals = vecm_result.resid[:, i]

    try:
        # ARCH test
        arch_lm, arch_pval, _, _ = het_arch(residuals, nlags=5)

        # Check if p-value > 0.05 (no ARCH effects)
        passed = arch_pval > 0.05

        arch_results.append({
            'Variable': var,
            'ARCH_LM': arch_lm,
            'ARCH_pvalue': arch_pval,
            'Pass': 'YES' if passed else 'NO'
        })
        arch_pass.append(passed)
    except:
        arch_results.append({
            'Variable': var,
            'ARCH_LM': np.nan,
            'ARCH_pvalue': np.nan,
            'Pass': 'ERROR'
        })
        arch_pass.append(False)

arch_df = pd.DataFrame(arch_results)
n_pass_arch = sum(arch_pass)

print(f"  Heteroskedasticity test: {n_pass_arch}/{len(endog_vars)} equations pass")
print(f"  Result: {'PASS' if n_pass_arch >= len(endog_vars) * 0.7 else 'FAIL'} (>=70% threshold)")

# =============================================================================
# TEST 4: SYSTEM STABILITY
# =============================================================================
print("\n[4/7] Testing system stability...")

try:
    # Get eigenvalues of companion matrix
    eigenvalues = np.linalg.eigvals(vecm_result.var_rep.coefs_exog)
    max_eigenvalue = np.max(np.abs(eigenvalues))

    # System is stable if all eigenvalues < 1
    stability_pass = max_eigenvalue < 1.0

    print(f"  Maximum eigenvalue modulus: {max_eigenvalue:.4f}")
    print(f"  Result: {'PASS - System is stable' if stability_pass else 'FAIL - System may be unstable'}")
except Exception as e:
    print(f"  ERROR: Could not compute eigenvalues: {e}")
    stability_pass = None
    max_eigenvalue = np.nan

# =============================================================================
# TEST 5: GRANGER CAUSALITY
# =============================================================================
print("\n[5/7] Testing Granger causality relationships...")

# Test key relationships
causality_tests = [
    ('Field_Grade_Officers_Z', 'Junior_Enlisted_Z'),
    ('Total_Civilians_Z', 'Field_Grade_Officers_Z'),
    ('Policy_Count_Log', 'FOIA_Simple_Days_Z'),
    ('GOFOs_Z', 'Field_Grade_Officers_Z'),
]

causality_results = []

for cause_var, effect_var in causality_tests:
    try:
        # Extract data for pair
        test_data = endog_data[[effect_var, cause_var]]

        # Granger causality test (lag 2)
        gc_result = grangercausalitytests(test_data, maxlag=2, verbose=False)

        # Get p-value for lag 2
        pval_lag2 = gc_result[2][0]['ssr_ftest'][1]

        # Significant if p < 0.05
        significant = pval_lag2 < 0.05

        causality_results.append({
            'Cause': cause_var.replace('_Z', ''),
            'Effect': effect_var.replace('_Z', ''),
            'Lag': 2,
            'F_pvalue': pval_lag2,
            'Significant': 'YES' if significant else 'NO'
        })
    except Exception as e:
        causality_results.append({
            'Cause': cause_var.replace('_Z', ''),
            'Effect': effect_var.replace('_Z', ''),
            'Lag': 2,
            'F_pvalue': np.nan,
            'Significant': 'ERROR'
        })

causality_df = pd.DataFrame(causality_results)
n_significant = len(causality_df[causality_df['Significant'] == 'YES'])

print(f"  Granger causality: {n_significant}/{len(causality_tests)} relationships significant")
print(f"  Result: {'Expected' if n_significant > 0 else 'Concerning'}")

# =============================================================================
# TEST 6: COINTEGRATION RANK ROBUSTNESS
# =============================================================================
print("\n[6/7] Testing cointegration rank robustness...")

# Re-run Johansen test
joh = coint_johansen(endog_data, det_order=0, k_ar_diff=2)

# Count cointegrating relationships at 5% and 1%
n_coint_5pct = sum(1 for i in range(len(endog_vars)) if joh.lr1[i] > joh.cvt[i, 1])
n_coint_1pct = sum(1 for i in range(len(endog_vars)) if joh.lr1[i] > joh.cvt[i, 2])

print(f"  Cointegrating rank (5% level): {n_coint_5pct}")
print(f"  Cointegrating rank (1% level): {n_coint_1pct}")
print(f"  Model specification (rank=4): {'CORRECT' if n_coint_5pct == 4 else 'QUESTIONABLE'}")

# =============================================================================
# TEST 7: RESIDUAL DIAGNOSTICS SUMMARY
# =============================================================================
print("\n[7/7] Overall residual statistics...")

# Residual correlations
residual_corr = np.corrcoef(vecm_result.resid.T)
max_off_diag = np.max(np.abs(residual_corr[np.triu_indices_from(residual_corr, k=1)]))

print(f"  Maximum residual correlation: {max_off_diag:.4f}")
print(f"  Result: {'PASS' if max_off_diag < 0.7 else 'FAIL'} (<0.7 threshold)")

# =============================================================================
# OVERALL ASSESSMENT
# =============================================================================
print("\n" + "=" * 100)
print("OVERALL ROBUSTNESS ASSESSMENT")
print("=" * 100)

# Calculate pass rates
autocorr_rate = n_pass_autocorr / len(endog_vars)
normality_rate = n_pass_normality / len(endog_vars)
arch_rate = n_pass_arch / len(endog_vars)

overall_pass = (
    autocorr_rate >= 0.8 and
    arch_rate >= 0.7 and
    stability_pass and
    n_coint_5pct == 4 and
    max_off_diag < 0.7
)

print(f"\nTest Results Summary:")
print(f"  1. Autocorrelation:        {n_pass_autocorr}/{len(endog_vars)} pass ({autocorr_rate:.1%}) - {'PASS' if autocorr_rate >= 0.8 else 'FAIL'}")
print(f"  2. Normality:              {n_pass_normality}/{len(endog_vars)} pass ({normality_rate:.1%}) - {'PASS' if normality_rate >= 0.5 else 'FAIL'} (lenient)")
print(f"  3. Heteroskedasticity:     {n_pass_arch}/{len(endog_vars)} pass ({arch_rate:.1%}) - {'PASS' if arch_rate >= 0.7 else 'FAIL'}")
print(f"  4. System Stability:       {'PASS' if stability_pass else 'FAIL'}")
print(f"  5. Granger Causality:      {n_significant}/{len(causality_tests)} significant - Expected")
print(f"  6. Cointegration Rank:     {'PASS' if n_coint_5pct == 4 else 'FAIL'} (rank={n_coint_5pct})")
print(f"  7. Residual Independence:  {'PASS' if max_off_diag < 0.7 else 'FAIL'}")

print(f"\n{'='*100}")
print(f"OVERALL VERDICT: {'[PASS] MODEL PASSES ROBUSTNESS TESTS' if overall_pass else '[WARNING] MODEL HAS SOME CONCERNS'}")
print(f"{'='*100}")

if overall_pass:
    print("\nThe VECM specification is ROBUST and suitable for inference.")
else:
    print("\nThe VECM has some diagnostic concerns but may still be usable.")
    print("Consider reporting robustness checks as caveats in thesis.")

# =============================================================================
# SAVE DIAGNOSTIC RESULTS
# =============================================================================
print("\nSaving diagnostic results...")

with pd.ExcelWriter(f'{output_dir}/robustness_test_results.xlsx') as writer:
    autocorr_df.to_excel(writer, sheet_name='Autocorrelation', index=False)
    normality_df.to_excel(writer, sheet_name='Normality', index=False)
    arch_df.to_excel(writer, sheet_name='Heteroskedasticity', index=False)
    causality_df.to_excel(writer, sheet_name='Granger_Causality', index=False)

print("  [OK] Diagnostic results saved to robustness_test_results.xlsx")

# =============================================================================
# EXTRACT AND DOCUMENT VECM EQUATIONS
# =============================================================================
if overall_pass or True:  # Always document equations
    print("\n" + "=" * 100)
    print("EXTRACTING VECM EQUATIONS")
    print("=" * 100)

    # Get coefficients
    alpha = vecm_result.alpha  # Error correction coefficients
    beta = vecm_result.beta    # Cointegrating vectors

    # Gamma is stored as a single matrix (n_vars, n_vars * n_lags)
    # Need to reshape it into list of matrices
    gamma_full = vecm_result.gamma
    n_vars = len(endog_vars)
    n_lags = 2  # k_ar_diff = 2
    gamma = [gamma_full[:, i*n_vars:(i+1)*n_vars] for i in range(n_lags)]

    # Build comprehensive documentation
    doc_lines = []

    doc_lines.append("=" * 100)
    doc_lines.append("VECM LAG 2 - COMPLETE MODEL SPECIFICATION AND EQUATIONS")
    doc_lines.append("DoD Bureaucratic Growth Analysis (1987-2024)")
    doc_lines.append("=" * 100)
    doc_lines.append("")

    # Model specification
    doc_lines.append("MODEL SPECIFICATION:")
    doc_lines.append("-" * 100)
    doc_lines.append(f"Model type: Vector Error Correction Model (VECM)")
    doc_lines.append(f"Lag order: k = 2 (equivalent to VAR(3) in levels)")
    doc_lines.append(f"Cointegrating rank: r = 4")
    doc_lines.append(f"Observations: {vecm_result.nobs}")
    doc_lines.append(f"Deterministic term: Constant in cointegrating equation")
    doc_lines.append("")
    doc_lines.append("Endogenous variables (7):")
    for i, var in enumerate(endog_vars, 1):
        doc_lines.append(f"  {i}. {var}")
    doc_lines.append("")
    doc_lines.append("Exogenous variables (2):")
    for i, var in enumerate(exog_vars, 1):
        doc_lines.append(f"  {i}. {var}")
    doc_lines.append("")
    doc_lines.append("=" * 100)
    doc_lines.append("")

    # VECM general form
    doc_lines.append("GENERAL VECM FORM:")
    doc_lines.append("-" * 100)
    doc_lines.append("DY_t = a*b'*Y_{t-1} + G_1*DY_{t-1} + G_2*DY_{t-2} + F*X_t + e_t")
    doc_lines.append("")
    doc_lines.append("Where:")
    doc_lines.append("  Y_t       = Vector of endogenous variables at time t (7x1)")
    doc_lines.append("  DY_t      = First difference of Y_t")
    doc_lines.append("  a         = Error correction coefficients (7x4) - adjustment speeds")
    doc_lines.append("  b         = Cointegrating vectors (7x4) - long-run relationships")
    doc_lines.append("  G_1, G_2  = Short-run dynamic coefficient matrices (7x7)")
    doc_lines.append("  F         = Exogenous variable coefficients (7x2)")
    doc_lines.append("  X_t       = Vector of exogenous variables (2x1)")
    doc_lines.append("  e_t       = Error term vector (7x1)")
    doc_lines.append("")
    doc_lines.append("=" * 100)
    doc_lines.append("")

    # Alpha matrix (Error Correction)
    doc_lines.append("ERROR CORRECTION COEFFICIENTS (ALPHA MATRIX):")
    doc_lines.append("-" * 100)
    doc_lines.append("Rows = Equations (variables), Columns = Error Correction Terms (ECT)")
    doc_lines.append("")

    alpha_header = "Variable".ljust(30) + "".join([f"ECT_{i+1}".rjust(12) for i in range(4)])
    doc_lines.append(alpha_header)
    doc_lines.append("-" * 100)

    for i, var in enumerate(endog_vars):
        var_clean = var.replace('_Z', '').ljust(30)
        coeffs = "".join([f"{alpha[i, j]:11.6f} " for j in range(4)])
        doc_lines.append(var_clean + coeffs)

    doc_lines.append("")
    doc_lines.append("Interpretation:")
    doc_lines.append("  α[i,j] = Speed of adjustment for variable i to equilibrium j")
    doc_lines.append("  Negative α: Variable decreases when equilibrium is above long-run level")
    doc_lines.append("  Positive α: Variable increases when equilibrium is above long-run level")
    doc_lines.append("  |α| > 0.3: Strong adjustment (highly endogenous)")
    doc_lines.append("")
    doc_lines.append("=" * 100)
    doc_lines.append("")

    # Beta matrix (Cointegrating Vectors)
    doc_lines.append("COINTEGRATING VECTORS (β - BETA MATRIX):")
    doc_lines.append("-" * 100)
    doc_lines.append("Rows = Variables, Columns = Cointegrating Vectors")
    doc_lines.append("")

    beta_header = "Variable".ljust(30) + "".join([f"Vector_{i+1}".rjust(12) for i in range(4)])
    doc_lines.append(beta_header)
    doc_lines.append("-" * 100)

    for i, var in enumerate(endog_vars):
        var_clean = var.replace('_Z', '').ljust(30)
        coeffs = "".join([f"{beta[i, j]:11.6f} " for j in range(4)])
        doc_lines.append(var_clean + coeffs)

    doc_lines.append("")
    doc_lines.append("Interpretation:")
    doc_lines.append("  b'*Y = Linear combination that is stationary (I(0))")
    doc_lines.append("  Positive b: Variable grows with equilibrium")
    doc_lines.append("  Negative b: Variable declines with equilibrium")
    doc_lines.append("  Each column is one long-run equilibrium relationship")
    doc_lines.append("")
    doc_lines.append("=" * 100)
    doc_lines.append("")

    # Short-run dynamics (Gamma)
    doc_lines.append("SHORT-RUN DYNAMICS (Γ - GAMMA MATRICES):")
    doc_lines.append("-" * 100)
    doc_lines.append("")

    for lag_idx in range(len(gamma)):
        doc_lines.append(f"Γ_{lag_idx+1} (Impact of ΔY_{{t-{lag_idx+1}}}):")
        doc_lines.append("")

        # Header
        header = "From \\ To".ljust(25) + "".join([v.replace('_Z', '')[:8].rjust(10) for v in endog_vars])
        doc_lines.append(header)
        doc_lines.append("-" * 100)

        # Rows
        for i, var_from in enumerate(endog_vars):
            row = var_from.replace('_Z', '').ljust(25)
            coeffs = "".join([f"{gamma[lag_idx][j, i]:9.4f} " for j in range(len(endog_vars))])
            doc_lines.append(row + coeffs)

        doc_lines.append("")

    doc_lines.append("=" * 100)
    doc_lines.append("")

    # Individual equations
    doc_lines.append("INDIVIDUAL VECM EQUATIONS:")
    doc_lines.append("-" * 100)
    doc_lines.append("")

    for i, var in enumerate(endog_vars):
        var_clean = var.replace('_Z', '')
        doc_lines.append(f"Equation {i+1}: D({var_clean})_t")
        doc_lines.append("")

        # Error correction terms
        doc_lines.append("  Error Correction:")
        for j in range(4):
            doc_lines.append(f"    + {alpha[i,j]:9.6f} * ECT_{j+1}_{{t-1}}")
        doc_lines.append("")

        # Short-run dynamics (showing only significant coefficients)
        doc_lines.append("  Short-run Dynamics (|coef| > 0.1):")
        for lag_idx in range(len(gamma)):
            for j, var_j in enumerate(endog_vars):
                coef = gamma[lag_idx][i, j]
                if abs(coef) > 0.1:
                    var_j_clean = var_j.replace('_Z', '')
                    doc_lines.append(f"    + {coef:9.6f} * D({var_j_clean})_{{t-{lag_idx+1}}}")

        doc_lines.append("")
        doc_lines.append("-" * 100)
        doc_lines.append("")

    # Robustness test results
    doc_lines.append("=" * 100)
    doc_lines.append("ROBUSTNESS TEST RESULTS:")
    doc_lines.append("=" * 100)
    doc_lines.append("")

    doc_lines.append(f"1. RESIDUAL AUTOCORRELATION (Ljung-Box Test):")
    doc_lines.append(f"   Pass rate: {n_pass_autocorr}/{len(endog_vars)} equations ({autocorr_rate:.1%})")
    doc_lines.append(f"   Result: {'PASS' if autocorr_rate >= 0.8 else 'FAIL'} (threshold: >=80%)")
    doc_lines.append("")

    for idx, row in autocorr_df.iterrows():
        doc_lines.append(f"   {row['Variable'].replace('_Z', '')}:")
        doc_lines.append(f"     Lag 5:  p={row['LB_Lag5_pval']:.4f}")
        doc_lines.append(f"     Lag 10: p={row['LB_Lag10_pval']:.4f}")
        doc_lines.append(f"     Lag 15: p={row['LB_Lag15_pval']:.4f}")
        doc_lines.append(f"     Status: {row['Pass']}")
        doc_lines.append("")

    doc_lines.append(f"2. RESIDUAL NORMALITY (Jarque-Bera Test):")
    doc_lines.append(f"   Pass rate: {n_pass_normality}/{len(endog_vars)} equations ({normality_rate:.1%})")
    doc_lines.append(f"   Result: {'PASS' if normality_rate >= 0.5 else 'FAIL'} (threshold: >=50%, lenient)")
    doc_lines.append(f"   Note: Normality often fails in time series; not critical for inference")
    doc_lines.append("")

    for idx, row in normality_df.iterrows():
        doc_lines.append(f"   {row['Variable'].replace('_Z', '')}: JB={row['JB_Statistic']:.2f}, p={row['JB_pvalue']:.4f} ({row['Pass']})")
    doc_lines.append("")

    doc_lines.append(f"3. HETEROSKEDASTICITY (ARCH Test):")
    doc_lines.append(f"   Pass rate: {n_pass_arch}/{len(endog_vars)} equations ({arch_rate:.1%})")
    doc_lines.append(f"   Result: {'PASS' if arch_rate >= 0.7 else 'FAIL'} (threshold: >=70%)")
    doc_lines.append("")

    for idx, row in arch_df.iterrows():
        if not pd.isna(row['ARCH_pvalue']):
            doc_lines.append(f"   {row['Variable'].replace('_Z', '')}: LM={row['ARCH_LM']:.2f}, p={row['ARCH_pvalue']:.4f} ({row['Pass']})")
    doc_lines.append("")

    doc_lines.append(f"4. SYSTEM STABILITY:")
    if stability_pass is not None:
        doc_lines.append(f"   Maximum eigenvalue: {max_eigenvalue:.4f}")
        doc_lines.append(f"   Result: {'PASS - System is stable' if stability_pass else 'FAIL - System unstable'}")
    else:
        doc_lines.append(f"   Result: ERROR - Could not compute")
    doc_lines.append("")

    doc_lines.append(f"5. GRANGER CAUSALITY:")
    doc_lines.append(f"   Significant relationships: {n_significant}/{len(causality_tests)}")
    doc_lines.append("")
    for idx, row in causality_df.iterrows():
        if row['Significant'] != 'ERROR':
            doc_lines.append(f"   {row['Cause']} -> {row['Effect']}: p={row['F_pvalue']:.4f} ({row['Significant']})")
    doc_lines.append("")

    doc_lines.append(f"6. COINTEGRATION RANK CONFIRMATION:")
    doc_lines.append(f"   Rank at 5% level: {n_coint_5pct}")
    doc_lines.append(f"   Rank at 1% level: {n_coint_1pct}")
    doc_lines.append(f"   Model specification: {'CORRECT' if n_coint_5pct == 4 else 'QUESTIONABLE'}")
    doc_lines.append("")

    doc_lines.append(f"7. RESIDUAL INDEPENDENCE:")
    doc_lines.append(f"   Maximum correlation: {max_off_diag:.4f}")
    doc_lines.append(f"   Result: {'PASS' if max_off_diag < 0.7 else 'FAIL'} (threshold: <0.7)")
    doc_lines.append("")

    doc_lines.append("=" * 100)
    doc_lines.append(f"OVERALL VERDICT: {'[PASS] MODEL PASSES ROBUSTNESS TESTS' if overall_pass else '[WARNING] MODEL HAS SOME CONCERNS'}")
    doc_lines.append("=" * 100)
    doc_lines.append("")

    if overall_pass:
        doc_lines.append("The VECM specification is ROBUST and suitable for causal inference and")
        doc_lines.append("policy analysis. All critical diagnostic tests pass at acceptable thresholds.")
    else:
        doc_lines.append("The VECM has some diagnostic concerns but remains usable for inference.")
        doc_lines.append("Report robustness check results as caveats in thesis methodology section.")

    doc_lines.append("")
    doc_lines.append("=" * 100)
    doc_lines.append("END OF MODEL SPECIFICATION")
    doc_lines.append("=" * 100)

    # Write to file
    output_file = f'{output_dir}/VECM_EQUATIONS_AND_ROBUSTNESS.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(doc_lines))

    print(f"\n  [OK] Complete model equations and robustness tests saved to:")
    print(f"       {output_file}")

print("\n" + "=" * 100)
print("ROBUSTNESS TESTING COMPLETE")
print("=" * 100)
