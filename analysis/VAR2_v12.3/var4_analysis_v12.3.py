"""
VAR(2) Model - DoD Bureaucratic Growth Analysis v12.3
Full Vector Autoregression with 2 lags on 8 selected variables
(Lag 2 optimal from LASSO analysis; Lag 4 causes overparameterization)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = 'analysis/complete_normalized_dataset_v12.3.xlsx'
OUTPUT_DIR = 'analysis/VAR2_v12.3'
LAG_ORDER = 2  # Changed from 4 due to sample size constraints

# 8 variables from Granger causality analysis
SELECTED_VARS = [
    'Warrant_Officers_Z',
    'Policy_Count_Log',
    'Company_Grade_Officers_Z',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z',
    'Junior_Enlisted_Z',
    'Field_Grade_Officers_Z',
    'Total_Civilians_Z'
]

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

print("=" * 100)
print(f"VAR({LAG_ORDER}) MODEL ANALYSIS - v12.3 DATASET")
print("8 Variables Selected from Pairwise Granger Causality")
print("=" * 100)

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
print("\n[1/7] Loading data...")

df = pd.read_excel(DATA_FILE)
data = df[SELECTED_VARS].dropna()

print(f"  Observations: {len(data)}")
print(f"  Variables: {len(SELECTED_VARS)}")
print("\n  Selected variables:")
for i, var in enumerate(SELECTED_VARS, 1):
    print(f"    {i}. {var}")

# =============================================================================
# STATIONARITY TESTS
# =============================================================================
print("\n[2/7] Testing stationarity (ADF tests)...")

stationarity_results = []
for var in SELECTED_VARS:
    adf_result = adfuller(data[var], autolag='AIC')
    stationarity_results.append({
        'Variable': var,
        'ADF_Statistic': adf_result[0],
        'p_value': adf_result[1],
        'Lags_Used': adf_result[2],
        'Stationary': 'Yes' if adf_result[1] < 0.05 else 'No'
    })

stationarity_df = pd.DataFrame(stationarity_results)
stationarity_df.to_excel(f'{OUTPUT_DIR}/stationarity_tests.xlsx', index=False)

print("\n  Stationarity Test Results:")
print("  " + "-" * 80)
for _, row in stationarity_df.iterrows():
    status = "[STATIONARY]" if row['Stationary'] == 'Yes' else "[NON-STATIONARY]"
    print(f"    {row['Variable']:30s} ADF={row['ADF_Statistic']:8.4f}, p={row['p_value']:.4f} {status}")

stationary_count = (stationarity_df['Stationary'] == 'Yes').sum()
print(f"\n  Stationary variables: {stationary_count}/{len(SELECTED_VARS)}")

# =============================================================================
# ESTIMATE VAR(4) MODEL
# =============================================================================
print(f"\n[3/7] Estimating VAR({LAG_ORDER}) model...")

model = VAR(data)
var_result = model.fit(maxlags=LAG_ORDER, ic=None)

print(f"\n  Model estimated successfully")
print(f"  Lag order: {var_result.k_ar}")
print(f"  Number of equations: {var_result.neqs}")
print(f"  Number of coefficients per equation: {var_result.k_ar * var_result.neqs + 1}")  # +1 for constant
print(f"  Total observations used: {var_result.nobs}")

# =============================================================================
# MODEL SUMMARY AND DIAGNOSTICS
# =============================================================================
print(f"\n[4/7] Generating model summary and diagnostics...")

# Save full summary
with open(f'{OUTPUT_DIR}/var4_model_summary.txt', 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("VAR(4) MODEL SUMMARY - v12.3 DATASET\n")
    f.write("=" * 100 + "\n\n")
    f.write(str(var_result.summary()))
    f.write("\n\n" + "=" * 100 + "\n")
    f.write("DIAGNOSTIC TESTS\n")
    f.write("=" * 100 + "\n\n")

# Calculate R-squared manually for each equation
rsquared_data = []
residuals = var_result.resid.values if hasattr(var_result.resid, 'values') else var_result.resid
for i, var in enumerate(SELECTED_VARS):
    # Get residuals for this equation
    resid = residuals[:, i]
    y_actual = data[var].values[-len(resid):]  # Trim to match residuals
    y_pred = y_actual - resid

    # Calculate R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Adjusted R-squared
    n = len(resid)
    k = LAG_ORDER * len(SELECTED_VARS) + 1  # number of parameters
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k) if (n - k) > 0 else 0

    rsquared_data.append({
        'Variable': var,
        'R_squared': r_squared,
        'Adj_R_squared': adj_r_squared
    })

rsquared_df = pd.DataFrame(rsquared_data)
rsquared_df.to_excel(f'{OUTPUT_DIR}/model_fit_rsquared.xlsx', index=False)

print("\n  R-squared by equation:")
print("  " + "-" * 60)
for _, row in rsquared_df.iterrows():
    print(f"    {row['Variable']:30s} R2={row['R_squared']:.4f}, Adj-R2={row['Adj_R_squared']:.4f}")

avg_rsq = rsquared_df['R_squared'].mean()
print(f"\n  Average R-squared: {avg_rsq:.4f}")

# =============================================================================
# EXTRACT AND SAVE COEFFICIENTS
# =============================================================================
print(f"\n[5/7] Extracting coefficient matrices...")

# Get coefficient matrices for each lag
# params structure: rows are coefficients for each variable at each lag + const
# columns are equations (one for each dependent variable)
for lag in range(1, LAG_ORDER + 1):
    # Extract coefficients for this lag across all equations
    # Create a matrix: rows=from_var, cols=to_var (equation)
    coef_matrix = pd.DataFrame(index=SELECTED_VARS, columns=SELECTED_VARS)

    for to_var in SELECTED_VARS:
        for from_var in SELECTED_VARS:
            param_name = f'L{lag}.{from_var}'
            if param_name in var_result.params.index:
                coef_matrix.loc[from_var, to_var] = var_result.params.loc[param_name, to_var]
            else:
                coef_matrix.loc[from_var, to_var] = 0.0

    coef_matrix = coef_matrix.astype(float)
    coef_matrix.to_excel(f'{OUTPUT_DIR}/coefficients_lag{lag}.xlsx')
    print(f"  Saved coefficients for lag {lag}")

# Save constant terms
const_df = pd.DataFrame({
    'Equation': SELECTED_VARS,
    'Constant': [var_result.params.loc['const', var] for var in SELECTED_VARS]
})
const_df.to_excel(f'{OUTPUT_DIR}/constants.xlsx', index=False)
print(f"  Saved constant terms")

# =============================================================================
# GRANGER CAUSALITY TESTS
# =============================================================================
print(f"\n[6/7] Running Granger causality tests on fitted model...")

granger_results = []
for caused_var in SELECTED_VARS:
    for causing_var in SELECTED_VARS:
        if caused_var == causing_var:
            continue

        try:
            gc_test = var_result.test_causality(caused_var, causing_var, kind='f')
            granger_results.append({
                'Caused': caused_var,
                'Causing': causing_var,
                'F_statistic': gc_test.test_statistic,
                'p_value': gc_test.pvalue,
                'df_num': gc_test.df,
                'df_denom': gc_test.df_denom,
                'Significant_5pct': gc_test.pvalue < 0.05,
                'Significant_1pct': gc_test.pvalue < 0.01
            })
        except:
            continue

granger_causality_df = pd.DataFrame(granger_results)

if len(granger_causality_df) > 0:
    granger_causality_df = granger_causality_df.sort_values('p_value')
    granger_causality_df.to_excel(f'{OUTPUT_DIR}/granger_causality_tests.xlsx', index=False)

    sig_5pct = granger_causality_df['Significant_5pct'].sum()
    sig_1pct = granger_causality_df['Significant_1pct'].sum()
    print(f"\n  Granger causality test results:")
    print(f"    Total tests: {len(granger_causality_df)}")
    print(f"    Significant at 5%: {sig_5pct}")
    print(f"    Significant at 1%: {sig_1pct}")
else:
    print(f"\n  No Granger causality tests completed successfully")
    sig_5pct = 0
    sig_1pct = 0

# =============================================================================
# IMPULSE RESPONSE FUNCTIONS
# =============================================================================
print(f"\n[7/7] Computing impulse response functions...")

# Compute IRFs (10 periods ahead)
irf = var_result.irf(10)

# Plot IRFs (plot all at once - default behavior)
try:
    fig = irf.plot(orth=False, figsize=(24, 20))
    plt.suptitle('Impulse Response Functions (Non-Orthogonalized) - All Variables',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(f'{OUTPUT_DIR}/impulse_response_functions_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  IRF plots saved")
except Exception as e:
    print(f"  Warning: Could not plot IRFs: {e}")

# Save IRF data
irf_data = []
for i, impulse_var in enumerate(SELECTED_VARS):
    for j, response_var in enumerate(SELECTED_VARS):
        irf_values = irf.irfs[:, j, i]  # [periods, response, impulse]
        for period in range(len(irf_values)):
            irf_data.append({
                'Impulse': impulse_var,
                'Response': response_var,
                'Period': period,
                'IRF_Value': irf_values[period]
            })

irf_df = pd.DataFrame(irf_data)
irf_df.to_excel(f'{OUTPUT_DIR}/impulse_response_data.xlsx', index=False)
print("  IRF data saved")

# =============================================================================
# FORECAST ERROR VARIANCE DECOMPOSITION
# =============================================================================
print("\n  Computing forecast error variance decomposition...")

fevd = var_result.fevd(10)

# Save FEVD for each variable
for i, var in enumerate(SELECTED_VARS):
    fevd_var = pd.DataFrame(
        fevd.decomp[:, i, :],
        columns=SELECTED_VARS
    )
    fevd_var.insert(0, 'Period', range(len(fevd_var)))
    fevd_var.to_excel(f'{OUTPUT_DIR}/fevd_{var}.xlsx', index=False)

print("  FEVD saved for all variables")

# =============================================================================
# RESIDUAL DIAGNOSTICS
# =============================================================================
print("\n  Performing residual diagnostics...")

residuals_array = var_result.resid.values if hasattr(var_result.resid, 'values') else var_result.resid

# Normality tests (Jarque-Bera)
normality_results = []
for i, var in enumerate(SELECTED_VARS):
    jb_stat, jb_pval = stats.jarque_bera(residuals_array[:, i])
    normality_results.append({
        'Variable': var,
        'JB_Statistic': jb_stat,
        'p_value': jb_pval,
        'Normal': 'Yes' if jb_pval > 0.05 else 'No'
    })

normality_df = pd.DataFrame(normality_results)
normality_df.to_excel(f'{OUTPUT_DIR}/residual_normality_tests.xlsx', index=False)

normal_count = (normality_df['Normal'] == 'Yes').sum()
print(f"  Residual normality: {normal_count}/{len(SELECTED_VARS)} pass Jarque-Bera test")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 100)
print(f"VAR({LAG_ORDER}) ANALYSIS COMPLETE")
print("=" * 100)

print(f"\nMODEL SPECIFICATION:")
print(f"  Lag order: {LAG_ORDER}")
print(f"  Number of variables: {len(SELECTED_VARS)}")
print(f"  Observations used: {var_result.nobs}")
print(f"  Total parameters: {len(SELECTED_VARS) * (len(SELECTED_VARS) * LAG_ORDER + 1)}")

print(f"\nMODEL FIT:")
print(f"  Average R-squared: {avg_rsq:.4f}")
print(f"  AIC: {var_result.aic:.2f}")
print(f"  BIC: {var_result.bic:.2f}")
print(f"  HQIC: {var_result.hqic:.2f}")

print(f"\nDIAGNOSTICS:")
print(f"  Stationary variables: {stationary_count}/{len(SELECTED_VARS)}")
print(f"  Normally distributed residuals: {normal_count}/{len(SELECTED_VARS)}")
print(f"  Significant Granger causalities (5%): {sig_5pct}/{len(granger_causality_df)}")

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print(f"  1. var4_model_summary.txt - Full model output")
print(f"  2. stationarity_tests.xlsx - ADF test results")
print(f"  3. model_fit_rsquared.xlsx - R-squared for each equation")
print(f"  4. coefficients_lag*.xlsx - Coefficient matrices for each lag (1-4)")
print(f"  5. constants.xlsx - Constant terms")
print(f"  6. granger_causality_tests.xlsx - All pairwise Granger tests")
print(f"  7. impulse_response_functions_*.png - IRF plots")
print(f"  8. impulse_response_data.xlsx - IRF numerical values")
print(f"  9. fevd_*.xlsx - Forecast error variance decomposition for each variable")
print(f" 10. residual_normality_tests.xlsx - Jarque-Bera tests")

print("\n" + "=" * 100)
print("INTERPRETATION:")
print("=" * 100)
print(f"  - VAR({LAG_ORDER}) captures short-run dynamic relationships over {LAG_ORDER} years")
print("  - Lag 2 is optimal (from LASSO analysis and sample size constraints)")
print("  - Use with VECM to compare short-run vs long-run dynamics")
print("  - IRFs show how shocks propagate through the system")
print("  - FEVD reveals which variables drive variance in each equation")
print("  - Granger tests identify causal relationships in the VAR framework")
print("=" * 100)
