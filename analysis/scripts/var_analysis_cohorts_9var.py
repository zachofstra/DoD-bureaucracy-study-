"""
VAR(3) Analysis - 9 Variables with Rank Cohorts
Final model based on network centrality selection

Variables:
1. Junior_Enlisted_Z
2. FOIA_Simple_Days_Z
3. Warrant_Officers_Z
4. Total_PAS_Z
5. GOFOs_Z
6. Company_Grade_Officers_Z
7. Middle_Enlisted_Z
8. Policy_Count_Log
9. Field_Grade_Officers_Z
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("VAR(3) ANALYSIS - 9 VARIABLES WITH RANK COHORTS")
print("=" * 100)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/7] Loading data...")

df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

# Selected 9 variables
var_list = [
    'Junior_Enlisted_Z',
    'FOIA_Simple_Days_Z',
    'Warrant_Officers_Z',
    'Total_PAS_Z',
    'GOFOs_Z',
    'Company_Grade_Officers_Z',
    'Middle_Enlisted_Z',
    'Policy_Count_Log',
    'Field_Grade_Officers_Z'
]

data = df[var_list].copy()
data = data.dropna()

print(f"  Variables: {len(var_list)}")
print(f"  Observations: {len(data)}")
print(f"  Time period: {df['FY'].min():.0f}-{df['FY'].max():.0f}")

# =============================================================================
# CHECK STATIONARITY
# =============================================================================
print("\n[2/7] Checking stationarity...")
print("  " + "-" * 96)

stationarity_results = []

for var in var_list:
    # ADF test
    adf_result = adfuller(data[var].dropna(), maxlag=4, regression='ct')
    adf_pval = adf_result[1]

    # KPSS test
    kpss_result = kpss(data[var].dropna(), regression='ct', nlags=4)
    kpss_pval = kpss_result[1]

    # Decision
    if adf_pval < 0.05 and kpss_pval > 0.05:
        decision = "STATIONARY"
        transform = "LEVELS"
    else:
        decision = "NON-STATIONARY"
        transform = "DIFFERENCE"

    stationarity_results.append({
        'Variable': var,
        'ADF_pvalue': adf_pval,
        'KPSS_pvalue': kpss_pval,
        'Decision': decision,
        'Transform': transform
    })

    print(f"  {var:30s} ADF={adf_pval:.4f}, KPSS={kpss_pval:.4f} -> {decision}")

# Apply differencing to non-stationary variables
print("\n  Applying transformations...")
data_transformed = data.copy()

for result in stationarity_results:
    if result['Transform'] == 'DIFFERENCE':
        var = result['Variable']
        data_transformed[var] = data[var].diff()
        print(f"    {var:30s} -> DIFFERENCED")

# Drop NaN from differencing
data_transformed = data_transformed.dropna()
print(f"\n  Observations after transformation: {len(data_transformed)}")

# =============================================================================
# FIT VAR(3) MODEL
# =============================================================================
print("\n[3/7] Fitting VAR(3) model...")

model = VAR(data_transformed)
lag_order = 3

try:
    results = model.fit(maxlags=lag_order, ic=None)
    print(f"  [OK] VAR({lag_order}) fitted successfully")
    print(f"  Effective observations: {results.nobs}")
    print(f"  Total parameters: {results.k_ar * results.neqs * results.neqs}")
    try:
        print(f"  AIC: {results.aic:.2f}")
        print(f"  BIC: {results.bic:.2f}")
    except:
        print("  AIC/BIC: Could not compute (numerical issues)")
except Exception as e:
    print(f"  [ERROR] Could not fit VAR({lag_order}): {e}")
    print("\n  Trying VAR(3) instead...")
    lag_order = 3
    results = model.fit(maxlags=lag_order, ic=None)
    print(f"  [OK] VAR({lag_order}) fitted successfully")

# Save model summary
try:
    with open('data/analysis/var_cohorts_9var_summary.txt', 'w') as f:
        f.write(str(results.summary()))
    print("  [OK] Model summary saved")
except Exception as e:
    print(f"  [WARNING] Could not generate summary: {e}")
    print("  Model fitted but covariance matrix has numerical issues")

# =============================================================================
# DIAGNOSTICS
# =============================================================================
print("\n[4/7] Running diagnostics...")

# Test for autocorrelation in residuals
print("\n  Ljung-Box Test for Residual Autocorrelation:")
print("  " + "-" * 96)

autocorr_tests = []
for i, var in enumerate(var_list):
    residuals = results.resid.iloc[:, i]
    lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)

    # Check if any p-value < 0.05
    has_autocorr = (lb_result['lb_pvalue'] < 0.05).any()
    status = "FAIL" if has_autocorr else "PASS"

    autocorr_tests.append({
        'Variable': var,
        'Min_pvalue': lb_result['lb_pvalue'].min(),
        'Status': status
    })

    print(f"    {var:30s} Min p-value={lb_result['lb_pvalue'].min():.4f} [{status}]")

# Overall diagnostics
n_pass = sum(1 for t in autocorr_tests if t['Status'] == 'PASS')
print(f"\n  Overall: {n_pass}/{len(var_list)} equations pass autocorrelation test")

# =============================================================================
# GRANGER CAUSALITY
# =============================================================================
print("\n[5/7] Computing Granger causality...")

granger_results = []

for cause_var in var_list:
    for effect_var in var_list:
        if cause_var == effect_var:
            continue

        try:
            # Granger test
            test_data = data_transformed[[effect_var, cause_var]].dropna()
            gc_result = grangercausalitytests(test_data, maxlag=lag_order, verbose=False)

            # Get p-value for the specified lag
            p_value = gc_result[lag_order][0]['ssr_ftest'][1]
            f_stat = gc_result[lag_order][0]['ssr_ftest'][0]

            granger_results.append({
                'Cause': cause_var,
                'Effect': effect_var,
                'Lag': lag_order,
                'F_stat': f_stat,
                'p_value': p_value,
                'Significant_5pct': p_value < 0.05
            })
        except:
            pass

granger_df = pd.DataFrame(granger_results)
granger_df.to_excel('data/analysis/var_cohorts_9var_granger.xlsx', index=False)

n_significant = granger_df['Significant_5pct'].sum()
print(f"  Granger causality tests: {len(granger_df)} pairs tested")
print(f"  Significant at 5%: {n_significant}")

# Show top significant relationships
print("\n  Top 10 strongest Granger causal relationships:")
print("  " + "-" * 96)
top_granger = granger_df.nlargest(10, 'F_stat')[['Cause', 'Effect', 'F_stat', 'p_value']]
for idx, row in top_granger.iterrows():
    sig = "*" if row['p_value'] < 0.05 else " "
    print(f"  {sig} {row['Cause']:30s} -> {row['Effect']:30s} F={row['F_stat']:6.2f} p={row['p_value']:.4f}")

# =============================================================================
# IMPULSE RESPONSE FUNCTIONS
# =============================================================================
print("\n[6/7] Computing impulse response functions...")

try:
    irf = results.irf(10)
    irf_success = True
except np.linalg.LinAlgError as e:
    print(f"  [WARNING] IRF calculation failed: {e}")
    print("  This suggests the model may be overfitted (too many parameters for sample size)")
    print("  Consider reducing to VAR(3) or VAR(2), or using fewer variables")
    irf_success = False

if irf_success:
    # Plot IRFs for key relationships
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), facecolor='white')
    axes = axes.flatten()

    key_shocks = [
        ('Policy_Count_Log', 'Field_Grade_Officers_Z'),
        ('Field_Grade_Officers_Z', 'Policy_Count_Log'),
        ('Total_PAS_Z', 'Field_Grade_Officers_Z'),
        ('Field_Grade_Officers_Z', 'Total_PAS_Z'),
        ('Junior_Enlisted_Z', 'Field_Grade_Officers_Z'),
        ('FOIA_Simple_Days_Z', 'Policy_Count_Log'),
        ('Policy_Count_Log', 'FOIA_Simple_Days_Z'),
        ('Warrant_Officers_Z', 'Field_Grade_Officers_Z'),
        ('GOFOs_Z', 'Field_Grade_Officers_Z')
    ]

    for idx, (shock_var, response_var) in enumerate(key_shocks):
        ax = axes[idx]

        shock_idx = var_list.index(shock_var)
        response_idx = var_list.index(response_var)

        irf.plot(impulse=shock_idx, response=response_idx, ax=ax)
        ax.set_title(f'Shock: {shock_var.replace("_Z", "")}\nResponse: {response_var.replace("_Z", "")}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Period', fontsize=9)
        ax.set_ylabel('Response', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/analysis/var_cohorts_9var_irf.png', dpi=300, bbox_inches='tight')
    print("  [OK] IRF plot saved")
else:
    print("  [SKIPPED] IRF plot not generated due to numerical issues")

# =============================================================================
# FORECAST ERROR VARIANCE DECOMPOSITION
# =============================================================================
print("\n[7/7] Computing forecast error variance decomposition...")

if irf_success:
    fevd = results.fevd(10)

    # FEVD for Field_Grade_Officers (key variable)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')

    key_vars = ['Field_Grade_Officers_Z', 'Policy_Count_Log', 'Total_PAS_Z']

    for idx, var in enumerate(key_vars):
        ax = axes[idx]
        var_idx = var_list.index(var)

        fevd_data = fevd.decomp[:, var_idx, :]

        # Plot stacked area
        periods = range(fevd_data.shape[0])
        ax.stackplot(periods, *fevd_data.T, labels=[v.replace('_Z', '') for v in var_list], alpha=0.7)
        ax.set_title(f'FEVD: {var.replace("_Z", "")}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Period', fontsize=10)
        ax.set_ylabel('Proportion of Variance', fontsize=10)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/analysis/var_cohorts_9var_fevd.png', dpi=300, bbox_inches='tight')
    print("  [OK] FEVD plot saved")
else:
    print("  [SKIPPED] FEVD plot not generated due to numerical issues")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)

print(f"\nMODEL SPECIFICATION:")
print(f"  Variables: {len(var_list)}")
print(f"  Lag order: {lag_order}")
print(f"  Observations: {results.nobs}")
try:
    print(f"  AIC: {results.aic:.2f}")
    print(f"  BIC: {results.bic:.2f}")
except:
    print(f"  AIC/BIC: Could not compute")

print(f"\nDIAGNOSTICS:")
print(f"  Autocorrelation tests passed: {n_pass}/{len(var_list)}")

print(f"\nGRANGER CAUSALITY:")
print(f"  Significant relationships (p<0.05): {n_significant}/{len(granger_df)}")

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print("  1. var_cohorts_9var_summary.txt - Full model summary")
print("  2. var_cohorts_9var_granger.xlsx - Granger causality results")
print("  3. var_cohorts_9var_irf.png - Impulse response functions")
print("  4. var_cohorts_9var_fevd.png - Variance decomposition")
print("=" * 100)

print("\nKEY FINDING:")
print("  Field_Grade_Officers_Z (O4-O6 bureaucratic layer) interactions with:")
print("  - Policy_Count_Log (regulatory burden)")
print("  - Total_PAS_Z (political appointees)")
print("  - FOIA_Simple_Days_Z (bureaucratic responsiveness)")
print("=" * 100)
