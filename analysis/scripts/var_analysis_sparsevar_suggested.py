"""
VAR(2) Analysis - SparseVAR Suggested Variables
Testing if SparseVAR's top variables hold up in proper VAR framework

Endogenous Variables (7):
1. Warrant_Officers_Z
2. Total_PAS_Z
3. Field_Grade_Officers_Z
4. FOIA_Complex_Days_Z
5. Junior_Enlisted_Z
6. President_Republican
7. Total_Civilians_Z

Exogenous Variables (2):
- GDP_Growth
- Major_Conflict
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
print("VAR(2) ANALYSIS - SPARSEVAR SUGGESTED VARIABLES")
print("=" * 100)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/7] Loading data...")

df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

# Endogenous variables (from SparseVAR top centrality)
endog_vars = [
    'Warrant_Officers_Z',
    'Total_PAS_Z',
    'Field_Grade_Officers_Z',
    'FOIA_Complex_Days_Z',
    'Junior_Enlisted_Z',
    'President_Republican',
    'Total_Civilians_Z'
]

# Exogenous variables
exog_vars = [
    'GDP_Growth',
    'Major_Conflict'
]

all_vars = endog_vars + exog_vars
data = df[all_vars].copy()
data = data.dropna()

print(f"  Endogenous variables: {len(endog_vars)}")
print(f"  Exogenous variables: {len(exog_vars)}")
print(f"  Observations: {len(data)}")
print(f"  Time period: {df['FY'].min():.0f}-{df['FY'].max():.0f}")

# =============================================================================
# CHECK STATIONARITY
# =============================================================================
print("\n[2/7] Checking stationarity...")
print("  " + "-" * 96)

stationarity_results = []

for var in endog_vars:
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
endog_data = data[endog_vars].copy()
exog_data = data[exog_vars].copy()

for result in stationarity_results:
    if result['Transform'] == 'DIFFERENCE':
        var = result['Variable']
        endog_data[var] = endog_data[var].diff()
        print(f"    {var:30s} -> DIFFERENCED")

# Align endogenous and exogenous data after differencing
endog_data = endog_data.dropna()
exog_data = exog_data.loc[endog_data.index]

print(f"\n  Observations after transformation: {len(endog_data)}")

# =============================================================================
# FIT VAR(2) MODEL WITH EXOGENOUS VARIABLES
# =============================================================================
print("\n[3/7] Fitting VAR(2) model with exogenous controls...")

model = VAR(endog_data, exog=exog_data)
lag_order = 2

try:
    results = model.fit(maxlags=lag_order, ic=None)
    print(f"  [OK] VAR({lag_order}) fitted successfully")
    print(f"  Effective observations: {results.nobs}")
    print(f"  Endogenous parameters: {results.k_ar * results.neqs * results.neqs}")
    print(f"  Exogenous parameters: {len(exog_vars) * results.neqs}")
    try:
        print(f"  AIC: {results.aic:.2f}")
        print(f"  BIC: {results.bic:.2f}")
    except:
        print("  AIC/BIC: Could not compute (numerical issues)")
except Exception as e:
    print(f"  [ERROR] Could not fit VAR({lag_order}): {e}")
    exit(1)

# Save model summary
try:
    with open('data/analysis/var_sparsevar_suggested_summary.txt', 'w') as f:
        f.write(str(results.summary()))
    print("  [OK] Model summary saved")
except Exception as e:
    print(f"  [WARNING] Could not generate summary: {e}")

# =============================================================================
# DIAGNOSTICS
# =============================================================================
print("\n[4/7] Running diagnostics...")

# Test for autocorrelation in residuals
print("\n  Ljung-Box Test for Residual Autocorrelation:")
print("  " + "-" * 96)

autocorr_tests = []
for i, var in enumerate(endog_vars):
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
print(f"\n  Overall: {n_pass}/{len(endog_vars)} equations pass autocorrelation test")

# =============================================================================
# GRANGER CAUSALITY
# =============================================================================
print("\n[5/7] Computing Granger causality...")

granger_results = []

for cause_var in endog_vars:
    for effect_var in endog_vars:
        if cause_var == effect_var:
            continue

        try:
            # Granger test
            test_data = endog_data[[effect_var, cause_var]].dropna()
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
granger_df.to_excel('data/analysis/var_sparsevar_suggested_granger.xlsx', index=False)

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

# Check for key bureaucratic relationships
print("\n  Checking key bureaucratic relationships:")
print("  " + "-" * 96)
key_pairs = [
    ('Field_Grade_Officers_Z', 'FOIA_Complex_Days_Z'),
    ('Warrant_Officers_Z', 'Total_PAS_Z'),
    ('Warrant_Officers_Z', 'FOIA_Complex_Days_Z'),
    ('Total_Civilians_Z', 'Total_PAS_Z'),
    ('President_Republican', 'Junior_Enlisted_Z')
]

for cause, effect in key_pairs:
    match = granger_df[(granger_df['Cause'] == cause) & (granger_df['Effect'] == effect)]
    if len(match) > 0:
        row = match.iloc[0]
        sig = "**" if row['p_value'] < 0.05 else "  "
        print(f"  {sig} {cause:30s} -> {effect:30s} F={row['F_stat']:6.2f} p={row['p_value']:.4f}")

# =============================================================================
# IMPULSE RESPONSE FUNCTIONS
# =============================================================================
print("\n[6/7] Computing impulse response functions...")

try:
    irf = results.irf(10)
    irf_success = True
except np.linalg.LinAlgError as e:
    print(f"  [WARNING] IRF calculation failed: {e}")
    irf_success = False

if irf_success:
    # Plot IRFs for key relationships
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
    axes = axes.flatten()

    key_shocks = [
        ('Warrant_Officers_Z', 'Total_PAS_Z'),
        ('Warrant_Officers_Z', 'FOIA_Complex_Days_Z'),
        ('Field_Grade_Officers_Z', 'FOIA_Complex_Days_Z'),
        ('Total_Civilians_Z', 'Total_PAS_Z'),
        ('President_Republican', 'Junior_Enlisted_Z'),
        ('Total_PAS_Z', 'Junior_Enlisted_Z')
    ]

    for idx, (shock_var, response_var) in enumerate(key_shocks):
        ax = axes[idx]

        shock_idx = endog_vars.index(shock_var)
        response_idx = endog_vars.index(response_var)

        # Get IRF data and plot manually
        irf_data = irf.irfs[:, response_idx, shock_idx]
        periods = range(len(irf_data))
        ax.plot(periods, irf_data, linewidth=2, color='#3498db')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.fill_between(periods, 0, irf_data, alpha=0.3, color='#3498db')

        ax.set_title(f'Shock: {shock_var.replace("_Z", "")}\\nResponse: {response_var.replace("_Z", "")}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Period', fontsize=9)
        ax.set_ylabel('Response', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/analysis/var_sparsevar_suggested_irf.png', dpi=300, bbox_inches='tight')
    print("  [OK] IRF plot saved")
else:
    print("  [SKIPPED] IRF plot not generated due to numerical issues")

# =============================================================================
# FORECAST ERROR VARIANCE DECOMPOSITION
# =============================================================================
print("\n[7/7] Computing forecast error variance decomposition...")

if irf_success:
    fevd = results.fevd(10)

    # FEVD for key variables
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')

    key_vars = ['Warrant_Officers_Z', 'Total_PAS_Z', 'FOIA_Complex_Days_Z']

    for idx, var in enumerate(key_vars):
        ax = axes[idx]
        var_idx = endog_vars.index(var)

        fevd_data = fevd.decomp[:, var_idx, :]

        # Plot stacked area
        periods = range(fevd_data.shape[0])
        ax.stackplot(periods, *fevd_data.T, labels=[v.replace('_Z', '') for v in endog_vars], alpha=0.7)
        ax.set_title(f'FEVD: {var.replace("_Z", "")}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Period', fontsize=10)
        ax.set_ylabel('Proportion of Variance', fontsize=10)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/analysis/var_sparsevar_suggested_fevd.png', dpi=300, bbox_inches='tight')
    print("  [OK] FEVD plot saved")
else:
    print("  [SKIPPED] FEVD plot not generated due to numerical issues")

# =============================================================================
# COMPARISON WITH ORIGINAL VAR(2)
# =============================================================================
print("\n" + "=" * 100)
print("COMPARISON WITH ORIGINAL VAR(2) MODEL")
print("=" * 100)

print("\nORIGINAL MODEL (Field_Grade focus):")
print("  Variables: Junior_Enlisted, FOIA_Simple_Days, Total_PAS, Middle_Enlisted,")
print("             Policy_Count, Field_Grade_Officers, GOFOs")
print("  Key finding: Field_Grade_Officers <-> Policy_Count relationship")

print("\nNEW MODEL (SparseVAR suggested):")
print("  Variables: Warrant_Officers, Total_PAS, Field_Grade_Officers, FOIA_Complex_Days,")
print("             Junior_Enlisted, President_Republican, Total_Civilians")
print("  Key finding: Warrant_Officers as bridge between political and bureaucratic layers")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)

print(f"\nMODEL SPECIFICATION:")
print(f"  Endogenous variables: {len(endog_vars)}")
print(f"  Exogenous variables: {len(exog_vars)}")
print(f"  Lag order: {lag_order}")
print(f"  Observations: {results.nobs}")
try:
    print(f"  AIC: {results.aic:.2f}")
    print(f"  BIC: {results.bic:.2f}")
except:
    print(f"  AIC/BIC: Could not compute")

print(f"\nDIAGNOSTICS:")
print(f"  Autocorrelation tests passed: {n_pass}/{len(endog_vars)}")

print(f"\nGRANGER CAUSALITY:")
print(f"  Significant relationships (p<0.05): {n_significant}/{len(granger_df)}")

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print("  1. var_sparsevar_suggested_summary.txt - Full model summary")
print("  2. var_sparsevar_suggested_granger.xlsx - Granger causality results")
print("  3. var_sparsevar_suggested_irf.png - Impulse response functions")
print("  4. var_sparsevar_suggested_fevd.png - Variance decomposition")
print("=" * 100)

print("\nKEY QUESTION:")
print("  Does Warrant_Officers centrality hold up in proper VAR framework?")
print("  Compare diagnostics and Granger results with original VAR(2) model.")
print("=" * 100)
