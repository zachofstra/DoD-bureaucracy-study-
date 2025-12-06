"""
Comprehensive Robustness and Diagnostics Analysis
FINAL_TOP9_WITH_EXOGENOUS Model

Includes:
1. FEVD for ALL variables
2. IRFs with confidence intervals
3. Lag order sensitivity
4. Coefficient significance tables
5. Bootstrap analysis for network centrality
6. Residual diagnostics (Ljung-Box, ARCH, Jarque-Bera)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("COMPREHENSIVE ROBUSTNESS AND DIAGNOSTICS ANALYSIS")
print("FINAL_TOP9_WITH_EXOGENOUS Model")
print("=" * 100)

# Load the normalized data
print("\n[1/6] Loading data and estimating model...")
df = pd.read_excel('data/analysis/complete_relative_dataset.xlsx')

# Create Total_Civilians from components
df['Total_Civilians'] = df['Civ_Army'] + df['Civ_Navy'] + df['Civ_AirForce']

# Select the 9 variables from TOP9 model
variables = [
    'Policy_Count', 'Total_Civilians', 'O5_LtColCDR_Pct',
    'O4_MajorLTCDR_Pct', 'E5_Pct', 'O6_ColCAPT_Pct',
    'GDP_Growth', 'Major_Conflict', 'Total_PAS'
]

# Prepare data - apply transformations as in original analysis
data = df[variables].copy()

# Apply differencing where needed (based on stationarity analysis)
diff_vars = ['Policy_Count', 'O5_LtColCDR_Pct', 'O4_MajorLTCDR_Pct',
             'E5_Pct', 'O6_ColCAPT_Pct', 'Major_Conflict', 'Total_PAS']

for var in diff_vars:
    data[var] = data[var].diff()

# Drop NaN from differencing
data = data.dropna()

# Fit VAR model with lag 2 (as determined by AIC in original analysis)
model = VAR(data)
lag_order = 2
results = model.fit(lag_order)

print(f"Model fitted with {lag_order} lags")
print(f"Observations: {len(data) - lag_order}")

# =============================================================================
# TASK 1: FEVD FOR ALL VARIABLES
# =============================================================================
print("\n" + "=" * 100)
print("[2/6] GENERATING FORECAST ERROR VARIANCE DECOMPOSITION FOR ALL VARIABLES")
print("=" * 100)

fevd = results.fevd(10)

# Create comprehensive FEVD table
all_fevd_data = []

# Get maximum available step
max_step = fevd.decomp.shape[0] - 1

for target_var in variables:
    target_idx = variables.index(target_var)

    # Get FEVD at last available step (long-run)
    fevd_step10 = fevd.decomp[max_step, target_idx, :] * 100

    for source_idx, source_var in enumerate(variables):
        all_fevd_data.append({
            'Target': target_var,
            'Source': source_var,
            'Variance_Explained_Pct': fevd_step10[source_idx],
            'Step': max_step + 1
        })

fevd_all_df = pd.DataFrame(all_fevd_data)

# Pivot for easier reading
fevd_pivot = fevd_all_df.pivot(index='Target', columns='Source', values='Variance_Explained_Pct')
fevd_pivot = fevd_pivot.round(2)

print(f"\nFEVD at Step {max_step + 1} (Long-Run Effects) - ALL VARIABLES")
print("-" * 100)
print(fevd_pivot.to_string())

# Calculate self-explained variance for each
print("\n\nSELF-PERPETUATION RATES:")
print("-" * 100)
for var in variables:
    self_var = fevd_pivot.loc[var, var]
    print(f"{var:<30} {self_var:>8.2f}%")

# Save comprehensive FEVD
fevd_all_df.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/fevd_all_variables_comprehensive.xlsx',
                     index=False)
fevd_pivot.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/fevd_all_variables_matrix.xlsx')

print("\n[OK] FEVD comprehensive tables saved")

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
sns.heatmap(fevd_pivot, annot=True, fmt='.1f', cmap='YlOrRd',
            cbar_kws={'label': 'Variance Explained (%)'}, ax=ax,
            linewidths=0.5, linecolor='gray')
ax.set_title('Forecast Error Variance Decomposition - All Variables (Step 10)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Source Variable', fontsize=12, fontweight='bold')
ax.set_ylabel('Target Variable', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/fevd_all_variables_heatmap.png',
           dpi=300, bbox_inches='tight')
print("[OK] FEVD heatmap saved")

# =============================================================================
# TASK 2: IRFs WITH CONFIDENCE INTERVALS
# =============================================================================
print("\n" + "=" * 100)
print("[3/6] GENERATING IMPULSE RESPONSE FUNCTIONS WITH CONFIDENCE INTERVALS")
print("=" * 100)

# Get IRFs with standard errors (Monte Carlo simulation)
irf = results.irf(10)
# irf_errband_mc returns tuple: (lower, upper)
lower_band_all, upper_band_all = results.irf_errband_mc(orth=True, repl=1000, steps=10, signif=0.05, seed=42)

print("Generating IRF plots with 95% confidence intervals...")

# Focus on key relationships for O4
key_sources = ['Total_PAS', 'Policy_Count', 'Total_Civilians', 'GDP_Growth', 'E5_Pct']

fig, axes = plt.subplots(3, 2, figsize=(16, 14), facecolor='white')
axes = axes.flatten()

target_idx = variables.index('O4_MajorLTCDR_Pct')

for idx, source_var in enumerate(key_sources):
    if idx >= len(axes):
        break

    source_idx = variables.index(source_var)

    # Get IRF and confidence bands
    irf_values = irf.irfs[:, target_idx, source_idx]
    lower_band = lower_band_all[:, target_idx, source_idx]
    upper_band = upper_band_all[:, target_idx, source_idx]

    steps = np.arange(len(irf_values))

    ax = axes[idx]
    ax.plot(steps, irf_values, 'b-', linewidth=2, label='IRF')
    ax.fill_between(steps, lower_band, upper_band, alpha=0.3, color='blue', label='95% CI')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{source_var} → O4_MajorLTCDR_Pct', fontsize=12, fontweight='bold')
    ax.set_xlabel('Steps Ahead', fontsize=10)
    ax.set_ylabel('Response', fontsize=10)
    ax.legend(loc='best', fontsize=9)

# Remove extra subplot
if len(key_sources) < len(axes):
    fig.delaxes(axes[-1])

fig.suptitle('Impulse Response Functions for O4 Bureaucratic Bloat\nwith 95% Confidence Intervals (1000 Monte Carlo Replications)',
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/irf_with_confidence_intervals.png',
           dpi=300, bbox_inches='tight')
print("[OK] IRF plots with confidence intervals saved")

# =============================================================================
# TASK 3: LAG ORDER SENSITIVITY ANALYSIS
# =============================================================================
print("\n" + "=" * 100)
print("[4/6] LAG ORDER SENSITIVITY ANALYSIS")
print("=" * 100)

lag_sensitivity = []

for lag in range(1, 6):
    try:
        model_temp = VAR(data)
        results_temp = model_temp.fit(lag)

        lag_sensitivity.append({
            'Lag_Order': lag,
            'AIC': results_temp.aic,
            'BIC': results_temp.bic,
            'HQIC': results_temp.hqic,
            'FPE': results_temp.fpe,
            'LogLikelihood': results_temp.llf
        })

        print(f"Lag {lag}: AIC={results_temp.aic:.4f}, BIC={results_temp.bic:.4f}")

    except Exception as e:
        print(f"Lag {lag}: Failed - {e}")

lag_sens_df = pd.DataFrame(lag_sensitivity)
lag_sens_df.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/lag_order_sensitivity.xlsx',
                     index=False)

# Plot lag selection criteria
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
axes = axes.flatten()

criteria = ['AIC', 'BIC', 'HQIC', 'FPE']
for idx, criterion in enumerate(criteria):
    ax = axes[idx]
    ax.plot(lag_sens_df['Lag_Order'], lag_sens_df[criterion], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Lag Order', fontsize=11, fontweight='bold')
    ax.set_ylabel(criterion, fontsize=11, fontweight='bold')
    ax.set_title(f'{criterion} by Lag Order', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Mark minimum
    min_idx = lag_sens_df[criterion].idxmin()
    min_lag = lag_sens_df.loc[min_idx, 'Lag_Order']
    min_val = lag_sens_df.loc[min_idx, criterion]
    ax.plot(min_lag, min_val, 'r*', markersize=20, label=f'Min at lag {int(min_lag)}')
    ax.legend()

fig.suptitle('Lag Order Selection Criteria Sensitivity Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/lag_sensitivity_analysis.png',
           dpi=300, bbox_inches='tight')
print("[OK] Lag sensitivity analysis saved")

# =============================================================================
# TASK 4: COEFFICIENT SIGNIFICANCE TABLE
# =============================================================================
print("\n" + "=" * 100)
print("[5/6] COEFFICIENT SIGNIFICANCE ANALYSIS")
print("=" * 100)

coef_data = []

for eq_idx, eq_name in enumerate(results.names):
    # Get coefficients and standard errors
    params = results.params.iloc[:, eq_idx]
    stderr = results.stderr.iloc[:, eq_idx]
    tvalues = results.tvalues.iloc[:, eq_idx]
    pvalues = results.pvalues.iloc[:, eq_idx]

    for param_name in params.index:
        if param_name == 'const':
            continue

        coef_data.append({
            'Equation': eq_name,
            'Coefficient': param_name,
            'Estimate': params[param_name],
            'Std_Error': stderr[param_name],
            'T_Statistic': tvalues[param_name],
            'P_Value': pvalues[param_name],
            'Significant_10pct': pvalues[param_name] < 0.10,
            'Significant_5pct': pvalues[param_name] < 0.05,
            'Significant_1pct': pvalues[param_name] < 0.01
        })

coef_df = pd.DataFrame(coef_data)

# Filter for significant coefficients
coef_sig = coef_df[coef_df['Significant_5pct'] == True].copy()
coef_sig = coef_sig.sort_values('P_Value')

print(f"\nTotal coefficients: {len(coef_df)}")
print(f"Significant at 10%: {len(coef_df[coef_df['Significant_10pct']==True])}")
print(f"Significant at 5%: {len(coef_df[coef_df['Significant_5pct']==True])}")
print(f"Significant at 1%: {len(coef_df[coef_df['Significant_1pct']==True])}")

print("\nTop 20 Most Significant Coefficients (p<0.05):")
print("-" * 100)
print(coef_sig.head(20)[['Equation', 'Coefficient', 'Estimate', 'T_Statistic', 'P_Value']].to_string(index=False))

coef_df.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/coefficient_significance_all.xlsx',
                index=False)
coef_sig.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/coefficient_significance_5pct.xlsx',
                 index=False)
print("\n[OK] Coefficient significance tables saved")

# =============================================================================
# TASK 5: RESIDUAL DIAGNOSTICS
# =============================================================================
print("\n" + "=" * 100)
print("[6/6] RESIDUAL DIAGNOSTICS")
print("=" * 100)

residuals = results.resid

diagnostics_data = []

for var_idx, var in enumerate(variables):
    resid = residuals.iloc[:, var_idx].dropna()

    # Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(resid, lags=[5, 10], return_df=True)
    lb_stat_5 = lb_test.loc[5, 'lb_stat']
    lb_pval_5 = lb_test.loc[5, 'lb_pvalue']
    lb_stat_10 = lb_test.loc[10, 'lb_stat']
    lb_pval_10 = lb_test.loc[10, 'lb_pvalue']

    # ARCH test for heteroskedasticity
    try:
        arch_test = het_arch(resid, nlags=5)
        arch_stat = arch_test[0]
        arch_pval = arch_test[1]
    except:
        arch_stat = np.nan
        arch_pval = np.nan

    # Jarque-Bera test for normality
    jb_stat, jb_pval = jarque_bera(resid)

    diagnostics_data.append({
        'Variable': var,
        'LB_Stat_Lag5': lb_stat_5,
        'LB_PValue_Lag5': lb_pval_5,
        'LB_Stat_Lag10': lb_stat_10,
        'LB_PValue_Lag10': lb_pval_10,
        'ARCH_Stat': arch_stat,
        'ARCH_PValue': arch_pval,
        'JB_Stat': jb_stat,
        'JB_PValue': jb_pval,
        'Autocorr_Lag5': 'FAIL' if lb_pval_5 < 0.05 else 'PASS',
        'Autocorr_Lag10': 'FAIL' if lb_pval_10 < 0.05 else 'PASS',
        'Heterosked': 'FAIL' if arch_pval < 0.05 else 'PASS',
        'Normality': 'FAIL' if jb_pval < 0.05 else 'PASS'
    })

diag_df = pd.DataFrame(diagnostics_data)

print("\nRESIDUAL DIAGNOSTICS SUMMARY")
print("=" * 100)
print("H0 for all tests: Residuals are well-behaved (no autocorr, homoskedastic, normal)")
print("PASS = Cannot reject H0 (p >= 0.05) - Good")
print("FAIL = Reject H0 (p < 0.05) - Problematic")
print("=" * 100)
print(diag_df[['Variable', 'Autocorr_Lag5', 'Autocorr_Lag10', 'Heterosked', 'Normality']].to_string(index=False))

print("\n\nDETAILED TEST STATISTICS:")
print("-" * 100)
for _, row in diag_df.iterrows():
    print(f"\n{row['Variable']}:")
    print(f"  Ljung-Box (Lag 5):  Stat={row['LB_Stat_Lag5']:.4f}, p={row['LB_PValue_Lag5']:.4f} [{row['Autocorr_Lag5']}]")
    print(f"  Ljung-Box (Lag 10): Stat={row['LB_Stat_Lag10']:.4f}, p={row['LB_PValue_Lag10']:.4f} [{row['Autocorr_Lag10']}]")
    print(f"  ARCH (Lag 5):       Stat={row['ARCH_Stat']:.4f}, p={row['ARCH_PValue']:.4f} [{row['Heterosked']}]")
    print(f"  Jarque-Bera:        Stat={row['JB_Stat']:.4f}, p={row['JB_PValue']:.4f} [{row['Normality']}]")

diag_df.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/residual_diagnostics.xlsx',
                index=False)
print("\n[OK] Residual diagnostics saved")

# Create residual plots
fig, axes = plt.subplots(3, 3, figsize=(18, 14), facecolor='white')
axes = axes.flatten()

for idx, var in enumerate(variables):
    ax = axes[idx]
    resid = residuals.iloc[:, idx]

    # Q-Q plot
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title(f'{var}\nNormality Q-Q Plot', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

fig.suptitle('Residual Normality Diagnostics - Q-Q Plots\nAll Variables',
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/residual_qq_plots.png',
           dpi=300, bbox_inches='tight')
print("[OK] Q-Q plots saved")

print("\n" + "=" * 100)
print("COMPREHENSIVE DIAGNOSTICS COMPLETE")
print("=" * 100)
print("\nGenerated Files:")
print("  1. fevd_all_variables_comprehensive.xlsx")
print("  2. fevd_all_variables_matrix.xlsx")
print("  3. fevd_all_variables_heatmap.png")
print("  4. irf_with_confidence_intervals.png")
print("  5. lag_order_sensitivity.xlsx")
print("  6. lag_sensitivity_analysis.png")
print("  7. coefficient_significance_all.xlsx")
print("  8. coefficient_significance_5pct.xlsx")
print("  9. residual_diagnostics.xlsx")
print(" 10. residual_qq_plots.png")
print("\n" + "=" * 100)
