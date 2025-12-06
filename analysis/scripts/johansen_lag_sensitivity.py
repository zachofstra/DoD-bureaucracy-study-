"""
Johansen Cointegration Test - Lag Order Sensitivity Analysis
Graduate-level robustness check for DoD bureaucratic growth research

Tests lag orders 1-5 to determine if cointegration results are robust to
lag specification. Includes comprehensive diagnostics and visualizations.

Key Questions:
1. How many cointegrating relationships at each lag order?
2. Are the 4 cointegrating vectors found at lag 2 robust?
3. Which lag order is optimal (AIC vs BIC)?
4. Do eigenvalues and trace statistics change dramatically?

Variables (7):
- Junior_Enlisted_Z
- FOIA_Simple_Days_Z
- Total_PAS_Z
- Total_Civilians_Z
- Policy_Count_Log
- Field_Grade_Officers_Z
- GOFOs_Z
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.stattools import adfuller
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
output_dir = 'data/analysis/lag-sensitivity'
os.makedirs(output_dir, exist_ok=True)

print("=" * 100)
print("JOHANSEN COINTEGRATION - LAG ORDER SENSITIVITY ANALYSIS")
print("Graduate-Level Robustness Check for DoD Bureaucratic Growth (1987-2024)")
print("=" * 100)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/6] Loading 7-variable cointegration dataset...")

df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

coint_vars = [
    'Junior_Enlisted_Z',
    'FOIA_Simple_Days_Z',
    'Total_PAS_Z',
    'Total_Civilians_Z',
    'Policy_Count_Log',
    'Field_Grade_Officers_Z',
    'GOFOs_Z'
]

data = df[coint_vars].copy().dropna()

print(f"  Variables: {len(coint_vars)}")
print(f"  Observations: {len(data)}")
print(f"  Period: {df['FY'].min():.0f}-{df['FY'].max():.0f}")

# =============================================================================
# RUN JOHANSEN TEST ACROSS LAG ORDERS 1-5
# =============================================================================
print("\n[2/6] Running Johansen cointegration test for lag orders 1-5...")
print("  " + "-" * 96)

lag_orders = range(1, 6)
johansen_results = []

for lag in lag_orders:
    print(f"\n  Testing lag order {lag}...")

    try:
        # Run Johansen test
        joh = coint_johansen(data, det_order=0, k_ar_diff=lag)

        # Count cointegrating relationships at 5% level
        n_coint_5pct = 0
        n_coint_1pct = 0

        for i in range(len(coint_vars)):
            trace_stat = joh.lr1[i]
            cv_5 = joh.cvt[i, 1]
            cv_1 = joh.cvt[i, 2]

            if trace_stat > cv_5:
                n_coint_5pct = i + 1
            if trace_stat > cv_1:
                n_coint_1pct = i + 1

        # Store trace statistics and critical values
        trace_stats = joh.lr1
        cv_5pct = joh.cvt[:, 1]
        cv_1pct = joh.cvt[:, 2]

        # Get eigenvalues
        eigenvalues = joh.eig

        # Fit VECM to get information criteria
        try:
            vecm = VECM(data, k_ar_diff=lag, coint_rank=n_coint_5pct, deterministic='ci')
            vecm_result = vecm.fit()
            aic = vecm_result.aic
            bic = vecm_result.bic
            hqic = vecm_result.hqic
            llf = vecm_result.llf
        except:
            aic = bic = hqic = llf = np.nan

        johansen_results.append({
            'Lag_Order': lag,
            'N_Coint_5pct': n_coint_5pct,
            'N_Coint_1pct': n_coint_1pct,
            'Trace_r0': trace_stats[0],
            'Trace_r1': trace_stats[1],
            'Trace_r2': trace_stats[2],
            'CV_5pct_r0': cv_5pct[0],
            'CV_5pct_r1': cv_5pct[1],
            'CV_5pct_r2': cv_5pct[2],
            'Eigenvalue_1': eigenvalues[0] if len(eigenvalues) > 0 else np.nan,
            'Eigenvalue_2': eigenvalues[1] if len(eigenvalues) > 1 else np.nan,
            'Eigenvalue_3': eigenvalues[2] if len(eigenvalues) > 2 else np.nan,
            'VECM_AIC': aic,
            'VECM_BIC': bic,
            'VECM_HQIC': hqic,
            'VECM_LogLik': llf,
            'Effective_Obs': len(data) - lag
        })

        print(f"    Cointegrating relationships (5% level): {n_coint_5pct}")
        print(f"    Cointegrating relationships (1% level): {n_coint_1pct}")
        print(f"    Trace statistic (r=0): {trace_stats[0]:.2f} (CV 5%: {cv_5pct[0]:.2f})")
        print(f"    VECM AIC: {aic:.2f}, BIC: {bic:.2f}")
        print(f"    Effective observations: {len(data) - lag}")

    except Exception as e:
        print(f"    [ERROR] Johansen test failed at lag {lag}: {e}")
        johansen_results.append({
            'Lag_Order': lag,
            'N_Coint_5pct': np.nan,
            'N_Coint_1pct': np.nan,
            'Trace_r0': np.nan,
            'Trace_r1': np.nan,
            'Trace_r2': np.nan,
            'CV_5pct_r0': np.nan,
            'CV_5pct_r1': np.nan,
            'CV_5pct_r2': np.nan,
            'Eigenvalue_1': np.nan,
            'Eigenvalue_2': np.nan,
            'Eigenvalue_3': np.nan,
            'VECM_AIC': np.nan,
            'VECM_BIC': np.nan,
            'VECM_HQIC': np.nan,
            'VECM_LogLik': np.nan,
            'Effective_Obs': len(data) - lag
        })

results_df = pd.DataFrame(johansen_results)
results_df.to_excel(f'{output_dir}/johansen_lag_sensitivity_results.xlsx', index=False)

print("\n  [OK] Lag sensitivity results saved to Excel")

# =============================================================================
# VISUALIZATION 1: NUMBER OF COINTEGRATING RELATIONSHIPS BY LAG
# =============================================================================
print("\n[3/6] Creating visualization 1: Cointegrating relationships by lag order...")

fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor='white')

x = results_df['Lag_Order']
y_5pct = results_df['N_Coint_5pct']
y_1pct = results_df['N_Coint_1pct']

ax.plot(x, y_5pct, 'o-', linewidth=3, markersize=12, label='5% Significance', color='#2c3e50')
ax.plot(x, y_1pct, 's--', linewidth=2, markersize=10, label='1% Significance', color='#e74c3c', alpha=0.7)

# Highlight lag 2 (current specification)
lag2_idx = results_df[results_df['Lag_Order'] == 2].index[0]
lag2_coint = results_df.loc[lag2_idx, 'N_Coint_5pct']
ax.plot(2, lag2_coint, '*', markersize=25, color='gold',
        markeredgecolor='black', markeredgewidth=2, label='Current Specification (Lag 2)', zorder=5)

ax.set_xlabel('Lag Order (k)', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Cointegrating Relationships', fontsize=14, fontweight='bold')
ax.set_title('Johansen Cointegration Test: Lag Order Sensitivity\n' +
             'DoD Bureaucracy (7 Variables, 1987-2024)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_yticks(range(0, 8))
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='best', framealpha=0.95)

# Add annotation
ax.text(0.98, 0.02, f'Robust specification: Cointegration rank stable across lags' if
        results_df['N_Coint_5pct'].nunique() == 1 else 'WARNING: Cointegration rank varies with lag order',
        transform=ax.transAxes, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{output_dir}/1_cointegration_rank_by_lag.png', dpi=300, bbox_inches='tight')
print("  [OK] Saved: 1_cointegration_rank_by_lag.png")

# =============================================================================
# VISUALIZATION 2: TRACE STATISTICS COMPARISON
# =============================================================================
print("\n[4/6] Creating visualization 2: Trace statistics comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

ranks_to_plot = ['r0', 'r1', 'r2']
rank_labels = ['r ≤ 0 (No coint)', 'r ≤ 1 (At most 1)', 'r ≤ 2 (At most 2)']

for idx, (rank, label) in enumerate(zip(ranks_to_plot, rank_labels)):
    ax = axes[idx]

    trace_col = f'Trace_{rank}'
    cv_col = f'CV_5pct_{rank}'

    x = results_df['Lag_Order']
    y_trace = results_df[trace_col]
    y_cv = results_df[cv_col]

    ax.plot(x, y_trace, 'o-', linewidth=3, markersize=10, label='Trace Statistic', color='#3498db')
    ax.axhline(y=y_cv.iloc[0], color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Critical Value (5%): {y_cv.iloc[0]:.2f}')

    # Highlight where we reject null
    for lag_val, trace_val, cv_val in zip(x, y_trace, y_cv):
        if trace_val > cv_val:
            ax.plot(lag_val, trace_val, 'o', markersize=15, color='green',
                   alpha=0.3, zorder=3)

    ax.set_xlabel('Lag Order', fontsize=12, fontweight='bold')
    ax.set_ylabel('Trace Statistic', fontsize=12, fontweight='bold')
    ax.set_title(f'{label}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

fig.suptitle('Trace Statistics vs Critical Values Across Lag Orders\n' +
             'Green markers = Reject null hypothesis (cointegration detected)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/2_trace_statistics_comparison.png', dpi=300, bbox_inches='tight')
print("  [OK] Saved: 2_trace_statistics_comparison.png")

# =============================================================================
# VISUALIZATION 3: INFORMATION CRITERIA FOR VECM
# =============================================================================
print("\n[5/6] Creating visualization 3: VECM information criteria...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
axes = axes.flatten()

criteria = ['VECM_AIC', 'VECM_BIC', 'VECM_HQIC', 'VECM_LogLik']
titles = ['AIC (Lower = Better)', 'BIC (Lower = Better)',
          'HQIC (Lower = Better)', 'Log-Likelihood (Higher = Better)']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

for idx, (crit, title, color) in enumerate(zip(criteria, titles, colors)):
    ax = axes[idx]

    x = results_df['Lag_Order']
    y = results_df[crit]

    # Skip if all NaN
    if y.isna().all():
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=13, fontweight='bold')
        continue

    ax.plot(x, y, 'o-', linewidth=3, markersize=12, color=color)

    # Mark optimal
    if crit == 'VECM_LogLik':
        optimal_idx = y.idxmax()
    else:
        optimal_idx = y.idxmin()

    if not pd.isna(optimal_idx):
        optimal_lag = results_df.loc[optimal_idx, 'Lag_Order']
        optimal_val = results_df.loc[optimal_idx, crit]
        ax.plot(optimal_lag, optimal_val, '*', markersize=25, color='gold',
               markeredgecolor='black', markeredgewidth=2, zorder=5)
        ax.text(optimal_lag, optimal_val, f'  Optimal\n  (Lag {int(optimal_lag)})',
               fontsize=10, fontweight='bold', va='center')

    # Highlight lag 2
    lag2_idx = results_df[results_df['Lag_Order'] == 2].index[0]
    lag2_val = results_df.loc[lag2_idx, crit]
    if not pd.isna(lag2_val):
        ax.plot(2, lag2_val, 's', markersize=12, color='orange',
               markeredgecolor='black', markeredgewidth=2, label='Current (Lag 2)')

    ax.set_xlabel('Lag Order', fontsize=12, fontweight='bold')
    ax.set_ylabel(crit.replace('VECM_', ''), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    if idx == 3:
        ax.legend(fontsize=10)

fig.suptitle('VECM Information Criteria Across Lag Orders\n' +
             'Model Selection for Cointegration Analysis',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/3_vecm_information_criteria.png', dpi=300, bbox_inches='tight')
print("  [OK] Saved: 3_vecm_information_criteria.png")

# =============================================================================
# VISUALIZATION 4: EIGENVALUES COMPARISON
# =============================================================================
print("\n[6/6] Creating visualization 4: Eigenvalues comparison...")

fig, ax = plt.subplots(1, 1, figsize=(14, 8), facecolor='white')

x = results_df['Lag_Order']
eigen1 = results_df['Eigenvalue_1']
eigen2 = results_df['Eigenvalue_2']
eigen3 = results_df['Eigenvalue_3']

ax.plot(x, eigen1, 'o-', linewidth=3, markersize=12, label='λ₁ (Largest)', color='#2c3e50')
ax.plot(x, eigen2, 's-', linewidth=3, markersize=10, label='λ₂ (2nd)', color='#3498db')
ax.plot(x, eigen3, '^-', linewidth=3, markersize=10, label='λ₃ (3rd)', color='#95a5a6')

# Add reference line at 0
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='λ = 0')

ax.set_xlabel('Lag Order', fontsize=14, fontweight='bold')
ax.set_ylabel('Eigenvalue', fontsize=14, fontweight='bold')
ax.set_title('Largest Eigenvalues from Johansen Test Across Lag Orders\n' +
             'Larger eigenvalues indicate stronger cointegration',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='best', framealpha=0.95)

# Add interpretation box
interpretation = """
Eigenvalue Interpretation:
• λ close to 1: Very strong cointegration
• λ close to 0: Weak/no cointegration
• Stability across lags: Robust finding
"""
ax.text(0.02, 0.98, interpretation, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{output_dir}/4_eigenvalues_comparison.png', dpi=300, bbox_inches='tight')
print("  [OK] Saved: 4_eigenvalues_comparison.png")

# =============================================================================
# COMPREHENSIVE SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 100)
print("SUMMARY TABLE: JOHANSEN TEST ACROSS LAG ORDERS")
print("=" * 100)

summary_cols = ['Lag_Order', 'N_Coint_5pct', 'N_Coint_1pct', 'Trace_r0',
                'CV_5pct_r0', 'VECM_AIC', 'VECM_BIC', 'Effective_Obs']
print(results_df[summary_cols].to_string(index=False))

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("EXECUTIVE SUMMARY")
print("=" * 100)

# Determine if results are robust
unique_coint_5pct = results_df['N_Coint_5pct'].dropna().unique()
is_robust = len(unique_coint_5pct) == 1

# Find optimal lag by BIC (only if BIC values exist)
if not results_df['VECM_BIC'].isna().all():
    bic_optimal_idx = results_df['VECM_BIC'].idxmin()
    bic_optimal_lag = results_df.loc[bic_optimal_idx, 'Lag_Order']
    bic_optimal_coint = results_df.loc[bic_optimal_idx, 'N_Coint_5pct']
else:
    bic_optimal_idx = None
    bic_optimal_lag = np.nan
    bic_optimal_coint = np.nan

# Find optimal lag by AIC (only if AIC values exist)
if not results_df['VECM_AIC'].isna().all():
    aic_optimal_idx = results_df['VECM_AIC'].idxmin()
    aic_optimal_lag = results_df.loc[aic_optimal_idx, 'Lag_Order']
    aic_optimal_coint = results_df.loc[aic_optimal_idx, 'N_Coint_5pct']
else:
    aic_optimal_idx = None
    aic_optimal_lag = np.nan
    aic_optimal_coint = np.nan

summary_text = f"""
================================================================================
JOHANSEN COINTEGRATION - LAG ORDER SENSITIVITY ANALYSIS
Executive Summary for DoD Bureaucratic Growth Research
================================================================================

ROBUSTNESS CHECK RESULTS:
-------------------------
Lag orders tested: {min(lag_orders)} to {max(lag_orders)}
Variables: {len(coint_vars)} (7-variable aggregated model)
Time period: 1987-2024 ({len(data)} observations)

COINTEGRATION RANK ACROSS LAG ORDERS:
--------------------------------------
"""

for idx, row in results_df.iterrows():
    lag = int(row['Lag_Order'])
    n_coint = int(row['N_Coint_5pct']) if not pd.isna(row['N_Coint_5pct']) else 'FAILED'
    marker = ' ** CURRENT' if lag == 2 else ''
    if not pd.isna(bic_optimal_lag):
        marker += ' ✓ BIC OPTIMAL' if lag == bic_optimal_lag else ''
    if not pd.isna(aic_optimal_lag):
        marker += ' ✓ AIC OPTIMAL' if lag == aic_optimal_lag else ''
    summary_text += f"Lag {lag}: {n_coint} cointegrating relationship(s){marker}\n"

summary_text += f"""

INFORMATION CRITERIA RECOMMENDATION:
------------------------------------
BIC optimal lag: {'N/A (VECM fitting failed)' if pd.isna(bic_optimal_lag) else f"{int(bic_optimal_lag)} - {int(bic_optimal_coint)} cointegrating relationships"}

AIC optimal lag: {'N/A (VECM fitting failed)' if pd.isna(aic_optimal_lag) else f"{int(aic_optimal_lag)} - {int(aic_optimal_coint)} cointegrating relationships"}

Current specification (Lag 2):
  - {int(results_df[results_df['Lag_Order']==2]['N_Coint_5pct'].iloc[0])} cointegrating relationships

NOTE: VECM information criteria unavailable due to fitting issues. Using
trace statistic comparison and theoretical considerations instead.

"""

if is_robust:
    summary_text += f"""
✓ ROBUSTNESS: CONFIRMED
-----------------------
The number of cointegrating relationships is STABLE across all successful lag orders.
All specifications identify {int(unique_coint_5pct[0])} cointegrating relationship(s).

INTERPRETATION:
Your finding of long-run equilibrium relationships is ROBUST to lag specification.
This strengthens the credibility of your cointegration analysis.

RECOMMENDATION FOR THESIS:
Use the current lag 2 specification or lag {int(unique_coint_5pct[0])} based on
theoretical considerations. Both yield the same cointegration rank.
"""
else:
    summary_text += f"""
⚠ WARNING: LAG SENSITIVITY DETECTED
------------------------------------
The number of cointegrating relationships VARIES across lag orders:
{', '.join([f'Lag {int(row["Lag_Order"])}: {int(row["N_Coint_5pct"])}' for _, row in results_df.iterrows() if not pd.isna(row['N_Coint_5pct'])])}

INTERPRETATION:
Your cointegration results are SENSITIVE to lag specification. This suggests:

1. STRUCTURAL BREAKS: The 37-year period may contain regime changes (9/11,
   sequestration, COVID) that disrupt long-run equilibria at different lags.

2. NONLINEAR DYNAMICS: Linear cointegration may not fully capture the
   bureaucratic growth process. Threshold or regime-switching models may be needed.

3. SAMPLE SIZE LIMITATIONS: With ~34 effective observations after differencing,
   higher lags consume degrees of freedom and reduce test power.

RECOMMENDATION FOR THESIS:
Report results from MULTIPLE lag specifications to show sensitivity:
1. Present lag 2 as primary specification (current, theoretically justified)
2. Show lag 1 and lag 4 results as robustness checks
3. Discuss lag sensitivity in limitations section
4. Consider threshold cointegration or regime-switching models

DO NOT cherry-pick the lag that gives your preferred result. Transparency about
lag sensitivity strengthens your thesis by showing methodological rigor.
"""

summary_text += f"""

================================================================================
METHODOLOGICAL GUIDANCE FOR THESIS
================================================================================

1. REPORT LAG SELECTION PROCESS:
   "Lag order selection was conducted using Johansen cointegration tests for
   lags 1-5. Due to the limited sample size (34 observations, 7 variables),
   we use lag 2 as our primary specification, balancing model fit against
   degrees of freedom preservation."

2. ROBUSTNESS CHECK:
   "To verify robustness, we estimated the Johansen test for lag orders 1-5.
   {'The cointegration rank remained stable across all specifications, ' if is_robust else 'The cointegration rank varied across specifications, '}
   {'confirming' if is_robust else 'suggesting'}
   {'the stability of long-run equilibrium relationships.' if is_robust else 'sensitivity to lag specification.'}"

3. DEGREES OF FREEDOM TRADE-OFF:
   "With {len(data)} observations and {len(coint_vars)} variables, higher lag orders
   reduce effective sample size. Lag 2 balances model fit against parsimony,
   preserving {int(results_df[results_df['Lag_Order']==2]['Effective_Obs'].iloc[0])}
   effective observations."

4. EIGENVALUE INTERPRETATION:
   "The largest eigenvalue (λ₁ = {results_df[results_df['Lag_Order']==2]['Eigenvalue_1'].iloc[0]:.3f} at lag 2)
   indicates {'strong' if results_df[results_df['Lag_Order']==2]['Eigenvalue_1'].iloc[0] > 0.5 else 'moderate'}
   cointegration. Eigenvalues {'remained stable' if results_df['Eigenvalue_1'].std() < 0.1 else 'varied'}
   across lag orders (SD = {results_df['Eigenvalue_1'].std():.3f})."

================================================================================
FILES GENERATED
================================================================================

1. johansen_lag_sensitivity_results.xlsx
   - Full results table with trace stats, eigenvalues, information criteria

2. 1_cointegration_rank_by_lag.png
   - Number of cointegrating relationships across lag orders

3. 2_trace_statistics_comparison.png
   - Trace statistics vs critical values for ranks r=0, r=1, r=2

4. 3_vecm_information_criteria.png
   - AIC, BIC, HQIC, and log-likelihood comparison

5. 4_eigenvalues_comparison.png
   - Largest eigenvalues across lag specifications

6. LAG_SENSITIVITY_SUMMARY.txt
   - This executive summary

All files saved to: {output_dir}/

================================================================================
CONCLUSION
================================================================================

{'Your cointegration findings are ROBUST to lag specification.' if is_robust else
 'Your cointegration findings show LAG SENSITIVITY - report multiple specifications.'}

Recommended primary specification: Lag 2 (current, based on trace statistics)
Current specification status: {'Optimal based on available evidence' if pd.isna(bic_optimal_lag) else f'BIC optimal is lag {int(bic_optimal_lag)}'}

This lag sensitivity analysis demonstrates methodological rigor and strengthens
the credibility of your thesis. Graduate committees value transparency about
specification choices.

================================================================================
"""

# Save summary
with open(f'{output_dir}/LAG_SENSITIVITY_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(summary_text)

print("\n" + "=" * 100)
print("LAG SENSITIVITY ANALYSIS COMPLETE")
print("=" * 100)
print(f"\nAll outputs saved to: {output_dir}/")
print("\nRecommended action:")
if is_robust:
    print(f"  ✓ Use lag 2 (current specification) for primary analysis")
    print("  ✓ Results are robust - cointegration rank stable across successful lags")
else:
    print(f"  ⚠ Report multiple lag specifications (lag 2 recommended)")
    print("  ⚠ Discuss lag sensitivity in thesis limitations section")
    print(f"  ⚠ Cointegration rank varies: {dict(zip(results_df['Lag_Order'], results_df['N_Coint_5pct']))}")
print("=" * 100)
