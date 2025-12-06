"""
Re-run VAR Analysis with Lag 5 (AIC/BIC Optimal)
Compare results to original Lag 2 model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("VAR MODEL RE-ESTIMATION WITH LAG 5")
print("Comparing to Original Lag 2 Results")
print("=" * 100)

# Load data
df = pd.read_excel('data/analysis/complete_relative_dataset.xlsx')
df['Total_Civilians'] = df['Civ_Army'] + df['Civ_Navy'] + df['Civ_AirForce']

variables = [
    'Policy_Count', 'Total_Civilians', 'O5_LtColCDR_Pct',
    'O4_MajorLTCDR_Pct', 'E5_Pct', 'O6_ColCAPT_Pct',
    'GDP_Growth', 'Major_Conflict', 'Total_PAS'
]

# Prepare data with differencing
data = df[variables].copy()
diff_vars = ['Policy_Count', 'O5_LtColCDR_Pct', 'O4_MajorLTCDR_Pct',
             'E5_Pct', 'O6_ColCAPT_Pct', 'Major_Conflict', 'Total_PAS']

for var in diff_vars:
    data[var] = data[var].diff()

data = data.dropna()

print(f"\nData prepared: {len(data)} observations")

# =============================================================================
# ESTIMATE LAG 2 MODEL (Original)
# =============================================================================
print("\n" + "=" * 100)
print("ESTIMATING LAG 2 MODEL (Original)")
print("=" * 100)

model_lag2 = VAR(data)
try:
    results_lag2 = model_lag2.fit(2)
    print(f"Lag 2 Model: SUCCESS")
    print(f"  Observations: {len(data) - 2}")
    print(f"  AIC: {results_lag2.aic:.4f}")
    print(f"  BIC: {results_lag2.bic:.4f}")
    lag2_success = True
except Exception as e:
    print(f"Lag 2 Model: FAILED - {e}")
    lag2_success = False
    results_lag2 = None

# =============================================================================
# ESTIMATE LAG 5 MODEL (AIC/BIC Optimal)
# =============================================================================
print("\n" + "=" * 100)
print("ESTIMATING LAG 5 MODEL (AIC/BIC Optimal)")
print("=" * 100)

model_lag5 = VAR(data)
try:
    results_lag5 = model_lag5.fit(5)
    print(f"Lag 5 Model: SUCCESS")
    print(f"  Observations: {len(data) - 5}")
    print(f"  AIC: {results_lag5.aic:.4f}")
    print(f"  BIC: {results_lag5.bic:.4f}")
    lag5_success = True
except Exception as e:
    print(f"Lag 5 Model: FAILED - {e}")
    lag5_success = False
    results_lag5 = None

if not lag5_success:
    print("\n[ERROR] Cannot proceed - Lag 5 estimation failed")
    exit(1)

# =============================================================================
# GRANGER CAUSALITY COMPARISON
# =============================================================================
print("\n" + "=" * 100)
print("GRANGER CAUSALITY ANALYSIS - LAG 5 MODEL")
print("=" * 100)

# Run Granger causality for lag 5
granger_lag5 = []

for cause_var in variables:
    for effect_var in variables:
        if cause_var == effect_var:
            continue

        try:
            # Test with maxlag=5
            test_result = grangercausalitytests(
                data[[effect_var, cause_var]].dropna(),
                maxlag=5,
                verbose=False
            )

            # Extract results for each lag
            for lag in [1, 2, 3, 4, 5]:
                if lag in test_result:
                    f_stat = test_result[lag][0]['ssr_ftest'][0]
                    p_value = test_result[lag][0]['ssr_ftest'][1]

                    granger_lag5.append({
                        'Cause': cause_var,
                        'Effect': effect_var,
                        'Lag': lag,
                        'F_statistic': f_stat,
                        'p_value': p_value,
                        'Significant_10pct': p_value < 0.10,
                        'Significant_5pct': p_value < 0.05,
                        'Significant_1pct': p_value < 0.01
                    })
        except:
            continue

granger_lag5_df = pd.DataFrame(granger_lag5)
granger_lag5_sig = granger_lag5_df[granger_lag5_df['Significant_5pct'] == True].copy()
granger_lag5_sig = granger_lag5_sig.sort_values('p_value')

print(f"\nTotal Granger tests: {len(granger_lag5_df)}")
print(f"Significant at 5%: {len(granger_lag5_sig)}")
print(f"Significant at 1%: {len(granger_lag5_df[granger_lag5_df['Significant_1pct']==True])}")

granger_lag5_df.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/LAG5_granger_all.xlsx', index=False)
granger_lag5_sig.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/LAG5_granger_significant.xlsx', index=False)

print("\nTop 15 Significant Relationships (Lag 5 Model):")
print("-" * 100)
print(granger_lag5_sig.head(15)[['Cause', 'Effect', 'Lag', 'F_statistic', 'p_value']].to_string(index=False))

# =============================================================================
# COMPARE TO LAG 2 RESULTS
# =============================================================================
print("\n" + "=" * 100)
print("COMPARISON: LAG 2 vs LAG 5")
print("=" * 100)

# Load lag 2 Granger results
try:
    granger_lag2_df = pd.read_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/granger_significant.xlsx')
    granger_lag2_sig = granger_lag2_df[granger_lag2_df['Significant_5pct'] == True].copy()

    print("\n1. NUMBER OF SIGNIFICANT RELATIONSHIPS (5% level):")
    print("-" * 100)
    print(f"Lag 2 Model: {len(granger_lag2_sig)} significant relationships")
    print(f"Lag 5 Model: {len(granger_lag5_sig)} significant relationships")
    print(f"Change: {len(granger_lag5_sig) - len(granger_lag2_sig):+d} relationships")

    # Create relationship identifiers
    lag2_relationships = set(
        granger_lag2_sig.apply(lambda x: f"{x['Cause']} -> {x['Effect']}", axis=1)
    )
    lag5_relationships = set(
        granger_lag5_sig.apply(lambda x: f"{x['Cause']} -> {x['Effect']}", axis=1)
    )

    # Find differences
    only_in_lag2 = lag2_relationships - lag5_relationships
    only_in_lag5 = lag5_relationships - lag2_relationships
    common = lag2_relationships & lag5_relationships

    print(f"\nCommon to both: {len(common)}")
    print(f"Only in Lag 2: {len(only_in_lag2)}")
    print(f"Only in Lag 5: {len(only_in_lag5)}")

    if len(only_in_lag2) > 0:
        print("\nRelationships LOST in Lag 5 (were significant in Lag 2):")
        for rel in sorted(only_in_lag2):
            print(f"  - {rel}")

    if len(only_in_lag5) > 0:
        print("\nRelationships GAINED in Lag 5 (newly significant):")
        for rel in sorted(only_in_lag5):
            print(f"  - {rel}")

except Exception as e:
    print(f"Could not load Lag 2 results: {e}")
    granger_lag2_df = None

# =============================================================================
# FEVD COMPARISON
# =============================================================================
print("\n" + "=" * 100)
print("FORECAST ERROR VARIANCE DECOMPOSITION - LAG 5")
print("=" * 100)

fevd_lag5 = results_lag5.fevd(10)
max_step = fevd_lag5.decomp.shape[0] - 1

# Create FEVD table for lag 5
fevd_lag5_data = []
for target_var in variables:
    target_idx = variables.index(target_var)
    fevd_values = fevd_lag5.decomp[max_step, target_idx, :] * 100

    for source_idx, source_var in enumerate(variables):
        fevd_lag5_data.append({
            'Target': target_var,
            'Source': source_var,
            'Variance_Explained_Pct': fevd_values[source_idx],
            'Step': max_step + 1
        })

fevd_lag5_df = pd.DataFrame(fevd_lag5_data)
fevd_lag5_pivot = fevd_lag5_df.pivot(index='Target', columns='Source', values='Variance_Explained_Pct')
fevd_lag5_pivot = fevd_lag5_pivot.round(2)

print(f"\nFEVD at Step {max_step + 1} (Lag 5 Model):")
print("-" * 100)
print(fevd_lag5_pivot.to_string())

fevd_lag5_df.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/LAG5_fevd_data.xlsx', index=False)
fevd_lag5_pivot.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/LAG5_fevd_matrix.xlsx')

# Compare O4 FEVD
print("\n" + "=" * 100)
print("O4 VARIANCE DECOMPOSITION COMPARISON")
print("=" * 100)

o4_lag5 = fevd_lag5_pivot.loc['O4_MajorLTCDR_Pct'].sort_values(ascending=False)

# Try to load lag 2 FEVD
try:
    fevd_lag2_df = pd.read_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/fevd_data.xlsx')
    # Get O4 columns
    o4_cols = [col for col in fevd_lag2_df.columns if col.startswith('O4_MajorLTCDR_Pct_from_')]
    max_step_lag2 = fevd_lag2_df['Step'].max()
    o4_lag2_data = fevd_lag2_df[fevd_lag2_df['Step'] == max_step_lag2]

    o4_lag2 = {}
    for col in o4_cols:
        var = col.replace('O4_MajorLTCDR_Pct_from_', '')
        val = o4_lag2_data[col].iloc[0]
        o4_lag2[var] = val if val > 1 else val * 100

    o4_lag2 = pd.Series(o4_lag2).sort_values(ascending=False)

    print(f"\n{'Source Variable':<25} {'Lag 2 (%)':<15} {'Lag 5 (%)':<15} {'Change':<15}")
    print("-" * 70)

    all_sources = set(list(o4_lag2.index) + list(o4_lag5.index))
    for source in sorted(all_sources, key=lambda x: o4_lag5.get(x, 0), reverse=True):
        val_lag2 = o4_lag2.get(source, 0)
        val_lag5 = o4_lag5.get(source, 0)
        change = val_lag5 - val_lag2

        change_str = f"{change:+.2f}%" if source in o4_lag2 else "NEW"
        print(f"{source:<25} {val_lag2:>13.2f}% {val_lag5:>13.2f}% {change_str:>15}")

    # Calculate exogenous totals
    exog_lag2 = o4_lag2.get('GDP_Growth', 0) + o4_lag2.get('Major_Conflict', 0)
    exog_lag5 = o4_lag5.get('GDP_Growth', 0) + o4_lag5.get('Major_Conflict', 0)

    print("-" * 70)
    print(f"{'EXOGENOUS TOTAL':<25} {exog_lag2:>13.2f}% {exog_lag5:>13.2f}% {exog_lag5-exog_lag2:+13.2f}%")
    print(f"{'Self-Perpetuation':<25} {o4_lag2.get('O4_MajorLTCDR_Pct', 0):>13.2f}% {o4_lag5.get('O4_MajorLTCDR_Pct', 0):>13.2f}% {o4_lag5.get('O4_MajorLTCDR_Pct', 0)-o4_lag2.get('O4_MajorLTCDR_Pct', 0):+13.2f}%")

except Exception as e:
    print(f"Could not load Lag 2 FEVD: {e}")

# =============================================================================
# CREATE COMPARISON VISUALIZATION
# =============================================================================
print("\n" + "=" * 100)
print("GENERATING COMPARISON VISUALIZATIONS")
print("=" * 100)

fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='white')

# Plot 1: Number of significant relationships by lag
if granger_lag2_df is not None:
    ax = axes[0, 0]
    categories = ['Total Tests', 'Sig at 10%', 'Sig at 5%', 'Sig at 1%']
    lag2_counts = [
        len(granger_lag2_df),
        len(granger_lag2_df[granger_lag2_df['Significant_10pct']==True]),
        len(granger_lag2_df[granger_lag2_df['Significant_5pct']==True]),
        len(granger_lag2_df[granger_lag2_df['Significant_1pct']==True])
    ]
    lag5_counts = [
        len(granger_lag5_df),
        len(granger_lag5_df[granger_lag5_df['Significant_10pct']==True]),
        len(granger_lag5_df[granger_lag5_df['Significant_5pct']==True]),
        len(granger_lag5_df[granger_lag5_df['Significant_1pct']==True])
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, lag2_counts, width, label='Lag 2', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, lag5_counts, width, label='Lag 5', alpha=0.8, color='coral')

    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Granger Causality: Lag 2 vs Lag 5', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

# Plot 2: O4 FEVD Comparison
if 'o4_lag2' in locals():
    ax = axes[0, 1]
    sources = sorted(set(list(o4_lag2.index) + list(o4_lag5.index)),
                    key=lambda x: o4_lag5.get(x, 0), reverse=True)[:8]
    y_pos = np.arange(len(sources))

    lag2_vals = [o4_lag2.get(s, 0) for s in sources]
    lag5_vals = [o4_lag5.get(s, 0) for s in sources]

    ax.barh(y_pos - 0.2, lag2_vals, 0.4, label='Lag 2', alpha=0.8, color='steelblue')
    ax.barh(y_pos + 0.2, lag5_vals, 0.4, label='Lag 5', alpha=0.8, color='coral')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s.replace('_', ' ') for s in sources], fontsize=9)
    ax.set_xlabel('Variance Explained (%)', fontsize=11, fontweight='bold')
    ax.set_title('O4 Drivers: Lag 2 vs Lag 5', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

# Plot 3: FEVD Heatmap for Lag 5
ax = axes[1, 0]
sns.heatmap(fevd_lag5_pivot, annot=True, fmt='.1f', cmap='YlOrRd',
           cbar_kws={'label': 'Variance (%)'}, ax=ax, linewidths=0.5)
ax.set_title('FEVD Heatmap - Lag 5 Model', fontsize=14, fontweight='bold')
ax.set_xlabel('Source', fontsize=11, fontweight='bold')
ax.set_ylabel('Target', fontsize=11, fontweight='bold')

# Plot 4: Model fit comparison
ax = axes[1, 1]
if lag2_success and lag5_success:
    metrics = ['AIC', 'BIC']
    lag2_metrics = [results_lag2.aic, results_lag2.bic]
    lag5_metrics = [results_lag5.aic, results_lag5.bic]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, lag2_metrics, width, label='Lag 2', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, lag5_metrics, width, label='Lag 5', alpha=0.8, color='coral')

    ax.set_ylabel('Criterion Value (lower is better)', fontsize=11, fontweight='bold')
    ax.set_title('Model Fit Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add annotations
    for i, (v2, v5) in enumerate(zip(lag2_metrics, lag5_metrics)):
        if v5 < v2:
            winner = "Lag 5 Better"
            color = 'green'
        else:
            winner = "Lag 2 Better"
            color = 'red'
        ax.text(i, max(v2, v5) * 1.1, winner, ha='center', fontweight='bold', color=color)

fig.suptitle('VAR Model Comparison: Lag 2 vs Lag 5 (AIC/BIC Optimal)',
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/LAG2_vs_LAG5_comparison.png',
           dpi=300, bbox_inches='tight')

print("[OK] Comparison visualizations saved")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 100)
print("SUMMARY: SHOULD YOU USE LAG 2 OR LAG 5?")
print("=" * 100)

print("\nMODEL FIT:")
if lag2_success and lag5_success:
    print(f"  Lag 2 - AIC: {results_lag2.aic:.4f}, BIC: {results_lag2.bic:.4f}")
    print(f"  Lag 5 - AIC: {results_lag5.aic:.4f}, BIC: {results_lag5.bic:.4f}")
    if results_lag5.aic < results_lag2.aic:
        print("  WINNER (AIC): Lag 5")
    else:
        print("  WINNER (AIC): Lag 2")

print("\nGRANGER CAUSALITY:")
if granger_lag2_df is not None:
    print(f"  Lag 2: {len(granger_lag2_sig)} significant relationships (5%)")
    print(f"  Lag 5: {len(granger_lag5_sig)} significant relationships (5%)")
    print(f"  Common: {len(common)}")
    print(f"  Stable: {len(common) / max(len(lag2_relationships), len(lag5_relationships)) * 100:.1f}%")

print("\nO4 VARIANCE EXPLAINED BY EXOGENOUS:")
if 'exog_lag2' in locals():
    print(f"  Lag 2: {exog_lag2:.2f}%")
    print(f"  Lag 5: {exog_lag5:.2f}%")
    print(f"  Change: {exog_lag5 - exog_lag2:+.2f} percentage points")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
print("\nFiles generated:")
print("  1. LAG5_granger_all.xlsx")
print("  2. LAG5_granger_significant.xlsx")
print("  3. LAG5_fevd_data.xlsx")
print("  4. LAG5_fevd_matrix.xlsx")
print("  5. LAG2_vs_LAG5_comparison.png")
print("\n" + "=" * 100)
