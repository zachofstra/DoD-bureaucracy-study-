"""
Comparative Analysis: WITH vs WITHOUT Exogenous Variables
Shows what GDP_Growth and Major_Conflict add to the model
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("COMPARATIVE ANALYSIS: WITH vs WITHOUT EXOGENOUS VARIABLES")
print("=" * 80)

# Load both analyses
top8_path = 'data/analysis/older_models_analysis/FINAL_TOP8_VAR'
top9_path = 'data/analysis/FINAL_TOP9_WITH_EXOGENOUS'

try:
    # Read Granger causality results
    top8_granger = pd.read_excel(f'{top8_path}/granger_significant.xlsx')
    top9_granger = pd.read_excel(f'{top9_path}/granger_significant.xlsx')

    print("\n1. MODEL COMPLEXITY COMPARISON")
    print("-" * 80)
    print(f"{'Metric':<40} {'WITHOUT Exogenous':<20} {'WITH Exogenous':<20}")
    print("-" * 80)
    print(f"{'Variables':<40} {'8':<20} {'9':<20}")
    print(f"{'Significant Relationships (5%)':<40} {len(top8_granger[top8_granger['Significant_5pct']==True]):<20} {len(top9_granger[top9_granger['Significant_5pct']==True]):<20}")
    print(f"{'Significant Relationships (1%)':<40} {len(top8_granger[top8_granger['Significant_1pct']==True]):<20} {len(top9_granger[top9_granger['Significant_1pct']==True]):<20}")

    # Compare network statistics
    top8_centrality = pd.read_excel(f'{top8_path}/network_centrality.xlsx')
    top9_centrality = pd.read_excel(f'{top9_path}/network_centrality.xlsx')

    print("\n2. NETWORK TOPOLOGY COMPARISON")
    print("-" * 80)
    print(f"{'Network Metric':<40} {'WITHOUT Exogenous':<20} {'WITH Exogenous':<20}")
    print("-" * 80)
    print(f"{'Total Nodes':<40} {len(top8_centrality):<20} {len(top9_centrality):<20}")
    print(f"{'Total Edges (5% sig)':<40} {top8_centrality['Out_Degree'].sum():<20} {top9_centrality['Out_Degree'].sum():<20}")
    print(f"{'Average Degree':<40} {top8_centrality['Total_Degree'].mean():.2f} {top9_centrality['Total_Degree'].mean():.2f}")

    # Compare FEVD for O4
    top8_fevd = pd.read_excel(f'{top8_path}/fevd_data.xlsx')
    top9_fevd = pd.read_excel(f'{top9_path}/fevd_data.xlsx')

    # Extract O4 variance at maximum available step
    max_step_top8 = top8_fevd['Step'].max()
    max_step_top9 = top9_fevd['Step'].max()

    top8_o4 = top8_fevd[top8_fevd['Step'] == max_step_top8]
    top9_o4 = top9_fevd[top9_fevd['Step'] == max_step_top9]

    print(f"\nUsing Step {max_step_top8} for TOP8, Step {max_step_top9} for TOP9")

    top8_o4_cols = [col for col in top8_fevd.columns if col.startswith('O4_MajorLTCDR_Pct_from_')]
    top9_o4_cols = [col for col in top9_fevd.columns if col.startswith('O4_MajorLTCDR_Pct_from_')]

    print("\n3. O4 VARIANCE DECOMPOSITION (Step 10)")
    print("-" * 80)

    # Build comparison table
    comparison = []

    # Get top8 values (already in percentage from some FEVD files)
    top8_vars = {}
    for col in top8_o4_cols:
        var = col.replace('O4_MajorLTCDR_Pct_from_', '')
        val = top8_o4[col].iloc[0]
        # Check if already in percentage (>1) or fraction (<1)
        top8_vars[var] = val if val > 1 else val * 100

    # Get top9 values
    top9_vars = {}
    for col in top9_o4_cols:
        var = col.replace('O4_MajorLTCDR_Pct_from_', '')
        val = top9_o4[col].iloc[0]
        top9_vars[var] = val if val > 1 else val * 100

    # Combine
    all_vars = set(list(top8_vars.keys()) + list(top9_vars.keys()))

    print(f"{'Variable':<30} {'WITHOUT Exog (%)':<20} {'WITH Exog (%)':<20} {'Change':<15}")
    print("-" * 80)

    for var in sorted(all_vars, key=lambda x: top9_vars.get(x, 0), reverse=True):
        val_without = top8_vars.get(var, 0)
        val_with = top9_vars.get(var, 0)
        change = val_with - val_without

        change_str = f"{change:+.2f}%" if var in top8_vars else "NEW"
        print(f"{var:<30} {val_without:>18.2f}% {val_with:>18.2f}% {change_str:>15}")

    # Calculate category totals for top9
    exog_total = top9_vars.get('GDP_Growth', 0) + top9_vars.get('Major_Conflict', 0)
    admin_total = top9_vars.get('Policy_Count', 0) + top9_vars.get('Total_Civilians', 0) + top9_vars.get('Total_PAS', 0)
    military_total = sum([v for k, v in top9_vars.items() if k.startswith('O') or k.startswith('E')])

    print("-" * 80)
    print(f"{'EXOGENOUS TOTAL (GDP+Conflict)':<30} {'':<20} {exog_total:>18.2f}%")
    print(f"{'ADMINISTRATIVE TOTAL':<30} {'':<20} {admin_total:>18.2f}%")
    print(f"{'MILITARY TOTAL':<30} {'':<20} {military_total:>18.2f}%")

    # Create visualization comparing FEVD
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor='white')

    # Sort variables
    top8_sorted = pd.Series(top8_vars).sort_values(ascending=True)
    top9_sorted = pd.Series(top9_vars).sort_values(ascending=True)

    # Plot WITHOUT exogenous
    ax1.barh(range(len(top8_sorted)), top8_sorted.values,
            color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_yticks(range(len(top8_sorted)))
    ax1.set_yticklabels([v.replace('_', ' ') for v in top8_sorted.index], fontsize=10)
    ax1.set_xlabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax1.set_title('WITHOUT Exogenous Variables\n(8-Variable Model)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    for i, val in enumerate(top8_sorted.values):
        ax1.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)

    # Plot WITH exogenous
    colors_with = ['red' if var in ['GDP_Growth', 'Major_Conflict'] else 'steelblue'
                  for var in top9_sorted.index]

    ax2.barh(range(len(top9_sorted)), top9_sorted.values,
            color=colors_with, alpha=0.8, edgecolor='black')
    ax2.set_yticks(range(len(top9_sorted)))
    ax2.set_yticklabels([v.replace('_', ' ') for v in top9_sorted.index], fontsize=10)
    ax2.set_xlabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax2.set_title('WITH Exogenous Variables\n(9-Variable Model)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    for i, val in enumerate(top9_sorted.values):
        ax2.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.8, label='Endogenous Variables'),
        Patch(facecolor='red', alpha=0.8, label='Exogenous Variables (NEW)')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')

    fig.suptitle('O4 (Major/LTCDR) Bureaucratic Bloat Drivers Comparison\nForecast Error Variance Decomposition (10 steps ahead)',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/comparison_with_vs_without_exogenous.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    print("\n[OK] Comparison visualization saved")

    # Identify new relationships from exogenous variables
    print("\n4. NEW CAUSAL RELATIONSHIPS (added by exogenous variables)")
    print("-" * 80)

    # Filter for exogenous causes
    exog_causes = top9_granger[
        (top9_granger['Cause'].isin(['GDP_Growth', 'Major_Conflict'])) &
        (top9_granger['Significant_5pct'] == True)
    ].sort_values('p_value')

    exog_effects = top9_granger[
        (top9_granger['Effect'].isin(['GDP_Growth', 'Major_Conflict'])) &
        (top9_granger['Significant_5pct'] == True)
    ].sort_values('p_value')

    print("\nEXOGENOUS -> ENDOGENOUS relationships:")
    print(f"{'Cause':<20} {'Effect':<25} {'F-stat':<12} {'p-value':<12}")
    print("-" * 70)
    for _, row in exog_causes.iterrows():
        print(f"{row['Cause']:<20} {row['Effect']:<25} {row['F_statistic']:<12.2f} {row['p_value']:<12.6f}")

    print("\nENDOGENOUS -> EXOGENOUS relationships:")
    print(f"{'Cause':<25} {'Effect':<20} {'F-stat':<12} {'p-value':<12}")
    print("-" * 70)
    for _, row in exog_effects.iterrows():
        print(f"{row['Cause']:<25} {row['Effect']:<20} {row['F_statistic']:<12.2f} {row['p_value']:<12.6f}")

    print("\n" + "=" * 80)
    print("KEY FINDING:")
    print(f"Exogenous factors (GDP + Conflict) explain {exog_total:.1f}% of O4 variance")
    print(f"This means {100-exog_total:.1f}% is driven by endogenous bureaucratic dynamics")
    print("=" * 80)

    # Create summary table
    summary_data = {
        'Metric': [
            'Number of Variables',
            'Significant Relationships (5%)',
            'Network Edges',
            'O4 Variance: Exogenous (%)',
            'O4 Variance: Endogenous (%)',
            'O4 Self-Perpetuation (%)'
        ],
        'WITHOUT_Exogenous': [
            8,
            len(top8_granger[top8_granger['Significant_5pct']==True]),
            int(top8_centrality['Out_Degree'].sum()),
            0,
            100 - top8_vars.get('O4_MajorLTCDR_Pct', 0),
            top8_vars.get('O4_MajorLTCDR_Pct', 0)
        ],
        'WITH_Exogenous': [
            9,
            len(top9_granger[top9_granger['Significant_5pct']==True]),
            int(top9_centrality['Out_Degree'].sum()),
            exog_total,
            100 - exog_total - top9_vars.get('O4_MajorLTCDR_Pct', 0),
            top9_vars.get('O4_MajorLTCDR_Pct', 0)
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/model_comparison_summary.xlsx', index=False)
    print("\n[OK] Summary table saved: model_comparison_summary.xlsx")

except Exception as e:
    print(f"\nERROR: {e}")
    print("\nNote: This comparison requires both FINAL_TOP8_VAR and FINAL_TOP9_WITH_EXOGENOUS directories")
    print("If FINAL_TOP8_VAR is in older_models_analysis, please copy it to data/analysis/")
