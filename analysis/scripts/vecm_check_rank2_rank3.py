"""
Check Rank=2 and Rank=3 Beta Importance
========================================
See if intermediate ranks (2, 3) provide more balanced story between
Rank=1 (GOFOs/Field Grade dominant) and Rank=6 (PAS dominant)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
OUTPUT_DIR = BASE_DIR / "VECM_Rank1_Final_Executive_Summary"

SELECTED_VARS = [
    'Junior_Enlisted_Z',
    'Company_Grade_Officers_Z',
    'Field_Grade_Officers_Z',
    'GOFOs_Z',
    'Warrant_Officers_Z',
    'Policy_Count_Log',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z'
]

DISPLAY_NAMES = {
    'Junior_Enlisted_Z': 'Junior Enlisted',
    'Company_Grade_Officers_Z': 'Company Grade',
    'Field_Grade_Officers_Z': 'Field Grade',
    'GOFOs_Z': 'GOFOs',
    'Warrant_Officers_Z': 'Warrant Officers',
    'Policy_Count_Log': 'Policy Volume',
    'Total_PAS_Z': 'Political Appointees (PAS)',
    'FOIA_Simple_Days_Z': 'FOIA Processing Delay'
}

print("=" * 80)
print("COMPARING BETA IMPORTANCE: RANK=1, RANK=2, RANK=3")
print("=" * 80)

# Load data
data_file = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\complete_normalized_dataset_v12.3.xlsx")
df = pd.read_excel(data_file)
df.columns = df.columns.str.strip()
data = df[SELECTED_VARS].dropna().copy()

print(f"\nData: {data.shape[0]} observations x {data.shape[1]} variables")

# Store beta importance for each rank
beta_importance = {}

for rank in [1, 2, 3]:
    print(f"\n{'='*80}")
    print(f"RANK={rank}")
    print(f"{'='*80}")

    vecm = VECM(data, k_ar_diff=1, coint_rank=rank, deterministic='nc')
    vecm_result = vecm.fit()

    beta = vecm_result.beta
    beta_df = pd.DataFrame(beta, index=SELECTED_VARS)

    print(f"\nBeta matrix shape: {beta_df.shape}")
    print("\nBeta coefficients:")
    print(beta_df.to_string())

    # Calculate total |beta| importance (sum across cointegration vectors)
    total_beta_importance = np.abs(beta_df).sum(axis=1)

    beta_importance[rank] = total_beta_importance.sort_values(ascending=False)

    print(f"\nBeta Importance (sum of |beta| across {rank} vector{'s' if rank > 1 else ''}):")
    for var, importance in beta_importance[rank].items():
        print(f"  {DISPLAY_NAMES[var]:30s}: {importance:8.2f}")

# Create comparison visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for idx, rank in enumerate([1, 2, 3]):
    ax = axes[idx]

    vars_sorted = beta_importance[rank].index
    values = beta_importance[rank].values
    display_names_sorted = [DISPLAY_NAMES[v] for v in vars_sorted]

    colors = ['red' if v == 'Total_PAS_Z' else
              'darkblue' if v == 'GOFOs_Z' else
              'navy' if v == 'Field_Grade_Officers_Z' else
              'orange' if v == 'FOIA_Simple_Days_Z' else
              'gray' for v in vars_sorted]

    ax.barh(range(len(vars_sorted)), values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(vars_sorted)))
    ax.set_yticklabels(display_names_sorted, fontsize=10)
    ax.set_xlabel('Total |Beta| Importance', fontsize=11, fontweight='bold')
    ax.set_title(f'Rank={rank}\n({rank} cointegration vector{"s" if rank > 1 else ""})',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, val in enumerate(values):
        ax.text(val + 2, i, f'{val:.1f}', va='center', fontsize=9)

fig.suptitle('Beta Importance Comparison: Does Rank Matter for Story?\n(Red=PAS, Blue=GOFOs, Navy=Field Grade, Orange=FOIA)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "beta_importance_comparison_rank123.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n{'='*80}")
print("COMPARISON SAVED")
print(f"{'='*80}")
print(f"\nPlot saved to: {OUTPUT_DIR / 'beta_importance_comparison_rank123.png'}")

# Create summary table
summary_data = []
for rank in [1, 2, 3]:
    for var in SELECTED_VARS:
        summary_data.append({
            'Rank': rank,
            'Variable': DISPLAY_NAMES[var],
            'Beta_Importance': beta_importance[rank][var],
            'Rank_Within_Model': list(beta_importance[rank].index).index(var) + 1
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_excel(OUTPUT_DIR / "beta_importance_rank123_comparison.xlsx", index=False)

print(f"Table saved to: {OUTPUT_DIR / 'beta_importance_rank123_comparison.xlsx'}")

# Print key findings
print(f"\n{'='*80}")
print("KEY FINDINGS")
print(f"{'='*80}")

print("\nTop 3 variables by rank:")
for rank in [1, 2, 3]:
    print(f"\nRank={rank}:")
    top3 = beta_importance[rank].head(3)
    for i, (var, val) in enumerate(top3.items(), 1):
        print(f"  {i}. {DISPLAY_NAMES[var]:30s}: {val:8.2f}")

print(f"\n{'='*80}")
