"""
Generate SIGNED Long-Run Influence Heatmap
Shows NET direction (positive vs negative) instead of just magnitude
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 100)
print("GENERATING SIGNED LONG-RUN INFLUENCE HEATMAP")
print("=" * 100)

# Configuration
INPUT_DIR = 'analysis/VECM_v12.3_Final'
OUTPUT_DIR = 'analysis/VECM_v12.3_Final/VECM_v12.3_Final_Executive_Summary'
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

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

SHORT_LABELS = [
    'Junior\nEnlisted',
    'Company\nGrade',
    'Field\nGrade',
    'GOFOs',
    'Warrant\nOfficers',
    'Policy\nCount',
    'Total\nPAS',
    'FOIA\nDays'
]

# Load coefficients
print("\nLoading VECM coefficients...")
alpha_df = pd.read_excel(f'{INPUT_DIR}/error_correction_alpha.xlsx', index_col=0)
beta_df = pd.read_excel(f'{INPUT_DIR}/cointegration_vectors_beta.xlsx', index_col=0)

# ============================================================================
# Calculate SIGNED long-run influence (preserves direction)
# ============================================================================
print("\nCalculating signed long-run influence...")

signed_influence = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))
unsigned_influence = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))

for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        signed_sum = 0
        unsigned_sum = 0

        for r in range(beta_df.shape[1]):  # For each cointegration vector
            alpha_i_r = alpha_df.iloc[i, r]  # Speed of adjustment for i
            beta_j_r = beta_df.iloc[j, r]    # Weight of j in equilibrium
            influence = alpha_i_r * beta_j_r

            signed_sum += influence      # Keep the sign
            unsigned_sum += abs(influence)  # Take absolute value

        signed_influence[i, j] = signed_sum
        unsigned_influence[i, j] = unsigned_sum

# ============================================================================
# Create signed influence heatmap
# ============================================================================
print("\nCreating signed influence heatmap...")

fig, ax = plt.subplots(figsize=(12, 10))

# Use diverging colormap (red=negative, blue=positive)
max_abs = np.max(np.abs(signed_influence))

sns.heatmap(signed_influence,
            xticklabels=SHORT_LABELS,
            yticklabels=SHORT_LABELS,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Net Effect (Signed)'},
            vmin=-max_abs,
            vmax=max_abs,
            linewidths=0.5,
            linecolor='gray',
            ax=ax)

ax.set_xlabel('From Variable (equilibrium deviation)', fontsize=12, fontweight='bold')
ax.set_ylabel('To Variable (adjustment)', fontsize=12, fontweight='bold')
ax.set_title('LONG-RUN INFLUENCE (SIGNED - Net Direction)\n' +
             'Positive = Reinforcing | Negative = Correcting',
             fontsize=14, fontweight='bold', pad=20)

# Add interpretation text
textstr = ('Red = Stabilizing (error correction)\n'
           'Blue = Destabilizing (reinforcing)\n'
           'Net sum of alpha[i,r] × beta[j,r]\n'
           'Shows actual direction of adjustment')
ax.text(1.15, 0.5, textstr, transform=ax.transAxes,
        fontsize=10, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/vecm_longrun_signed_influence.png', dpi=300, bbox_inches='tight')
print(f"  Saved: vecm_longrun_signed_influence.png")
plt.close()

# ============================================================================
# Create comparison: Signed vs Unsigned
# ============================================================================
print("\nCreating signed vs unsigned comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

# Signed (left)
sns.heatmap(signed_influence,
            xticklabels=SHORT_LABELS,
            yticklabels=SHORT_LABELS,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Net Effect'},
            vmin=-max_abs,
            vmax=max_abs,
            linewidths=0.5,
            linecolor='gray',
            ax=ax1)

ax1.set_xlabel('From Variable', fontsize=11, fontweight='bold')
ax1.set_ylabel('To Variable', fontsize=11, fontweight='bold')
ax1.set_title('SIGNED (Net Direction)\nRed=Correcting, Blue=Reinforcing',
              fontsize=13, fontweight='bold')

# Unsigned (right)
sns.heatmap(unsigned_influence,
            xticklabels=SHORT_LABELS,
            yticklabels=SHORT_LABELS,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Total Magnitude'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax2)

ax2.set_xlabel('From Variable', fontsize=11, fontweight='bold')
ax2.set_ylabel('To Variable', fontsize=11, fontweight='bold')
ax2.set_title('UNSIGNED (Total Magnitude)\nIgnores direction',
              fontsize=13, fontweight='bold')

plt.suptitle('LONG-RUN INFLUENCE: Signed vs Unsigned Comparison',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/vecm_longrun_signed_vs_unsigned.png', dpi=300, bbox_inches='tight')
print(f"  Saved: vecm_longrun_signed_vs_unsigned.png")
plt.close()

# ============================================================================
# Analysis: How much cancellation is happening?
# ============================================================================
print("\n" + "=" * 100)
print("CANCELLATION ANALYSIS")
print("=" * 100)

cancellation_ratio = np.abs(signed_influence) / (unsigned_influence + 1e-10)

print("\nRelationships with STRONG cancellation (opposing forces):")
print("(Low ratio = forces cancel out)")
print("-" * 100)

cancel_list = []
for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        if unsigned_influence[i, j] > 1.0:  # Only look at substantial relationships
            ratio = cancellation_ratio[i, j]
            cancel_list.append({
                'From': SELECTED_VARS[j],
                'To': SELECTED_VARS[i],
                'Signed': signed_influence[i, j],
                'Unsigned': unsigned_influence[i, j],
                'Ratio': ratio
            })

cancel_sorted = sorted(cancel_list, key=lambda x: x['Ratio'])

print(f"\n{'From':<30} {'To':<30} {'Signed':<10} {'Unsigned':<10} {'Ratio':<10}")
print("-" * 100)
for k, item in enumerate(cancel_sorted[:10], 1):
    print(f"{item['From']:<30} {item['To']:<30} {item['Signed']:>9.2f} {item['Unsigned']:>9.2f} {item['Ratio']:>9.2%}")

print("\n" + "=" * 100)
print("STRONGEST NET EFFECTS (After Cancellation)")
print("=" * 100)

net_list = []
for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        net_list.append({
            'From': SELECTED_VARS[j],
            'To': SELECTED_VARS[i],
            'Net_Effect': signed_influence[i, j],
            'Abs_Effect': abs(signed_influence[i, j])
        })

net_sorted = sorted(net_list, key=lambda x: x['Abs_Effect'], reverse=True)

print("\nTop 10 strongest NET effects:")
print(f"\n{'From':<30} {'To':<30} {'Net Effect':<12} {'Type':<15}")
print("-" * 100)
for k, item in enumerate(net_sorted[:10], 1):
    effect_type = 'Reinforcing (+)' if item['Net_Effect'] > 0 else 'Correcting (-)'
    print(f"{item['From']:<30} {item['To']:<30} {item['Net_Effect']:>11.2f} {effect_type:<15}")

print("\n" + "=" * 100)
print("FILES CREATED")
print("=" * 100)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("Files created:")
print("  1. vecm_longrun_signed_influence.png (SIGNED - shows direction)")
print("  2. vecm_longrun_signed_vs_unsigned.png (comparison)")
print("=" * 100)
