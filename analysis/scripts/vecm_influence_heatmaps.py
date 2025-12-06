"""
Generate Influence Heatmaps for VECM v12.3
- Short-run: Gamma coefficients (direct VAR effects)
- Long-run: Alpha × Beta (error correction influence)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 100)
print("GENERATING VECM INFLUENCE HEATMAPS")
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

# Short labels for better visualization
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

# ============================================================================
# Load VECM coefficients
# ============================================================================
print("\nLoading VECM coefficients...")

gamma_df = pd.read_excel(f'{INPUT_DIR}/short_run_gamma_lag1.xlsx', index_col=0)
alpha_df = pd.read_excel(f'{INPUT_DIR}/error_correction_alpha.xlsx', index_col=0)
beta_df = pd.read_excel(f'{INPUT_DIR}/cointegration_vectors_beta.xlsx', index_col=0)

print(f"  Gamma (short-run): {gamma_df.shape}")
print(f"  Alpha (error correction): {alpha_df.shape}")
print(f"  Beta (cointegration): {beta_df.shape}")

# ============================================================================
# HEATMAP 1: Short-Run Dynamics (Gamma)
# ============================================================================
print("\nCreating short-run influence heatmap...")

fig, ax = plt.subplots(figsize=(12, 10))

# Gamma shows: effect of lag of column variable on row variable
# gamma[i,j] = effect of j(t-1) on i(t)
gamma_matrix = gamma_df.values

# Create heatmap
sns.heatmap(gamma_matrix,
            xticklabels=SHORT_LABELS,
            yticklabels=SHORT_LABELS,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Coefficient Magnitude'},
            vmin=-np.abs(gamma_matrix).max(),
            vmax=np.abs(gamma_matrix).max(),
            linewidths=0.5,
            linecolor='gray',
            ax=ax)

ax.set_xlabel('From Variable (t-1)', fontsize=12, fontweight='bold')
ax.set_ylabel('To Variable (t)', fontsize=12, fontweight='bold')
ax.set_title('SHORT-RUN INFLUENCE (Gamma Coefficients)\n' +
             'How each variable\'s past value affects others\' current values',
             fontsize=14, fontweight='bold', pad=20)

# Add interpretation text
textstr = ('Red = Negative effect (dampening)\n'
           'Blue = Positive effect (amplifying)\n'
           'Read across rows: How variable j(t-1) affects variable i(t)')
ax.text(1.15, 0.5, textstr, transform=ax.transAxes,
        fontsize=10, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/vecm_shortrun_influence_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  Saved: vecm_shortrun_influence_heatmap.png")
plt.close()

# ============================================================================
# HEATMAP 2: Long-Run Dynamics (Alpha × Beta)
# ============================================================================
print("\nCreating long-run influence heatmap...")

# Calculate long-run influence matrix
# Influence of variable j on variable i = sum over r of |alpha[i,r] * beta[j,r]|
longrun_influence = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))

for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        if i != j:
            total_influence = 0
            for r in range(beta_df.shape[1]):  # For each cointegration vector
                alpha_i_r = alpha_df.iloc[i, r]  # Speed of adjustment for i
                beta_j_r = beta_df.iloc[j, r]    # Weight of j in equilibrium
                influence = alpha_i_r * beta_j_r
                total_influence += abs(influence)
            longrun_influence[i, j] = total_influence
        else:
            # Diagonal: self-adjustment through all equilibria
            total_influence = 0
            for r in range(beta_df.shape[1]):
                alpha_i_r = alpha_df.iloc[i, r]
                beta_i_r = beta_df.iloc[i, r]
                influence = alpha_i_r * beta_i_r
                total_influence += abs(influence)
            longrun_influence[i, i] = total_influence

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(longrun_influence,
            xticklabels=SHORT_LABELS,
            yticklabels=SHORT_LABELS,
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Total Influence Magnitude'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax)

ax.set_xlabel('From Variable (equilibrium deviation)', fontsize=12, fontweight='bold')
ax.set_ylabel('To Variable (adjustment speed)', fontsize=12, fontweight='bold')
ax.set_title('LONG-RUN INFLUENCE (Error Correction Mechanism)\n' +
             'How each variable\'s deviation from equilibrium affects others\' adjustments',
             fontsize=14, fontweight='bold', pad=20)

# Add interpretation text
textstr = ('Darker = Stronger influence\n'
           'Diagonal = Self-correction\n'
           'Off-diagonal = Cross-variable effects\n'
           'Sum of |alpha[i,r] × beta[j,r]| over all r')
ax.text(1.15, 0.5, textstr, transform=ax.transAxes,
        fontsize=10, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/vecm_longrun_influence_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  Saved: vecm_longrun_influence_heatmap.png")
plt.close()

# ============================================================================
# COMBINED FIGURE: Side-by-Side Comparison
# ============================================================================
print("\nCreating combined comparison heatmap...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

# Short-run (left)
sns.heatmap(gamma_matrix,
            xticklabels=SHORT_LABELS,
            yticklabels=SHORT_LABELS,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Coefficient'},
            vmin=-np.abs(gamma_matrix).max(),
            vmax=np.abs(gamma_matrix).max(),
            linewidths=0.5,
            linecolor='gray',
            ax=ax1)

ax1.set_xlabel('From Variable (t-1)', fontsize=11, fontweight='bold')
ax1.set_ylabel('To Variable (t)', fontsize=11, fontweight='bold')
ax1.set_title('SHORT-RUN DYNAMICS\n(Gamma Coefficients)',
              fontsize=13, fontweight='bold')

# Long-run (right)
sns.heatmap(longrun_influence,
            xticklabels=SHORT_LABELS,
            yticklabels=SHORT_LABELS,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Total Influence'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax2)

ax2.set_xlabel('From Variable (deviation)', fontsize=11, fontweight='bold')
ax2.set_ylabel('To Variable (adjustment)', fontsize=11, fontweight='bold')
ax2.set_title('LONG-RUN DYNAMICS\n(Error Correction)',
              fontsize=13, fontweight='bold')

plt.suptitle('VECM INFLUENCE COMPARISON: Short-Run vs Long-Run',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/vecm_influence_comparison.png', dpi=300, bbox_inches='tight')
print(f"  Saved: vecm_influence_comparison.png")
plt.close()

# ============================================================================
# Generate summary statistics
# ============================================================================
print("\n" + "=" * 100)
print("INFLUENCE SUMMARY STATISTICS")
print("=" * 100)

print("\nShort-Run (Gamma) Statistics:")
print(f"  Mean |coefficient|: {np.mean(np.abs(gamma_matrix)):.4f}")
print(f"  Max |coefficient|: {np.max(np.abs(gamma_matrix)):.4f}")
print(f"  Min |coefficient|: {np.min(np.abs(gamma_matrix)):.4f}")
print(f"\nStrongest short-run effects:")
gamma_flat = []
for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        gamma_flat.append({
            'From': SELECTED_VARS[j],
            'To': SELECTED_VARS[i],
            'Coefficient': gamma_matrix[i, j],
            'Abs_Coef': abs(gamma_matrix[i, j])
        })
gamma_ranked = sorted(gamma_flat, key=lambda x: x['Abs_Coef'], reverse=True)
for k, effect in enumerate(gamma_ranked[:5], 1):
    print(f"  {k}. {effect['From']:30s} -> {effect['To']:30s}: {effect['Coefficient']:7.4f}")

print("\nLong-Run (Alpha × Beta) Statistics:")
print(f"  Mean influence: {np.mean(longrun_influence):.4f}")
print(f"  Max influence: {np.max(longrun_influence):.4f}")
print(f"  Min influence: {np.min(longrun_influence):.4f}")
print(f"\nStrongest long-run influences:")
longrun_flat = []
for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        longrun_flat.append({
            'From': SELECTED_VARS[j],
            'To': SELECTED_VARS[i],
            'Influence': longrun_influence[i, j]
        })
longrun_ranked = sorted(longrun_flat, key=lambda x: x['Influence'], reverse=True)
for k, effect in enumerate(longrun_ranked[:5], 1):
    print(f"  {k}. {effect['From']:30s} -> {effect['To']:30s}: {effect['Influence']:7.4f}")

print("\n" + "=" * 100)
print("HEATMAPS GENERATED SUCCESSFULLY")
print("=" * 100)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("Files created:")
print("  1. vecm_shortrun_influence_heatmap.png")
print("  2. vecm_longrun_influence_heatmap.png")
print("  3. vecm_influence_comparison.png")
print("=" * 100)
