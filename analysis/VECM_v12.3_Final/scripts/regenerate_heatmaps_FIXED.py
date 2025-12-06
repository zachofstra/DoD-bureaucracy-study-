"""
Regenerate influence comparison heatmaps with FIXED sign direction
(Removed the incorrect * (-1) flip)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
INPUT_DIR = BASE_DIR / "VECM_Rank2_Final_Executive_Summary"
OUTPUT_DIR = INPUT_DIR

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
    'Junior_Enlisted_Z': 'Junior\nEnlisted',
    'Company_Grade_Officers_Z': 'Company\nGrade',
    'Field_Grade_Officers_Z': 'Field\nGrade',
    'GOFOs_Z': 'GOFOs',
    'Warrant_Officers_Z': 'Warrant\nOfficers',
    'Policy_Count_Log': 'Policy\nCount',
    'Total_PAS_Z': 'Total\nPAS',
    'FOIA_Simple_Days_Z': 'FOIA\nDays'
}

print("=" * 80)
print("REGENERATING HEATMAPS WITH FIXED SIGN DIRECTION")
print("=" * 80)

# Load matrices
print("\n[1] Loading matrices...")
alpha_df = pd.read_excel(INPUT_DIR / "alpha_matrix_rank2.xlsx", index_col=0)
beta_df = pd.read_excel(INPUT_DIR / "beta_matrix_rank2.xlsx", index_col=0)
gamma_df = pd.read_excel(INPUT_DIR / "gamma_matrix_rank2.xlsx", index_col=0)
longrun_df = pd.read_excel(INPUT_DIR / "longrun_influence_rank2.xlsx", index_col=0)

print(f"    Alpha: {alpha_df.shape}")
print(f"    Beta: {beta_df.shape}")
print(f"    Gamma: {gamma_df.shape}")
print(f"    Long-run: {longrun_df.shape}")

# Calculate signed direction (CORRECTLY, without the flip)
print("\n[2] Calculating signed direction (FIXED)...")
longrun_influence = longrun_df.values
signed_direction = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))

for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        signed_sum = 0
        for r in range(2):  # rank=2
            alpha_i = alpha_df.iloc[i, r]
            beta_j = beta_df.iloc[j, r]
            signed_sum += alpha_i * beta_j
        signed_direction[i, j] = np.sign(signed_sum)

# FIXED: No more * (-1) flip!
signed_magnitude = longrun_influence * signed_direction

print("    Signed direction calculated (without flip)")

# Create heatmaps
print("\n[3] Creating heatmaps...")

display_names = [DISPLAY_NAMES[var] for var in SELECTED_VARS]
gamma_values = gamma_df.values

# Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Short-run (LEFT)
sns.heatmap(gamma_values, annot=np.abs(gamma_values), fmt='.2f',
            cmap='RdBu_r', center=0, cbar_kws={'label': 'Coefficient'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=axes[0])
axes[0].set_title('SHORT-RUN DYNAMICS (Gamma)\nYear-to-year VAR effects',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('From Variable (t-1)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('To Variable', fontsize=11, fontweight='bold')

# Long-run (RIGHT) - FIXED COLORS
sns.heatmap(signed_magnitude, annot=longrun_influence, fmt='.2f',
            cmap='RdBu_r', center=0, cbar_kws={'label': 'Direction (RED=Amplifying, BLUE=Dampening)'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=axes[1])
axes[1].set_title('LONG-RUN INFLUENCE (Error Correction)\nMagnitude with directional coloring (sum across 2 vectors)',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('From Variable (equilibrium deviation)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('To Variable (adjustment)', fontsize=11, fontweight='bold')

fig.suptitle('VECM INFLUENCE COMPARISON: Short-Run vs Long-Run (RANK=2)\n(Magnitude values with directional coloring)',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_influence_comparison_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Comparison heatmap saved (FIXED)")

# Individual long-run heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(signed_magnitude, annot=longrun_influence, fmt='.2f',
            cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Direction (RED=Amplifying, BLUE=Dampening)'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=ax)
ax.set_title('LONG-RUN INFLUENCE (Error Correction) - Rank=2\nMagnitude with directional coloring (sum across 2 vectors)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('From Variable (equilibrium deviation)', fontsize=12, fontweight='bold')
ax.set_ylabel('To Variable (adjustment)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_longrun_influence_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Individual long-run heatmap saved (FIXED)")

print("\n" + "=" * 80)
print("COMPLETE! Heatmaps regenerated with correct sign direction")
print("=" * 80)
print(f"\nFiles saved to: {OUTPUT_DIR}")
print("\nKey fixes:")
print("  - Removed incorrect (* -1) flip")
print("  - RED now correctly shows AMPLIFYING relationships")
print("  - BLUE now correctly shows DAMPENING relationships")
print("\nGenerated:")
print("  1. vecm_influence_comparison_rank2.png (FIXED)")
print("  2. vecm_longrun_influence_rank2.png (FIXED)")
print("=" * 80)
