"""
Create Long-Run vs Short-Run Variable Importance Comparison for Rank=2
=======================================================================
Shows which variables are more important in equilibrium vs immediate dynamics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
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
    'Junior_Enlisted_Z': 'Junior Enlisted (E-1 to E-4)',
    'Company_Grade_Officers_Z': 'Company Grade (O-1 to O-3)',
    'Field_Grade_Officers_Z': 'Field Grade (O-4 to O-5)',
    'GOFOs_Z': 'General/Flag Officers',
    'Warrant_Officers_Z': 'Warrant Officers',
    'Policy_Count_Log': 'Policy Volume (Log)',
    'Total_PAS_Z': 'Political Appointees (PAS)',
    'FOIA_Simple_Days_Z': 'FOIA Processing Delay'
}

print("=" * 80)
print("CREATING LONG-RUN vs SHORT-RUN COMPARISON - RANK=2")
print("=" * 80)

# Load matrices
print("\n[1] Loading matrices...")
beta_df = pd.read_excel(INPUT_DIR / "beta_matrix_rank2.xlsx", index_col=0)
gamma_df = pd.read_excel(INPUT_DIR / "gamma_matrix_rank2.xlsx", index_col=0)

print(f"    Beta: {beta_df.shape}")
print(f"    Gamma: {gamma_df.shape}")

# Calculate long-run importance (beta)
print("\n[2] Calculating long-run importance...")
beta_importance = np.abs(beta_df).sum(axis=1)  # Sum across 2 cointegration vectors
print("    Beta importance calculated")

# Calculate short-run importance (gamma)
print("\n[3] Calculating short-run importance...")
# Sum of absolute gamma coefficients (how much variable influences others + is influenced)
gamma_importance = np.abs(gamma_df).sum(axis=0) + np.abs(gamma_df).sum(axis=1)
print("    Gamma importance calculated")

# Normalize to 0-100 scale
print("\n[4] Normalizing importance scores...")
beta_normalized = (beta_importance / beta_importance.max()) * 100
gamma_normalized = (gamma_importance / gamma_importance.max()) * 100

# Create DataFrame for plotting
importance_df = pd.DataFrame({
    'Variable': [DISPLAY_NAMES[var] for var in SELECTED_VARS],
    'Long_Run': beta_normalized.values,
    'Short_Run': gamma_normalized.values
})

# Sort by long-run importance (descending)
importance_df = importance_df.sort_values('Long_Run', ascending=True)

print("\n[5] Creating comparison chart...")

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

y_pos = np.arange(len(importance_df))
bar_height = 0.35

# Plot bars
bars1 = ax.barh(y_pos, importance_df['Long_Run'], bar_height,
                label='Long-Run Equilibrium', color='#FFD54F',
                edgecolor='black', linewidth=1.5, alpha=0.9)
bars2 = ax.barh(y_pos + bar_height, importance_df['Short_Run'], bar_height,
                label='Short-Run Dynamics', color='#81C784',
                edgecolor='black', linewidth=1.5, alpha=0.9)

# Customize
ax.set_yticks(y_pos + bar_height / 2)
ax.set_yticklabels(importance_df['Variable'], fontsize=11)
ax.set_xlabel('Importance Score (Normalized to 0-100)', fontsize=12, fontweight='bold')
ax.set_title('LONG-RUN vs SHORT-RUN Variable Importance\n(Shows which variables are more important in equilibrium vs immediate dynamics)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, 110)

# Add value labels on bars
for i, (lr, sr) in enumerate(zip(importance_df['Long_Run'], importance_df['Short_Run'])):
    if lr > 5:  # Only show if bar is visible
        ax.text(lr + 1, i, f'{lr:.0f}', va='center', fontsize=9, fontweight='bold')
    if sr > 5:
        ax.text(sr + 1, i + bar_height, f'{sr:.0f}', va='center', fontsize=9, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_longrun_vs_shortrun_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Comparison chart saved!")

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nVariable Importance Scores (0-100):")
print("-" * 80)
for _, row in importance_df.sort_values('Long_Run', ascending=False).iterrows():
    print(f"{row['Variable']:45s} LR={row['Long_Run']:5.1f}  SR={row['Short_Run']:5.1f}")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\nSaved to: {OUTPUT_DIR / 'vecm_longrun_vs_shortrun_comparison.png'}")
