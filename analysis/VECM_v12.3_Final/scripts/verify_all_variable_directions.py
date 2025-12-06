"""
Verify directional trends of all 8 VECM variables
Check if beta coefficients match empirical reality
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis")
DATA_FILE = BASE_DIR / "complete_normalized_dataset_v12.3.xlsx"
VECM_DIR = BASE_DIR / "VECM_v12.3_Final" / "VECM_Rank2_Final_Executive_Summary"
OUTPUT_DIR = VECM_DIR

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

print("=" * 80)
print("VERIFYING DIRECTIONAL TRENDS OF ALL 8 VECM VARIABLES")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
df = pd.read_excel(DATA_FILE)
df_clean = df.dropna(subset=SELECTED_VARS)
print(f"    Data shape after cleaning: {df_clean.shape}")
print(f"    Years: {df_clean['FY'].min():.0f} to {df_clean['FY'].max():.0f}")

# Calculate trends for each variable
print("\n[2] Calculating linear trends for each variable...")
trends = {}
for var in SELECTED_VARS:
    values = df_clean[var].values
    trend = stats.linregress(range(len(values)), values)
    trends[var] = {
        'slope': trend.slope,
        'r_squared': trend.rvalue**2,
        'direction': 'INCREASING' if trend.slope > 0 else 'DECREASING',
        'values': values
    }
    print(f"    {var:30s} slope={trend.slope:+.4f}  R2={trend.rvalue**2:.3f}  {trends[var]['direction']}")

# Load beta matrix
print("\n[3] Loading VECM beta coefficients...")
beta_df = pd.read_excel(VECM_DIR / "beta_matrix_rank2.xlsx", index_col=0)
print(f"    Beta matrix shape: {beta_df.shape}")
print("\nBeta coefficients:")
for var in SELECTED_VARS:
    ec1 = beta_df.loc[var, 'EC1']
    ec2 = beta_df.loc[var, 'EC2']
    print(f"    {var:30s} EC1={ec1:+.4f}  EC2={ec2:+.4f}")

# Calculate correlation matrix
print("\n[4] Calculating correlation matrix...")
corr_matrix = df_clean[SELECTED_VARS].corr()

# Identify which variables should have opposite signs
print("\n[5] Identifying variable relationships (for EC1 - primary cointegration)...")
print("\nEC1 Beta signs vs Empirical correlations:")
print("-" * 80)

# Use Junior_Enlisted as reference (beta = 1.0 in EC1)
reference_var = 'Junior_Enlisted_Z'
reference_beta_sign = np.sign(beta_df.loc[reference_var, 'EC1'])

print(f"\nReference variable: {reference_var} (beta_EC1 = {beta_df.loc[reference_var, 'EC1']:+.4f})")
print("\nFor other variables to be correctly specified:")
print("  - If correlated POSITIVELY with reference => beta should have SAME sign")
print("  - If correlated NEGATIVELY with reference => beta should have OPPOSITE sign")
print()

mismatches = []
for var in SELECTED_VARS:
    if var == reference_var:
        continue

    empirical_corr = corr_matrix.loc[reference_var, var]
    beta_ec1 = beta_df.loc[var, 'EC1']
    beta_sign = np.sign(beta_ec1)

    # Expected beta sign based on correlation
    if empirical_corr > 0:
        expected_sign = reference_beta_sign  # Same sign as reference
        relationship = "SAME direction"
    else:
        expected_sign = -reference_beta_sign  # Opposite sign
        relationship = "OPPOSITE direction"

    # Check if beta sign matches expected
    if abs(beta_ec1) < 0.01:  # Effectively zero
        match = "NOT IN EC1"
        mismatches.append((var, empirical_corr, beta_ec1, "Variable not in EC1"))
    elif beta_sign == expected_sign:
        match = "[OK] CORRECT"
    else:
        match = "[X] WRONG SIGN!"
        mismatches.append((var, empirical_corr, beta_ec1, f"Should be {'+' if expected_sign > 0 else '-'}"))

    print(f"  {var:30s}")
    print(f"    Correlation with reference: {empirical_corr:+.3f} ({relationship})")
    print(f"    Beta EC1: {beta_ec1:+.4f} (sign: {'+' if beta_sign > 0 else '-' if beta_sign < 0 else '0'})")
    print(f"    Status: {match}")
    print()

# Summary
print("\n[6] SUMMARY OF BETA SIGN ISSUES")
print("=" * 80)
if mismatches:
    print(f"\nFound {len(mismatches)} variables with incorrect beta signs:\n")
    for var, corr, beta, issue in mismatches:
        print(f"  • {var}")
        print(f"      Empirical correlation: {corr:+.3f}")
        print(f"      Current beta: {beta:+.4f}")
        print(f"      Issue: {issue}")
        print()
else:
    print("\n[OK] All beta signs match empirical correlations!")

# Create visualization
print("\n[7] Creating visualization...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Correlation heatmap
ax1 = fig.add_subplot(gs[0, :2])
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, mask=mask, cbar_kws={'label': 'Correlation'},
            linewidths=0.5, linecolor='black', ax=ax1)
ax1.set_title('Empirical Correlations (1987-2024)', fontsize=14, fontweight='bold')

# Plot 2: Beta coefficients heatmap
ax2 = fig.add_subplot(gs[0, 2])
beta_matrix = beta_df.values
sns.heatmap(beta_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            yticklabels=[v.replace('_Z', '').replace('_Log', '') for v in SELECTED_VARS],
            xticklabels=['EC1', 'EC2'], linewidths=0.5, linecolor='black', ax=ax2)
ax2.set_title('VECM Beta Coefficients', fontsize=14, fontweight='bold')

# Plot 3-8: Time series for each variable
for idx, var in enumerate(SELECTED_VARS):
    row = (idx // 3) + 1
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])

    years = df_clean['FY'].values
    values = trends[var]['values']
    slope = trends[var]['slope']

    # Plot data
    ax.plot(years, values, 'o-', linewidth=2, markersize=4, alpha=0.7)

    # Plot trend line
    x_vals = np.arange(len(values))
    trend_line = slope * x_vals + (values[0] - slope * 0)
    ax.plot(years, trend_line, '--', linewidth=2, color='red', alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Z-Score', fontsize=10)
    ax.set_title(f"{var.replace('_Z', '').replace('_Log', '')}\n(slope={slope:+.4f})",
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

fig.suptitle('Empirical Trends vs VECM Beta Coefficients - All 8 Variables',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(OUTPUT_DIR / "empirical_trends_all_variables.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Visualization saved!")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print(f"\nSaved to: {OUTPUT_DIR / 'empirical_trends_all_variables.png'}")

# Print recommendation
if mismatches:
    print("\n" + "!" * 80)
    print("RECOMMENDATION: VECM needs re-estimation with corrected beta signs")
    print("!" * 80)
    print("\nThe current VECM rank=2 specification has beta coefficients that don't")
    print("match the empirical directional relationships in the data.")
    print("\nOptions:")
    print("  1. Re-estimate with rank=1 (simpler model)")
    print("  2. Re-estimate with different variable ordering")
    print("  3. Use Johansen test with different normalization")
    print("  4. Manually specify which variables should have opposite signs")
else:
    print("\n[OK] Beta coefficients correctly reflect empirical relationships!")

print("=" * 80)
