"""
VECM Rank=2 Analysis - Tuberville Anomaly Impact (Version 2)
=============================================================
Proper experimental design:

BASELINE MODEL:
  - Training: 1987-2022 (36 obs, INCLUDES up to 2022)
  - Test: 2023-2024 (2 obs)
  - Tests model's ability to forecast the Tuberville period

NO-TUBERVILLE MODEL:
  - Training: 1987-2022 EXCLUDING 2023 data point (35 obs)
  - Test: 2024 only (1 obs)
  - Tests if excluding the anomaly improves general forecast

ALTERNATIVE COMPARISON:
  - Baseline: Full model 1987-2024 (38 obs) - in-sample fit
  - No-Tuberville: Model 1987-2024 excluding 2023 (37 obs) - in-sample fit
  - Compare: Residual diagnostics, cointegration strength
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
OUTPUT_DIR = BASE_DIR / "VECM_Rank2_No_Tuberville_v2"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

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
    'Policy_Count_Log': 'Policy Count (Log)',
    'Total_PAS_Z': 'Total PAS',
    'FOIA_Simple_Days_Z': 'FOIA Days'
}

print("=" * 80)
print("VECM RANK=2 - TUBERVILLE ANOMALY IMPACT ANALYSIS (V2)")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
data_file = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\complete_normalized_dataset_v12.3.xlsx")
df = pd.read_excel(data_file)
df.columns = df.columns.str.strip()

# Add year column if missing
if 'FY' in df.columns:
    year_col = 'FY'
elif 'Year' in df.columns:
    year_col = 'Year'
else:
    df['Year'] = range(1987, 1987 + len(df))
    year_col = 'Year'

# Clean data
data_full = df[SELECTED_VARS].dropna().copy()
years_full = df.loc[df[SELECTED_VARS].dropna().index, year_col].values

print(f"    Total observations: {len(data_full)}")
print(f"    Year range: {years_full.min():.0f} - {years_full.max():.0f}")

# Create clean dataframe
df_clean = pd.DataFrame(data_full.values, columns=SELECTED_VARS)
df_clean['Year'] = years_full

# Identify 2023
tuberville_idx = df_clean[df_clean['Year'] == 2023].index
if len(tuberville_idx) > 0:
    print(f"\n[2] Found 2023 at index: {tuberville_idx[0]}")
    print(f"    2023 GOFO value: {df_clean.loc[tuberville_idx[0], 'GOFOs_Z']:.4f}")
else:
    print("\n[2] ERROR: 2023 not found in dataset!")
    exit(1)

# ============================================================================
# EXPERIMENT 1: Out-of-sample forecast comparison
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: Out-of-Sample Forecast (2023-2024)")
print("=" * 80)

# Baseline: Train on 1987-2022, forecast 2023-2024
train_baseline = df_clean[df_clean['Year'] <= 2022][SELECTED_VARS]
test_2023_2024 = df_clean[df_clean['Year'] >= 2023][SELECTED_VARS]

print(f"\nBaseline Model:")
print(f"  Training: 1987-2022 ({len(train_baseline)} obs)")
print(f"  Test: 2023-2024 ({len(test_2023_2024)} obs)")

vecm_baseline = VECM(train_baseline, k_ar_diff=1, coint_rank=2, deterministic='nc')
result_baseline = vecm_baseline.fit()

forecast_baseline = result_baseline.predict(steps=len(test_2023_2024))
mae_baseline = np.mean(np.abs(forecast_baseline - test_2023_2024.values), axis=0)
mae_baseline_overall = np.mean(mae_baseline)

print(f"\n  Overall MAE: {mae_baseline_overall:.4f}")
print(f"  GOFOs MAE: {mae_baseline[SELECTED_VARS.index('GOFOs_Z')]:.4f}")

# No-Tuberville: Train on 1987-2024 EXCLUDING 2023, forecast 2024 only
df_no_tuberville = df_clean[df_clean['Year'] != 2023].copy()
train_no_tub = df_no_tuberville[df_no_tuberville['Year'] <= 2022][SELECTED_VARS]
test_2024 = df_no_tuberville[df_no_tuberville['Year'] == 2024][SELECTED_VARS]

print(f"\nNo-Tuberville Model:")
print(f"  Training: 1987-2022 excluding 2023 ({len(train_no_tub)} obs)")
print(f"  Test: 2024 only ({len(test_2024)} obs)")

vecm_no_tub = VECM(train_no_tub, k_ar_diff=1, coint_rank=2, deterministic='nc')
result_no_tub = vecm_no_tub.fit()

forecast_no_tub = result_no_tub.predict(steps=2)  # Forecast 2 steps to get to 2024
# Take only 2024 forecast (second step)
mae_no_tub_2024 = np.abs(forecast_no_tub[1, :] - test_2024.values[0, :])
mae_no_tub_overall = np.mean(mae_no_tub_2024)

print(f"\n  Overall MAE (2024 only): {mae_no_tub_overall:.4f}")
print(f"  GOFOs MAE: {mae_no_tub_2024[SELECTED_VARS.index('GOFOs_Z')]:.4f}")

# ============================================================================
# EXPERIMENT 2: In-sample model comparison
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: In-Sample Model Quality")
print("=" * 80)

# Baseline: ALL data (1987-2024, 38 obs)
data_baseline_full = df_clean[SELECTED_VARS]
vecm_full = VECM(data_baseline_full, k_ar_diff=1, coint_rank=2, deterministic='nc')
result_full = vecm_full.fit()

residuals_full = result_full.resid
sse_full = np.sum(residuals_full**2, axis=0)
sse_full_total = np.sum(sse_full)

print(f"\nBaseline (WITH 2023):")
print(f"  Observations: {len(data_baseline_full)}")
print(f"  Total SSE: {sse_full_total:.4f}")
print(f"  Mean SSE per variable: {np.mean(sse_full):.4f}")
print(f"  GOFOs SSE: {sse_full[SELECTED_VARS.index('GOFOs_Z')]:.4f}")

# No-Tuberville: Exclude 2023 (37 obs)
data_no_tub_full = df_clean[df_clean['Year'] != 2023][SELECTED_VARS]
vecm_no_tub_full = VECM(data_no_tub_full, k_ar_diff=1, coint_rank=2, deterministic='nc')
result_no_tub_full = vecm_no_tub_full.fit()

residuals_no_tub = result_no_tub_full.resid
sse_no_tub = np.sum(residuals_no_tub**2, axis=0)
sse_no_tub_total = np.sum(sse_no_tub)

print(f"\nNo-Tuberville (NO 2023):")
print(f"  Observations: {len(data_no_tub_full)}")
print(f"  Total SSE: {sse_no_tub_total:.4f}")
print(f"  Mean SSE per variable: {np.mean(sse_no_tub):.4f}")
print(f"  GOFOs SSE: {sse_no_tub[SELECTED_VARS.index('GOFOs_Z')]:.4f}")

# Compare
sse_improvement = sse_full_total - sse_no_tub_total
pct_improvement = (sse_improvement / sse_full_total) * 100

print(f"\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\nExperiment 1 (Out-of-sample):")
print(f"  Baseline MAE (2023-2024): {mae_baseline_overall:.4f}")
print(f"  No-Tuberville MAE (2024): {mae_no_tub_overall:.4f}")

print(f"\nExperiment 2 (In-sample fit):")
print(f"  Baseline SSE (WITH 2023): {sse_full_total:.4f}")
print(f"  No-Tuberville SSE (NO 2023): {sse_no_tub_total:.4f}")
print(f"  Improvement: {sse_improvement:.4f} ({pct_improvement:+.2f}%)")

if pct_improvement > 0:
    print(f"\n[+] Removing 2023 IMPROVED in-sample fit by {pct_improvement:.2f}%")
    print("    This suggests 2023 was indeed an outlier that harmed model quality.")
else:
    print(f"\n[-] Removing 2023 WORSENED in-sample fit by {abs(pct_improvement):.2f}%")
    print("    This suggests 2023 was not a severe outlier.")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Variable-specific SSE comparison
ax1 = axes[0, 0]
x = np.arange(len(SELECTED_VARS))
width = 0.35

bars1 = ax1.bar(x - width/2, sse_full, width, label='WITH 2023',
                color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax1.bar(x + width/2, sse_no_tub, width, label='NO 2023',
                color='#2ecc71', alpha=0.7, edgecolor='black')

ax1.set_ylabel('Sum of Squared Errors (SSE)', fontsize=11, fontweight='bold')
ax1.set_title('In-Sample Model Fit: Variable-Specific SSE', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([DISPLAY_NAMES[v] for v in SELECTED_VARS], rotation=45, ha='right', fontsize=9)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Total SSE comparison
ax2 = axes[0, 1]
models = ['WITH 2023', 'NO 2023']
sses = [sse_full_total, sse_no_tub_total]
colors = ['#e74c3c', '#2ecc71']

bars = ax2.bar(models, sses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Total SSE', fontsize=11, fontweight='bold')
ax2.set_title('Overall In-Sample Fit', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, sse in zip(bars, sses):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{sse:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 3: GOFO residuals over time (Baseline)
ax3 = axes[1, 0]
gofo_idx = SELECTED_VARS.index('GOFOs_Z')
# VECM with lag=1 loses first 2 observations (differencing + lag)
years_for_resid = df_clean['Year'].values[2:]  # Skip first 2 years
resid_gofos_full = residuals_full[:, gofo_idx]  # Residuals from full model

ax3.plot(years_for_resid, resid_gofos_full, marker='o', linewidth=2, markersize=6,
         color='#3498db', label='Residuals')
ax3.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

# Highlight 2023
if 2023 in years_for_resid:
    idx_2023 = np.where(years_for_resid == 2023)[0][0]
    if idx_2023 < len(resid_gofos_full):  # Check bounds
        ax3.scatter(2023, resid_gofos_full[idx_2023], s=200, color='red',
                    edgecolor='black', linewidth=2, zorder=5, label='2023 (Tuberville)')

ax3.set_xlabel('Year', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residual', fontsize=11, fontweight='bold')
ax3.set_title('GOFO Residuals (Baseline Model WITH 2023)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Comparison text summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
TUBERVILLE ANOMALY IMPACT

Context:
Senator Tuberville held all military
promotions in 2023, artificially
constraining GOFO appointments.

In-Sample Results:
  • Baseline SSE:    {sse_full_total:.2f}
  • No-Tub SSE:      {sse_no_tub_total:.2f}
  • Change:          {pct_improvement:+.2f}%

GOFO-Specific:
  • Baseline SSE:    {sse_full[gofo_idx]:.2f}
  • No-Tub SSE:      {sse_no_tub[gofo_idx]:.2f}

Conclusion:
"""

if pct_improvement > 0:
    summary_text += f"Removing 2023 improved model fit.\n2023 was indeed an outlier."
else:
    summary_text += f"2023 did not significantly harm\nmodel fit. Keep in analysis."

ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tuberville_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
print(f"\n[SAVED] {OUTPUT_DIR / 'tuberville_analysis_comprehensive.png'}")

# Save summary
with open(OUTPUT_DIR / 'ANALYSIS_SUMMARY.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("VECM RANK=2 - TUBERVILLE ANOMALY IMPACT ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

    f.write("EXPERIMENT 1: Out-of-Sample Forecast\n")
    f.write("-" * 80 + "\n")
    f.write(f"Baseline (train 1987-2022, test 2023-2024):\n")
    f.write(f"  Overall MAE: {mae_baseline_overall:.4f}\n")
    f.write(f"  GOFOs MAE:   {mae_baseline[SELECTED_VARS.index('GOFOs_Z')]:.4f}\n\n")

    f.write(f"No-Tuberville (exclude 2023, test 2024):\n")
    f.write(f"  Overall MAE: {mae_no_tub_overall:.4f}\n")
    f.write(f"  GOFOs MAE:   {mae_no_tub_2024[SELECTED_VARS.index('GOFOs_Z')]:.4f}\n\n")

    f.write("EXPERIMENT 2: In-Sample Model Fit\n")
    f.write("-" * 80 + "\n")
    f.write(f"Baseline (WITH 2023, n=38):\n")
    f.write(f"  Total SSE:   {sse_full_total:.4f}\n")
    f.write(f"  GOFOs SSE:   {sse_full[gofo_idx]:.4f}\n\n")

    f.write(f"No-Tuberville (NO 2023, n=37):\n")
    f.write(f"  Total SSE:   {sse_no_tub_total:.4f}\n")
    f.write(f"  GOFOs SSE:   {sse_no_tub[gofo_idx]:.4f}\n\n")

    f.write(f"Improvement: {sse_improvement:.4f} ({pct_improvement:+.2f}%)\n\n")

    f.write("CONCLUSION:\n")
    f.write("-" * 80 + "\n")
    if pct_improvement > 0:
        f.write(f"[+] Removing 2023 IMPROVED model fit by {pct_improvement:.2f}%.\n")
        f.write("This confirms the Tuberville hold created a significant outlier.\n")
        f.write("RECOMMENDATION: Consider sensitivity analysis with/without 2023.\n")
    else:
        f.write(f"[-] Removing 2023 did NOT improve model fit.\n")
        f.write("The Tuberville hold may not have been as severe as expected.\n")
        f.write("RECOMMENDATION: Keep 2023 in the analysis.\n")

print(f"[SAVED] {OUTPUT_DIR / 'ANALYSIS_SUMMARY.txt'}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
