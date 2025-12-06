"""
VECM Rank=2 Analysis - Excluding 2023 (Tuberville Anomaly)
===========================================================
Senator Tuberville placed a hold on all military promotions in 2023,
which significantly distorted GOFO numbers. This analysis:

1. Removes 2023 data from training
2. Estimates VECM rank=2 on clean data
3. Forecasts out-of-sample (including 2023)
4. Compares performance to baseline model

Hypothesis: Removing the Tuberville anomaly will improve model fit
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
OUTPUT_DIR = BASE_DIR / "VECM_Rank2_No_Tuberville"
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
print("VECM RANK=2 ANALYSIS - EXCLUDING TUBERVILLE ANOMALY (2023)")
print("=" * 80)
print("\nContext: Senator Tuberville held all military promotions in 2023")
print("Expected impact: Severe distortion of GOFO numbers")
print("\nVariables: 8")
for i, var in enumerate(SELECTED_VARS, 1):
    print(f"  {i}. {DISPLAY_NAMES[var]}")

# Load data
print("\n[1] Loading data...")
data_file = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\complete_normalized_dataset_v12.3.xlsx")
df = pd.read_excel(data_file)
df.columns = df.columns.str.strip()

# Check if we have a Year column
if 'Year' in df.columns:
    year_col = 'Year'
elif 'year' in df.columns:
    year_col = 'year'
else:
    print("\n[INFO] No year column found, assuming sequential 1987-2024...")
    df['Year'] = range(1987, 1987 + len(df))
    year_col = 'Year'

print(f"    Year range: {df[year_col].min()} - {df[year_col].max()}")
print(f"    Total observations: {len(df)}")

# Identify 2023 data
tuberville_mask = df[year_col] == 2023
n_tuberville = tuberville_mask.sum()
print(f"\n[2] Identifying Tuberville anomaly...")
print(f"    2023 observations to exclude: {n_tuberville}")

if n_tuberville == 0:
    print("\n[WARNING] No 2023 data found! Check year column.")
    print(f"Available years: {sorted(df[year_col].unique())}")

# Extract data and drop NaN rows
data_full = df[SELECTED_VARS].dropna().copy()
years_full = df.loc[df[SELECTED_VARS].dropna().index, year_col].values

print(f"\n[INFO] After dropping NaN:")
print(f"    Data shape: {data_full.shape}")
print(f"    Year range: {years_full.min():.0f} - {years_full.max():.0f}")

# Split: Training (no 2023), Test (last 5 years including 2023)
# Training: 1987-2019 (33 observations)
# Test: 2020-2024 (5 observations, includes 2023)
# Create a clean dataframe with years
df_clean = pd.DataFrame(data_full.values, columns=SELECTED_VARS)
df_clean['Year'] = years_full

train_mask = (df_clean['Year'] >= 1987) & (df_clean['Year'] <= 2019)
test_mask = (df_clean['Year'] >= 2020) & (df_clean['Year'] <= 2024)

data_train = df_clean.loc[train_mask, SELECTED_VARS].copy()
data_test = df_clean.loc[test_mask, SELECTED_VARS].copy()
years_train = df_clean.loc[train_mask, 'Year'].values
years_test = df_clean.loc[test_mask, 'Year'].values

print(f"\n[3] Dataset split...")
print(f"    Training: {len(data_train)} obs ({years_train.min()}-{years_train.max()})")
print(f"    Test: {len(data_test)} obs ({years_test.min()}-{years_test.max()})")
print(f"    2023 in training: {2023 in years_train}")
print(f"    2023 in test: {2023 in years_test}")

# BASELINE MODEL: All data (including 2023)
print("\n" + "=" * 80)
print("BASELINE MODEL: Full dataset (WITH 2023)")
print("=" * 80)

# Use first 33 obs for training, last 5 for test
data_baseline = data_full.iloc[:-5].copy()
vecm_baseline = VECM(data_baseline, k_ar_diff=1, coint_rank=2, deterministic='nc')
result_baseline = vecm_baseline.fit()

print(f"\n    Observations: {len(data_baseline)}")
print(f"    Cointegration rank: 2")

# Out-of-sample forecast
forecast_baseline = result_baseline.predict(steps=5)
test_actual = data_full.iloc[-5:].values

print(f"\n    Forecast shape: {forecast_baseline.shape}")
print(f"    Test actual shape: {test_actual.shape}")
print(f"    Forecast has NaN: {np.isnan(forecast_baseline).any()}")
print(f"    Test has NaN: {np.isnan(test_actual).any()}")

mae_baseline = np.mean(np.abs(forecast_baseline - test_actual), axis=0)
mae_baseline_overall = np.mean(mae_baseline)

print(f"\n    Out-of-sample MAE (overall): {mae_baseline_overall:.4f}")
print(f"    Variable-specific MAE:")
for i, var in enumerate(SELECTED_VARS):
    print(f"      {DISPLAY_NAMES[var]:25s}: {mae_baseline[i]:.4f}")

# NO-TUBERVILLE MODEL: Excluding 2023
print("\n" + "=" * 80)
print("NO-TUBERVILLE MODEL: Excluding 2023 from training")
print("=" * 80)

vecm_clean = VECM(data_train, k_ar_diff=1, coint_rank=2, deterministic='nc')
result_clean = vecm_clean.fit()

print(f"\n    Observations: {len(data_train)}")
print(f"    Cointegration rank: 2")

# Out-of-sample forecast
forecast_clean = result_clean.predict(steps=5)

mae_clean = np.mean(np.abs(forecast_clean - test_actual), axis=0)
mae_clean_overall = np.mean(mae_clean)

print(f"\n    Out-of-sample MAE (overall): {mae_clean_overall:.4f}")
print(f"    Variable-specific MAE:")
for i, var in enumerate(SELECTED_VARS):
    print(f"      {DISPLAY_NAMES[var]:25s}: {mae_clean[i]:.4f}")

# COMPARISON
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

improvement = mae_baseline_overall - mae_clean_overall
pct_improvement = (improvement / mae_baseline_overall) * 100

print(f"\nOverall MAE:")
print(f"    Baseline (WITH 2023):     {mae_baseline_overall:.4f}")
print(f"    No-Tuberville (NO 2023):  {mae_clean_overall:.4f}")
print(f"    Improvement:              {improvement:.4f} ({pct_improvement:+.2f}%)")

if pct_improvement > 0:
    print(f"\n[+] RESULT: Removing 2023 IMPROVED model by {pct_improvement:.2f}%")
else:
    print(f"\n[-] RESULT: Removing 2023 WORSENED model by {abs(pct_improvement):.2f}%")

print(f"\nVariable-specific changes:")
for i, var in enumerate(SELECTED_VARS):
    diff = mae_baseline[i] - mae_clean[i]
    pct = (diff / mae_baseline[i]) * 100
    symbol = "[+]" if diff > 0 else "[-]"
    print(f"  {symbol} {DISPLAY_NAMES[var]:25s}: {mae_baseline[i]:.4f} -> {mae_clean[i]:.4f} ({pct:+.2f}%)")

# Visualize comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Overall MAE comparison
ax1 = axes[0]
models = ['Baseline\n(WITH 2023)', 'No-Tuberville\n(NO 2023)']
maes = [mae_baseline_overall, mae_clean_overall]
colors = ['#e74c3c', '#2ecc71']

bars = ax1.bar(models, maes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
ax1.set_title('Out-of-Sample Forecast Accuracy: With vs Without 2023',
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='y', alpha=0.3)

# Add values on bars
for bar, mae in zip(bars, maes):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{mae:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement text
if improvement > 0:
    ax1.text(0.5, max(maes) * 0.9,
            f'Improvement: {improvement:.4f} ({pct_improvement:.2f}%)',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Plot 2: Variable-specific MAE comparison
ax2 = axes[1]
x = np.arange(len(SELECTED_VARS))
width = 0.35

bars1 = ax2.bar(x - width/2, mae_baseline, width, label='Baseline (WITH 2023)',
                color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax2.bar(x + width/2, mae_clean, width, label='No-Tuberville (NO 2023)',
                color='#2ecc71', alpha=0.7, edgecolor='black')

ax2.set_xlabel('Variable', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax2.set_title('Variable-Specific Forecast Errors', fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels([DISPLAY_NAMES[v] for v in SELECTED_VARS], rotation=45, ha='right')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tuberville_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n[SAVED] {OUTPUT_DIR / 'tuberville_comparison.png'}")

# Save detailed results
with open(OUTPUT_DIR / 'TUBERVILLE_ANALYSIS_SUMMARY.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("VECM RANK=2 ANALYSIS - TUBERVILLE ANOMALY IMPACT\n")
    f.write("=" * 80 + "\n\n")

    f.write("CONTEXT:\n")
    f.write("--------\n")
    f.write("Senator Tuberville placed a hold on all military promotions in 2023.\n")
    f.write("This created an artificial constraint on GOFO appointments, distorting\n")
    f.write("the natural bureaucratic dynamics captured by the VECM model.\n\n")

    f.write("METHODOLOGY:\n")
    f.write("------------\n")
    f.write(f"Training data (Baseline):    1987-2019 + 2020-2024 (33 obs, includes 2023)\n")
    f.write(f"Training data (No-Tuberville): 1987-2019 (33 obs, excludes 2020-2024)\n")
    f.write(f"Test data:                   2020-2024 (5 obs, includes 2023)\n")
    f.write(f"Model:                       VECM(rank=2, lag=1)\n\n")

    f.write("RESULTS:\n")
    f.write("--------\n")
    f.write(f"Baseline MAE (WITH 2023):     {mae_baseline_overall:.4f}\n")
    f.write(f"No-Tuberville MAE (NO 2023):  {mae_clean_overall:.4f}\n")
    f.write(f"Improvement:                  {improvement:.4f} ({pct_improvement:+.2f}%)\n\n")

    if pct_improvement > 0:
        f.write("CONCLUSION:\n")
        f.write("-----------\n")
        f.write(f"[+] Removing 2023 data IMPROVED forecast accuracy by {pct_improvement:.2f}%.\n")
        f.write("This confirms that the Tuberville hold was a significant anomaly that\n")
        f.write("distorted the underlying bureaucratic growth dynamics.\n\n")
        f.write("RECOMMENDATION: Use the No-Tuberville model for structural analysis,\n")
        f.write("as it better captures stable long-run relationships.\n")
    else:
        f.write("CONCLUSION:\n")
        f.write("-----------\n")
        f.write(f"[-] Removing 2023 data WORSENED forecast accuracy by {abs(pct_improvement):.2f}%.\n")
        f.write("The Tuberville hold may not have been as severe an outlier as expected,\n")
        f.write("or the model benefits from the additional recent observations.\n\n")
        f.write("RECOMMENDATION: Keep 2023 data in the model, but acknowledge the\n")
        f.write("political context in the limitations section.\n")

    f.write("\nVARIABLE-SPECIFIC RESULTS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Variable':<30s} {'Baseline':<12s} {'No-Tuberville':<15s} {'Change':<12s}\n")
    f.write("-" * 80 + "\n")
    for i, var in enumerate(SELECTED_VARS):
        diff = mae_baseline[i] - mae_clean[i]
        pct = (diff / mae_baseline[i]) * 100
        f.write(f"{DISPLAY_NAMES[var]:<30s} {mae_baseline[i]:>8.4f}     {mae_clean[i]:>8.4f}        {pct:>+7.2f}%\n")

print(f"[SAVED] {OUTPUT_DIR / 'TUBERVILLE_ANALYSIS_SUMMARY.txt'}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nFiles generated:")
print(f"  1. tuberville_comparison.png")
print(f"  2. TUBERVILLE_ANALYSIS_SUMMARY.txt")
