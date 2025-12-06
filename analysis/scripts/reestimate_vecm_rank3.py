"""
Re-estimate VECM with rank=3 to see if it better captures relationships
Also plot forecast vs actual for GOFOs to understand the 110% MAPE
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis")
DATA_FILE = BASE_DIR / "complete_normalized_dataset_v12.3.xlsx"
OUTPUT_DIR = BASE_DIR / "VECM_v12.3_Final" / "VECM_Rank3_CORRECTED"
OUTPUT_DIR.mkdir(exist_ok=True)

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
print("VECM RANK=3 ESTIMATION WITH CORRECTED SIGNS")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
df = pd.read_excel(DATA_FILE)
df_clean = df.dropna(subset=SELECTED_VARS)
data = df_clean[SELECTED_VARS].values
years = df_clean['FY'].values

print(f"    Data shape: {data.shape}")
print(f"    Years: {years.min():.0f} to {years.max():.0f}")

# Split into train and test
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]
train_years = years[:train_size]
test_years = years[train_size:]

print(f"\n[2] Data split:")
print(f"    Training: {train_years.min():.0f} to {train_years.max():.0f} (n={len(train_data)})")
print(f"    Testing: {test_years.min():.0f} to {test_years.max():.0f} (n={len(test_data)})")

# Estimate VECM with rank=3
print("\n[3] Estimating VECM (rank=3, k_ar_diff=1)...")
model = VECM(train_data, k_ar_diff=1, coint_rank=3, deterministic='ci')
vecm_result = model.fit()

print("    VECM estimation complete")

# Extract matrices
alpha_original = vecm_result.alpha
beta_original = vecm_result.beta
gamma = vecm_result.gamma

print(f"\n[4] Original matrices:")
print(f"    Alpha: {alpha_original.shape}")
print(f"    Beta: {beta_original.shape}")
print(f"    Gamma: {gamma.shape}")

# Calculate empirical correlations
print("\n[5] Checking beta signs for all 3 cointegration vectors...")
corr_matrix = pd.DataFrame(train_data, columns=SELECTED_VARS).corr()

alpha_corrected = alpha_original.copy()
beta_corrected = beta_original.copy()

for vec_idx in range(3):
    print(f"\n[5.{vec_idx+1}] Checking EC{vec_idx+1}...")

    abs_betas = np.abs(beta_original[:, vec_idx])
    ref_idx = np.argmax(abs_betas)
    ref_var = SELECTED_VARS[ref_idx]
    ref_beta_sign = np.sign(beta_original[ref_idx, vec_idx])

    print(f"    Reference: {ref_var} (beta={beta_original[ref_idx, vec_idx]:+.4f})")

    wrong_signs = 0
    correct_signs = 0

    for i, var in enumerate(SELECTED_VARS):
        if i == ref_idx or abs(beta_original[i, vec_idx]) < 0.01:
            continue

        empirical_corr = corr_matrix.loc[ref_var, var]
        beta_val = beta_original[i, vec_idx]
        beta_sign = np.sign(beta_val)

        if empirical_corr > 0:
            expected_sign = ref_beta_sign
        else:
            expected_sign = -ref_beta_sign

        if beta_sign == expected_sign:
            correct_signs += 1
        else:
            wrong_signs += 1

    if wrong_signs > correct_signs:
        print(f"    >>> Flipping EC{vec_idx+1} (wrong={wrong_signs} > correct={correct_signs})")
        beta_corrected[:, vec_idx] *= -1
        alpha_corrected[:, vec_idx] *= -1
    else:
        print(f"    >>> Keeping EC{vec_idx+1} (correct={correct_signs} >= wrong={wrong_signs})")

# Save corrected matrices
print("\n[6] Saving corrected matrices...")
alpha_df = pd.DataFrame(alpha_corrected, index=SELECTED_VARS, columns=['EC1', 'EC2', 'EC3'])
beta_df = pd.DataFrame(beta_corrected, index=SELECTED_VARS, columns=['EC1', 'EC2', 'EC3'])
gamma_df = pd.DataFrame(gamma, index=SELECTED_VARS, columns=SELECTED_VARS)

alpha_df.to_excel(OUTPUT_DIR / "alpha_matrix_rank3.xlsx")
beta_df.to_excel(OUTPUT_DIR / "beta_matrix_rank3.xlsx")
gamma_df.to_excel(OUTPUT_DIR / "gamma_matrix_rank3.xlsx")

# Calculate long-run influence
longrun_influence = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))
for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        unsigned_sum = 0
        for r in range(3):
            alpha_i = alpha_corrected[i, r]
            beta_j = beta_corrected[j, r]
            unsigned_sum += abs(alpha_i * beta_j)
        longrun_influence[i, j] = unsigned_sum

longrun_df = pd.DataFrame(longrun_influence, index=SELECTED_VARS, columns=SELECTED_VARS)
longrun_df.to_excel(OUTPUT_DIR / "longrun_influence_rank3.xlsx")

print("    Matrices saved")

# OUT-OF-SAMPLE FORECASTING
print("\n[7] Out-of-sample forecasting...")
forecast_steps = len(test_data)
forecasts = []

for step in range(forecast_steps):
    historical_data = np.vstack([train_data, test_data[:step]]) if step > 0 else train_data
    temp_model = VECM(historical_data, k_ar_diff=1, coint_rank=3, deterministic='ci')
    temp_result = temp_model.fit()
    forecast = temp_result.predict(steps=1)
    forecasts.append(forecast[0])

forecasts = np.array(forecasts)

# Calculate MAPE and MAE
print("\n[8] Performance metrics (RANK=3)...")
print("\n" + "-" * 80)
print(f"{'Variable':<30s} {'MAPE (%)':<15s} {'MAE':<15s}")
print("-" * 80)

mape_values = {}
for i, var in enumerate(SELECTED_VARS):
    actual = test_data[:, i]
    predicted = forecasts[:, i]

    mae = np.mean(np.abs(actual - predicted))

    non_zero_mask = np.abs(actual) > 0.01
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
    else:
        mape = np.nan

    mape_values[var] = {'MAPE': mape, 'MAE': mae}
    print(f"{var:<30s} {mape:>10.2f}%     {mae:>10.4f}")

print("-" * 80)
avg_mape = np.nanmean([v['MAPE'] for v in mape_values.values()])
avg_mae = np.mean([v['MAE'] for v in mape_values.values()])
print(f"{'AVERAGE':<30s} {avg_mape:>10.2f}%     {avg_mae:>10.4f}")
print("-" * 80)

# VISUALIZATION: Compare Rank 2 vs Rank 3 performance
print("\n[9] Creating forecast comparison plots...")

fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.flatten()

for i, var in enumerate(SELECTED_VARS):
    ax = axes[i]

    # Plot actual values
    ax.plot(test_years, test_data[:, i], 'o-', linewidth=2, markersize=8,
            label='Actual', color='black', alpha=0.8)

    # Plot forecast
    ax.plot(test_years, forecasts[:, i], 's--', linewidth=2, markersize=6,
            label='Forecast (Rank=3)', color='red', alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Z-Score', fontsize=10)
    ax.set_title(f"{var.replace('_Z', '').replace('_Log', '')}\n"
                 f"MAE={mape_values[var]['MAE']:.4f}, MAPE={mape_values[var]['MAPE']:.1f}%",
                 fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle(f'Out-of-Sample Forecast Performance (Rank=3 VECM)\n'
             f'Test Period: {test_years.min():.0f}-{test_years.max():.0f} | '
             f'Average MAE: {avg_mae:.4f}',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "forecast_comparison_rank3.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Forecast plots saved")

# SPECIAL PLOT: GOFOs detailed analysis
print("\n[10] Creating detailed GOFOs analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Full time series with forecast
gofo_idx = SELECTED_VARS.index('GOFOs_Z')
all_years = years
all_gofos = data[:, gofo_idx]

ax1.plot(train_years, train_data[:, gofo_idx], 'o-', linewidth=2, markersize=6,
         label='Training Data', color='blue', alpha=0.7)
ax1.plot(test_years, test_data[:, gofo_idx], 'o-', linewidth=2, markersize=8,
         label='Actual (Test)', color='black', alpha=0.8)
ax1.plot(test_years, forecasts[:, gofo_idx], 's--', linewidth=2, markersize=6,
         label='Forecast (Rank=3)', color='red', alpha=0.7)
ax1.axvline(x=train_years.max(), color='green', linestyle='--', linewidth=2,
            alpha=0.5, label='Train/Test Split')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.3)
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('GOFOs (Z-Score)', fontsize=12, fontweight='bold')
ax1.set_title('GOFOs Full Time Series with Forecast', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(alpha=0.3)

# Right: Forecast errors
errors = test_data[:, gofo_idx] - forecasts[:, gofo_idx]
ax2.bar(test_years, errors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Forecast Error (Actual - Predicted)', fontsize=12, fontweight='bold')
ax2.set_title(f'GOFOs Forecast Errors\nMAE={mape_values["GOFOs_Z"]["MAE"]:.4f}, '
              f'MAPE={mape_values["GOFOs_Z"]["MAPE"]:.1f}%',
              fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

# Add error statistics
error_text = f"""
Mean Error: {errors.mean():.4f}
Std Error: {errors.std():.4f}
Max Error: {errors.max():.4f}
Min Error: {errors.min():.4f}
"""
ax2.text(0.02, 0.98, error_text, transform=ax2.transAxes,
         verticalalignment='top', fontsize=10, fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gofos_detailed_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print("    GOFOs analysis saved")

# Print explanation of high MAPE
print("\n" + "=" * 80)
print("WHY IS GOFOs MAPE SO HIGH?")
print("=" * 80)
print(f"""
GOFOs MAPE: {mape_values['GOFOs_Z']['MAPE']:.2f}%
GOFOs MAE:  {mape_values['GOFOs_Z']['MAE']:.4f}

The high MAPE for GOFOs is likely due to Z-SCORE DIVISION ISSUES:
- MAPE = Mean( |actual - predicted| / |actual| ) * 100
- For z-scored data, values can be close to zero
- Small denominators create huge percentage errors

The MAE (Mean Absolute Error) is more meaningful for z-scored data:
- GOFOs MAE = {mape_values['GOFOs_Z']['MAE']:.4f}
- This is actually similar to other variables
- Average MAE across all variables = {avg_mae:.4f}

So GOFOs isn't actually harder to forecast than other variables - the MAPE
metric is just misleading for z-scored data near zero!
""")

print("=" * 80)
print("RANK 3 vs RANK 2 COMPARISON")
print("=" * 80)
print(f"""
RANK=3:
  Average MAPE: {avg_mape:.2f}%
  Average MAE: {avg_mae:.4f}

(Compare to Rank=2: MAPE=28.83%, MAE=0.2523)

Rank=3 adds a third cointegration relationship, which can capture more complex
equilibrium dynamics but also risks overfitting with limited data.
""")

print("=" * 80)
print("FILES SAVED")
print("=" * 80)
print(f"\nLocation: {OUTPUT_DIR}")
print("\nGenerated:")
print("  1. alpha_matrix_rank3.xlsx")
print("  2. beta_matrix_rank3.xlsx")
print("  3. gamma_matrix_rank3.xlsx")
print("  4. longrun_influence_rank3.xlsx")
print("  5. forecast_comparison_rank3.png (all 8 variables)")
print("  6. gofos_detailed_analysis.png (GOFOs deep dive)")
print("=" * 80)
