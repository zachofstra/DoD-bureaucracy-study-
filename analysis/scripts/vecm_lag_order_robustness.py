"""
VECM Lag Order Robustness Check
================================
Test if k_ar_diff=1 is optimal or if another lag order performs better.

For VECM with k_ar_diff=p, we have:
- p lags of DIFFERENCED variables (VAR in differences)
- Plus error correction term from levels

k_ar_diff=1 means VAR(2) in levels (1 lag of differences + 1 level lag from EC term)
k_ar_diff=2 means VAR(3) in levels (2 lags of differences + 1 level lag)
etc.

We test k_ar_diff from 1 to 4 and compare:
- Out-of-sample prediction accuracy
- Information criteria (AIC, BIC)
- Residual diagnostics
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
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

print("=" * 80)
print("VECM LAG ORDER ROBUSTNESS CHECK (RANK=1)")
print("=" * 80)
print("\nTesting lag orders k_ar_diff = 1, 2, 3, 4")
print("  k_ar_diff=1: VAR(2) in levels")
print("  k_ar_diff=2: VAR(3) in levels")
print("  k_ar_diff=3: VAR(4) in levels")
print("  k_ar_diff=4: VAR(5) in levels")
print("\n" + "=" * 80)

# Load data
print("\n[1] LOADING DATA...")
data_file = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\complete_normalized_dataset_v12.3.xlsx")
df = pd.read_excel(data_file)
df.columns = df.columns.str.strip()
data = df[SELECTED_VARS].dropna().copy()

print(f"    Data shape: {data.shape[0]} observations x {data.shape[1]} variables")

# Split data
train_data = data.iloc[:-5]
test_data = data.iloc[-5:]
print(f"    Training: {train_data.shape[0]} observations (1987-2019)")
print(f"    Test: {test_data.shape[0]} observations (2020-2024)")

# Test lag orders
results = {
    'lag_order': [],
    'llf': [],
    'aic': [],
    'bic': [],
    'hqic': [],
    'out_of_sample_rmse': [],
    'out_of_sample_mae': [],
    'out_of_sample_mape': [],
    'residual_stationarity_pct': [],
    'observations_used': []
}

print("\n[2] TESTING LAG ORDERS...")
print("=" * 80)

for k_ar_diff in range(1, 5):
    print(f"\n--- k_ar_diff={k_ar_diff} (VAR({k_ar_diff+1}) in levels) ---")

    try:
        # Fit on full data
        vecm_full = VECM(data, k_ar_diff=k_ar_diff, coint_rank=1, deterministic='nc')
        vecm_full_result = vecm_full.fit()

        # Fit on training data
        vecm_train = VECM(train_data, k_ar_diff=k_ar_diff, coint_rank=1, deterministic='nc')
        vecm_train_result = vecm_train.fit()

        # Log-likelihood and information criteria
        llf = vecm_full_result.llf
        nobs = len(data) - k_ar_diff  # Effective observations after lag loss
        neqs = len(SELECTED_VARS)

        # Count parameters (approximate)
        # alpha: neqs * 1 (rank=1)
        # beta: neqs * 1 (rank=1, but normalized)
        # gamma: neqs * neqs * k_ar_diff
        k_params = neqs * 1 + neqs * neqs * k_ar_diff + (neqs - 1)  # Approximate

        aic = -2 * llf + 2 * k_params
        bic = -2 * llf + k_params * np.log(nobs)
        hqic = -2 * llf + 2 * k_params * np.log(np.log(nobs))

        print(f"  LLF:  {llf:.2f}")
        print(f"  AIC:  {aic:.2f}")
        print(f"  BIC:  {bic:.2f}")
        print(f"  HQIC: {hqic:.2f}")
        print(f"  Obs used: {nobs}")

        # Out-of-sample forecast
        forecast = vecm_train_result.predict(steps=5)

        # Calculate errors
        errors = test_data.values - forecast
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))

        mape_values = []
        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[1]):
                actual = test_data.values[i, j]
                if abs(actual) > 0.01:
                    mape_values.append(abs(errors[i, j] / actual) * 100)
        mape = np.mean(mape_values) if mape_values else np.nan

        print(f"  Out-of-sample RMSE: {rmse:.4f}")
        print(f"  Out-of-sample MAE:  {mae:.4f}")
        print(f"  Out-of-sample MAPE: {mape:.2f}%")

        # Residual stationarity
        residuals = vecm_full_result.resid
        stationary_count = 0
        for var_idx in range(residuals.shape[1]):
            adf_result = adfuller(residuals[:, var_idx], autolag='AIC')
            if adf_result[1] < 0.05:
                stationary_count += 1

        stationarity_pct = (stationary_count / residuals.shape[1]) * 100
        print(f"  Residual stationarity: {stationary_count}/{residuals.shape[1]} ({stationarity_pct:.1f}%)")

        # Store results
        results['lag_order'].append(k_ar_diff)
        results['llf'].append(llf)
        results['aic'].append(aic)
        results['bic'].append(bic)
        results['hqic'].append(hqic)
        results['out_of_sample_rmse'].append(rmse)
        results['out_of_sample_mae'].append(mae)
        results['out_of_sample_mape'].append(mape)
        results['residual_stationarity_pct'].append(stationarity_pct)
        results['observations_used'].append(nobs)

    except Exception as e:
        print(f"  ERROR: {e}")
        results['lag_order'].append(k_ar_diff)
        results['llf'].append(np.nan)
        results['aic'].append(np.nan)
        results['bic'].append(np.nan)
        results['hqic'].append(np.nan)
        results['out_of_sample_rmse'].append(np.nan)
        results['out_of_sample_mae'].append(np.nan)
        results['out_of_sample_mape'].append(np.nan)
        results['residual_stationarity_pct'].append(np.nan)
        results['observations_used'].append(np.nan)

# Create results DataFrame
results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)
print("\n" + results_df.to_string(index=False))

# Save results
output_file = OUTPUT_DIR / "vecm_lag_order_robustness_rank1.xlsx"
results_df.to_excel(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Identify best lag order
print("\n" + "=" * 80)
print("BEST LAG ORDER BY EACH CRITERION")
print("=" * 80)

try:
    best_aic = results_df.loc[results_df['aic'].idxmin()]
    print(f"\nAIC (lower is better):           k_ar_diff={int(best_aic['lag_order'])} (AIC={best_aic['aic']:.2f})")
except:
    print(f"\nAIC (lower is better):           No valid results")

try:
    best_bic = results_df.loc[results_df['bic'].idxmin()]
    print(f"BIC (lower is better):           k_ar_diff={int(best_bic['lag_order'])} (BIC={best_bic['bic']:.2f})")
except:
    print(f"BIC (lower is better):           No valid results")

try:
    best_hqic = results_df.loc[results_df['hqic'].idxmin()]
    print(f"HQIC (lower is better):          k_ar_diff={int(best_hqic['lag_order'])} (HQIC={best_hqic['hqic']:.2f})")
except:
    print(f"HQIC (lower is better):          No valid results")

try:
    best_rmse = results_df.loc[results_df['out_of_sample_rmse'].idxmin()]
    print(f"Out-of-sample RMSE (lower):      k_ar_diff={int(best_rmse['lag_order'])} (RMSE={best_rmse['out_of_sample_rmse']:.4f})")
except:
    print(f"Out-of-sample RMSE (lower):      No valid results")

try:
    best_mape = results_df.loc[results_df['out_of_sample_mape'].idxmin()]
    print(f"Out-of-sample MAPE (lower):      k_ar_diff={int(best_mape['lag_order'])} (MAPE={best_mape['out_of_sample_mape']:.2f}%)")
except:
    print(f"Out-of-sample MAPE (lower):      No valid results")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Information Criteria
ax1 = axes[0, 0]
ax1.plot(results_df['lag_order'], results_df['aic'], 'o-', label='AIC', linewidth=2, markersize=8)
ax1.plot(results_df['lag_order'], results_df['bic'], 's-', label='BIC', linewidth=2, markersize=8)
ax1.plot(results_df['lag_order'], results_df['hqic'], '^-', label='HQIC', linewidth=2, markersize=8)
ax1.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current (k_ar_diff=1)')
ax1.set_xlabel('Lag Order (k_ar_diff)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Information Criterion', fontsize=11, fontweight='bold')
ax1.set_title('Information Criteria (lower is better)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks([1, 2, 3, 4])

# Plot 2: Out-of-sample MAPE
ax2 = axes[0, 1]
ax2.plot(results_df['lag_order'], results_df['out_of_sample_mape'], 'o-', color='navy', linewidth=2, markersize=8)
ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current (k_ar_diff=1)')
ax2.set_xlabel('Lag Order (k_ar_diff)', fontsize=11, fontweight='bold')
ax2.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
ax2.set_title('Out-of-Sample Prediction Error (lower is better)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks([1, 2, 3, 4])

# Plot 3: Log-Likelihood
ax3 = axes[1, 0]
ax3.plot(results_df['lag_order'], results_df['llf'], 'o-', color='darkgreen', linewidth=2, markersize=8)
ax3.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current (k_ar_diff=1)')
ax3.set_xlabel('Lag Order (k_ar_diff)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Log-Likelihood', fontsize=11, fontweight='bold')
ax3.set_title('Log-Likelihood (higher is better)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks([1, 2, 3, 4])

# Plot 4: Residual Stationarity
ax4 = axes[1, 1]
ax4.bar(results_df['lag_order'], results_df['residual_stationarity_pct'], color='teal', alpha=0.7, edgecolor='black')
ax4.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current (k_ar_diff=1)')
ax4.set_xlabel('Lag Order (k_ar_diff)', fontsize=11, fontweight='bold')
ax4.set_ylabel('% Variables Stationary', fontsize=11, fontweight='bold')
ax4.set_title('Residual Stationarity (higher is better)', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 105])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xticks([1, 2, 3, 4])

fig.suptitle('VECM Lag Order Selection (Rank=1): Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_lag_order_robustness_rank1.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nComparison plot saved to: {OUTPUT_DIR / 'vecm_lag_order_robustness_rank1.png'}")

# Recommendation
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if not results_df.empty:
    # Find most common winner
    criteria_winners = []

    if not results_df['aic'].isna().all():
        criteria_winners.append(results_df.loc[results_df['aic'].idxmin(), 'lag_order'])
    if not results_df['bic'].isna().all():
        criteria_winners.append(results_df.loc[results_df['bic'].idxmin(), 'lag_order'])
    if not results_df['out_of_sample_mape'].isna().all():
        criteria_winners.append(results_df.loc[results_df['out_of_sample_mape'].idxmin(), 'lag_order'])

    if criteria_winners:
        from collections import Counter
        winner_counts = Counter(criteria_winners)
        best_lag = winner_counts.most_common(1)[0][0]

        if best_lag == 1:
            print("\nCurrent lag order k_ar_diff=1 is OPTIMAL. No change needed.")
        else:
            print(f"\nLag order k_ar_diff={int(best_lag)} appears to perform better than current k_ar_diff=1.")
            print(f"Consider re-estimating with k_ar_diff={int(best_lag)}.")
    else:
        print("\nInsufficient data to make recommendation.")

print("\n" + "=" * 80)
print("LAG ORDER ROBUSTNESS CHECK COMPLETE")
print("=" * 80)
