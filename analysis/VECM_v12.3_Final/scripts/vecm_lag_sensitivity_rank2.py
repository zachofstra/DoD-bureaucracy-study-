"""
Lag Order Sensitivity for Rank=2
==================================
Test if rank=2 prefers different lag order than rank=1

Rank=1 preferred k_ar_diff=1 (BIC, out-of-sample prediction)
Does rank=2 also prefer k_ar_diff=1, or does having 2 cointegration
vectors change the optimal short-run lag structure?
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
print("LAG ORDER SENSITIVITY FOR RANK=2")
print("=" * 80)
print("\nComparing k_ar_diff = 1, 2, 3, 4 for rank=2")
print("(Previously found k_ar_diff=1 optimal for rank=1)")
print("\n" + "=" * 80)

# Load data
data_file = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\complete_normalized_dataset_v12.3.xlsx")
df = pd.read_excel(data_file)
df.columns = df.columns.str.strip()
data = df[SELECTED_VARS].dropna().copy()

train_data = data.iloc[:-5]
test_data = data.iloc[-5:]

print(f"\nData: {data.shape[0]} observations")
print(f"Training: {train_data.shape[0]} observations (1987-2019)")
print(f"Test: {test_data.shape[0]} observations (2020-2024)")

# Test lag orders for rank=2
results_rank2 = {
    'lag_order': [],
    'llf': [],
    'aic': [],
    'bic': [],
    'hqic': [],
    'out_of_sample_rmse': [],
    'out_of_sample_mae': [],
    'out_of_sample_mape': [],
    'residual_stationarity_pct': []
}

print("\n" + "=" * 80)
print("TESTING LAG ORDERS FOR RANK=2")
print("=" * 80)

for k_ar_diff in range(1, 5):
    print(f"\n--- k_ar_diff={k_ar_diff} ---")

    try:
        # Fit on full data
        vecm_full = VECM(data, k_ar_diff=k_ar_diff, coint_rank=2, deterministic='nc')
        vecm_full_result = vecm_full.fit()

        # Fit on training data
        vecm_train = VECM(train_data, k_ar_diff=k_ar_diff, coint_rank=2, deterministic='nc')
        vecm_train_result = vecm_train.fit()

        # Information criteria
        llf = vecm_full_result.llf
        nobs = len(data) - k_ar_diff
        neqs = len(SELECTED_VARS)

        # Parameters: alpha (neqs * rank=2) + beta (neqs * rank=2) + gamma (neqs * neqs * k_ar_diff)
        k_params = neqs * 2 + neqs * neqs * k_ar_diff + 2 * (neqs - 1)

        aic = -2 * llf + 2 * k_params
        bic = -2 * llf + k_params * np.log(nobs)
        hqic = -2 * llf + 2 * k_params * np.log(np.log(nobs))

        print(f"  LLF:  {llf:.2f}")
        print(f"  AIC:  {aic:.2f}")
        print(f"  BIC:  {bic:.2f}")
        print(f"  HQIC: {hqic:.2f}")

        # Out-of-sample forecast
        forecast = vecm_train_result.predict(steps=5)
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
        stationary_count = sum(1 for i in range(residuals.shape[1])
                              if adfuller(residuals[:, i], autolag='AIC')[1] < 0.05)
        stationarity_pct = (stationary_count / residuals.shape[1]) * 100

        print(f"  Residual stationarity: {stationary_count}/{residuals.shape[1]} ({stationarity_pct:.1f}%)")

        # Store results
        results_rank2['lag_order'].append(k_ar_diff)
        results_rank2['llf'].append(llf)
        results_rank2['aic'].append(aic)
        results_rank2['bic'].append(bic)
        results_rank2['hqic'].append(hqic)
        results_rank2['out_of_sample_rmse'].append(rmse)
        results_rank2['out_of_sample_mae'].append(mae)
        results_rank2['out_of_sample_mape'].append(mape)
        results_rank2['residual_stationarity_pct'].append(stationarity_pct)

    except Exception as e:
        print(f"  ERROR: {e}")
        results_rank2['lag_order'].append(k_ar_diff)
        for key in ['llf', 'aic', 'bic', 'hqic', 'out_of_sample_rmse',
                    'out_of_sample_mae', 'out_of_sample_mape', 'residual_stationarity_pct']:
            results_rank2[key].append(np.nan)

# Create results DataFrame
results_df_rank2 = pd.DataFrame(results_rank2)

print("\n" + "=" * 80)
print("RANK=2 LAG ORDER COMPARISON")
print("=" * 80)
print("\n" + results_df_rank2.to_string(index=False))

# Load rank=1 results for comparison
try:
    results_df_rank1 = pd.read_excel(OUTPUT_DIR / "vecm_lag_order_robustness_rank1.xlsx")
    has_rank1 = True
except:
    has_rank1 = False
    print("\nNote: Rank=1 results not found, showing only rank=2")

# Save results
output_file = OUTPUT_DIR / "vecm_lag_order_robustness_rank2.xlsx"
results_df_rank2.to_excel(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Best by each criterion
print("\n" + "=" * 80)
print("BEST LAG ORDER FOR RANK=2")
print("=" * 80)

try:
    best_bic = results_df_rank2.loc[results_df_rank2['bic'].idxmin()]
    print(f"\nBIC (lower is better):     k_ar_diff={int(best_bic['lag_order'])} (BIC={best_bic['bic']:.2f})")
except:
    print(f"\nBIC: No valid results")

try:
    best_mape = results_df_rank2.loc[results_df_rank2['out_of_sample_mape'].idxmin()]
    print(f"Out-of-sample MAPE:        k_ar_diff={int(best_mape['lag_order'])} (MAPE={best_mape['out_of_sample_mape']:.2f}%)")
except:
    print(f"Out-of-sample MAPE: No valid results")

# Create comparison visualization
if has_rank1:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: BIC comparison
    ax1 = axes[0, 0]
    ax1.plot(results_df_rank1['lag_order'], results_df_rank1['bic'], 'o-',
             label='Rank=1', linewidth=2, markersize=8, color='blue')
    ax1.plot(results_df_rank2['lag_order'], results_df_rank2['bic'], 's-',
             label='Rank=2', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Lag Order (k_ar_diff)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('BIC', fontsize=11, fontweight='bold')
    ax1.set_title('BIC Comparison (lower is better)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1, 2, 3, 4])

    # Plot 2: MAPE comparison
    ax2 = axes[0, 1]
    ax2.plot(results_df_rank1['lag_order'], results_df_rank1['out_of_sample_mape'], 'o-',
             label='Rank=1', linewidth=2, markersize=8, color='blue')
    ax2.plot(results_df_rank2['lag_order'], results_df_rank2['out_of_sample_mape'], 's-',
             label='Rank=2', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Lag Order (k_ar_diff)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Out-of-Sample MAPE (lower is better)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([1, 2, 3, 4])

    # Plot 3: LLF comparison
    ax3 = axes[1, 0]
    ax3.plot(results_df_rank1['lag_order'], results_df_rank1['llf'], 'o-',
             label='Rank=1', linewidth=2, markersize=8, color='blue')
    ax3.plot(results_df_rank2['lag_order'], results_df_rank2['llf'], 's-',
             label='Rank=2', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Lag Order (k_ar_diff)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Log-Likelihood', fontsize=11, fontweight='bold')
    ax3.set_title('Log-Likelihood (higher is better)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks([1, 2, 3, 4])

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = "OPTIMAL LAG ORDER:\n\n"
    summary_text += "Rank=1:\n"
    try:
        r1_best_bic = results_df_rank1.loc[results_df_rank1['bic'].idxmin()]
        summary_text += f"  BIC:  k_ar_diff={int(r1_best_bic['lag_order'])} (BIC={r1_best_bic['bic']:.1f})\n"
        r1_best_mape = results_df_rank1.loc[results_df_rank1['out_of_sample_mape'].idxmin()]
        summary_text += f"  MAPE: k_ar_diff={int(r1_best_mape['lag_order'])} ({r1_best_mape['out_of_sample_mape']:.1f}%)\n"
    except:
        summary_text += "  No valid results\n"

    summary_text += "\nRank=2:\n"
    try:
        r2_best_bic = results_df_rank2.loc[results_df_rank2['bic'].idxmin()]
        summary_text += f"  BIC:  k_ar_diff={int(r2_best_bic['lag_order'])} (BIC={r2_best_bic['bic']:.1f})\n"
        r2_best_mape = results_df_rank2.loc[results_df_rank2['out_of_sample_mape'].idxmin()]
        summary_text += f"  MAPE: k_ar_diff={int(r2_best_mape['lag_order'])} ({r2_best_mape['out_of_sample_mape']:.1f}%)\n"
    except:
        summary_text += "  No valid results\n"

    ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Lag Order Sensitivity: Rank=1 vs Rank=2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lag_sensitivity_rank1_vs_rank2.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nComparison plot saved to: {OUTPUT_DIR / 'lag_sensitivity_rank1_vs_rank2.png'}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if not results_df_rank2.empty and not results_df_rank2['bic'].isna().all():
    best_lag_bic = int(results_df_rank2.loc[results_df_rank2['bic'].idxmin(), 'lag_order'])
    best_lag_mape = int(results_df_rank2.loc[results_df_rank2['out_of_sample_mape'].idxmin(), 'lag_order'])

    if best_lag_bic == 1 and best_lag_mape == 1:
        print("\nRank=2 also prefers k_ar_diff=1 (same as rank=1)")
        print("RECOMMENDATION: Use rank=2, k_ar_diff=1")
    else:
        print(f"\nRank=2 prefers different lag order:")
        print(f"  BIC optimal: k_ar_diff={best_lag_bic}")
        print(f"  MAPE optimal: k_ar_diff={best_lag_mape}")
        print(f"RECOMMENDATION: Investigate why rank changes optimal lag structure")

print("\n" + "=" * 80)
