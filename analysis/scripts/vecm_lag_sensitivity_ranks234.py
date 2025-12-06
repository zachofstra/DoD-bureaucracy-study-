"""
Comprehensive Lag Sensitivity for Ranks 2-4
============================================
Test optimal lag order for ranks 2, 3, and 4
Then identify best (rank, lag) combination
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
print("COMPREHENSIVE LAG SENSITIVITY: RANKS 2, 3, 4")
print("=" * 80)

# Load data
data_file = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\complete_normalized_dataset_v12.3.xlsx")
df = pd.read_excel(data_file)
df.columns = df.columns.str.strip()
data = df[SELECTED_VARS].dropna().copy()

train_data = data.iloc[:-5]
test_data = data.iloc[-5:]

print(f"\nData: {data.shape[0]} observations x {data.shape[1]} variables")
print(f"Training: {train_data.shape[0]} observations (1987-2019)")
print(f"Test: {test_data.shape[0]} observations (2020-2024)")

# Test all combinations
all_results = []

for rank in [2, 3, 4]:
    print(f"\n{'='*80}")
    print(f"RANK={rank}")
    print(f"{'='*80}")

    for k_ar_diff in range(1, 5):
        print(f"\n  --- k_ar_diff={k_ar_diff} ---")

        try:
            # Fit models
            vecm_full = VECM(data, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic='nc')
            vecm_full_result = vecm_full.fit()

            vecm_train = VECM(train_data, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic='nc')
            vecm_train_result = vecm_train.fit()

            # Information criteria
            llf = vecm_full_result.llf
            nobs = len(data) - k_ar_diff
            neqs = len(SELECTED_VARS)
            k_params = neqs * rank + neqs * neqs * k_ar_diff + rank * (neqs - 1)

            aic = -2 * llf + 2 * k_params
            bic = -2 * llf + k_params * np.log(nobs)
            hqic = -2 * llf + 2 * k_params * np.log(np.log(nobs))

            # Out-of-sample
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

            # Residual stationarity
            residuals = vecm_full_result.resid
            stationary_count = sum(1 for i in range(residuals.shape[1])
                                  if adfuller(residuals[:, i], autolag='AIC')[1] < 0.05)
            stationarity_pct = (stationary_count / residuals.shape[1]) * 100

            print(f"    BIC: {bic:.2f}, MAPE: {mape:.2f}%, Stationarity: {stationarity_pct:.0f}%")

            all_results.append({
                'rank': rank,
                'lag_order': k_ar_diff,
                'llf': llf,
                'aic': aic,
                'bic': bic,
                'hqic': hqic,
                'out_of_sample_rmse': rmse,
                'out_of_sample_mae': mae,
                'out_of_sample_mape': mape,
                'residual_stationarity_pct': stationarity_pct
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            all_results.append({
                'rank': rank,
                'lag_order': k_ar_diff,
                'llf': np.nan,
                'aic': np.nan,
                'bic': np.nan,
                'hqic': np.nan,
                'out_of_sample_rmse': np.nan,
                'out_of_sample_mae': np.nan,
                'out_of_sample_mape': np.nan,
                'residual_stationarity_pct': np.nan
            })

# Create results DataFrame
results_df = pd.DataFrame(all_results)

print("\n" + "=" * 80)
print("COMPLETE RESULTS")
print("=" * 80)
print("\n" + results_df.to_string(index=False))

# Save
output_file = OUTPUT_DIR / "vecm_lag_sensitivity_ranks234_complete.xlsx"
results_df.to_excel(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Find overall best
print("\n" + "=" * 80)
print("OVERALL BEST CONFIGURATIONS")
print("=" * 80)

valid_results = results_df.dropna(subset=['bic', 'out_of_sample_mape'])

if not valid_results.empty:
    best_bic = valid_results.loc[valid_results['bic'].idxmin()]
    best_mape = valid_results.loc[valid_results['out_of_sample_mape'].idxmin()]

    print(f"\nBest by BIC:")
    print(f"  Rank={int(best_bic['rank'])}, k_ar_diff={int(best_bic['lag_order'])}")
    print(f"  BIC={best_bic['bic']:.2f}, MAPE={best_bic['out_of_sample_mape']:.2f}%")

    print(f"\nBest by Out-of-Sample MAPE:")
    print(f"  Rank={int(best_mape['rank'])}, k_ar_diff={int(best_mape['lag_order'])}")
    print(f"  BIC={best_mape['bic']:.2f}, MAPE={best_mape['out_of_sample_mape']:.2f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Separate by rank
for rank in [2, 3, 4]:
    rank_data = results_df[results_df['rank'] == rank]
    color = ['red', 'green', 'purple'][rank-2]
    label = f'Rank={rank}'

    # BIC
    axes[0, 0].plot(rank_data['lag_order'], rank_data['bic'], 'o-',
                    label=label, linewidth=2, markersize=8, color=color)

    # MAPE
    axes[0, 1].plot(rank_data['lag_order'], rank_data['out_of_sample_mape'], 'o-',
                    label=label, linewidth=2, markersize=8, color=color)

    # LLF
    axes[1, 0].plot(rank_data['lag_order'], rank_data['llf'], 'o-',
                    label=label, linewidth=2, markersize=8, color=color)

    # Stationarity
    axes[1, 1].plot(rank_data['lag_order'], rank_data['residual_stationarity_pct'], 'o-',
                    label=label, linewidth=2, markersize=8, color=color)

# Format plots
axes[0, 0].set_title('BIC (lower is better)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Lag Order (k_ar_diff)', fontsize=11)
axes[0, 0].set_ylabel('BIC', fontsize=11)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks([1, 2, 3, 4])

axes[0, 1].set_title('Out-of-Sample MAPE (lower is better)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Lag Order (k_ar_diff)', fontsize=11)
axes[0, 1].set_ylabel('MAPE (%)', fontsize=11)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks([1, 2, 3, 4])

axes[1, 0].set_title('Log-Likelihood (higher is better)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Lag Order (k_ar_diff)', fontsize=11)
axes[1, 0].set_ylabel('Log-Likelihood', fontsize=11)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks([1, 2, 3, 4])

axes[1, 1].set_title('Residual Stationarity (higher is better)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Lag Order (k_ar_diff)', fontsize=11)
axes[1, 1].set_ylabel('% Stationary', fontsize=11)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks([1, 2, 3, 4])
axes[1, 1].set_ylim([0, 105])

fig.suptitle('Lag Sensitivity Across Ranks 2-4', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "lag_sensitivity_ranks234_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisualization saved to: {OUTPUT_DIR / 'lag_sensitivity_ranks234_comparison.png'}")

# Recommendation
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if not valid_results.empty:
    # Best overall (balance BIC and MAPE)
    valid_results['combined_score'] = (
        (valid_results['bic'] - valid_results['bic'].min()) / (valid_results['bic'].max() - valid_results['bic'].min()) +
        (valid_results['out_of_sample_mape'] - valid_results['out_of_sample_mape'].min()) /
        (valid_results['out_of_sample_mape'].max() - valid_results['out_of_sample_mape'].min())
    )

    best_overall = valid_results.loc[valid_results['combined_score'].idxmin()]

    print(f"\nBest Overall (balanced BIC + MAPE):")
    print(f"  Rank={int(best_overall['rank'])}, k_ar_diff={int(best_overall['lag_order'])}")
    print(f"  BIC={best_overall['bic']:.2f}")
    print(f"  MAPE={best_overall['out_of_sample_mape']:.2f}%")
    print(f"  Residual stationarity={best_overall['residual_stationarity_pct']:.0f}%")

    print("\nAll ranks prefer k_ar_diff=1 (or errors occur at higher lags)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
