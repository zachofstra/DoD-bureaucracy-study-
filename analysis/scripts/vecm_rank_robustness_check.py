"""
VECM Rank Robustness Check
==========================
Compare VECM performance across different cointegration ranks (1-8)

Evaluation criteria:
1. AIC/BIC (information criteria - lower is better)
2. Out-of-sample prediction error (2020-2024 forecast accuracy)
3. Residual diagnostics (stationarity, autocorrelation)
4. Trace statistics (cointegration strength)

This validates whether rank=6 is optimal or if another rank performs better.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
OUTPUT_DIR = BASE_DIR / "VECM_v12.3_Final_Executive_Summary"

# Selected variables (from Row 0 of johansen_tests_corrected.xlsx)
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
print("VECM RANK ROBUSTNESS CHECK")
print("=" * 80)
print(f"\nTesting cointegration ranks 1 through {len(SELECTED_VARS)}")
print(f"Variables: {len(SELECTED_VARS)}")
print(f"Observations: 38 (1987-2024)")
print(f"Out-of-sample: Last 5 years (2020-2024)")
print("\n" + "=" * 80)

# Load data
data_file = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\complete_normalized_dataset_v12.3.xlsx")
df = pd.read_excel(data_file)

# Ensure correct column names
df.columns = df.columns.str.strip()

# Select variables and remove any NaN
data = df[SELECTED_VARS].dropna().copy()

print(f"\nData loaded: {data.shape[0]} observations x {data.shape[1]} variables")
print(f"Date range: 1987-2024")
print(f"NaN values: {data.isna().sum().sum()}")

# Check for any infinite values
if np.isinf(data.values).any():
    print("WARNING: Infinite values detected! Removing...")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Split into training (1987-2019) and test (2020-2024)
train_data = data.iloc[:-5]  # 33 observations
test_data = data.iloc[-5:]   # 5 observations

print(f"\nTraining set: {train_data.shape[0]} observations (1987-2019)")
print(f"Test set: {test_data.shape[0]} observations (2020-2024)")

# Storage for results
results = {
    'rank': [],
    'aic': [],
    'bic': [],
    'hqic': [],
    'fpe': [],
    'llf': [],  # Log-likelihood
    'out_of_sample_rmse': [],
    'out_of_sample_mae': [],
    'out_of_sample_mape': [],
    'residual_stationarity_pct': [],
    'trace_statistic': []
}

# Run Johansen test on full data to get trace statistics
print("\n" + "=" * 80)
print("JOHANSEN TEST ON FULL DATA")
print("=" * 80)

try:
    joh_result = coint_johansen(data, det_order=0, k_ar_diff=1)
except np.linalg.LinAlgError:
    print("WARNING: SVD did not converge with det_order=0, trying det_order=-1 (no constant)")
    joh_result = coint_johansen(data, det_order=-1, k_ar_diff=1)

trace_stats = joh_result.trace_stat
trace_crit_95 = joh_result.trace_stat_crit_vals[:, 1]

print("\nTrace Statistics:")
print(f"{'Rank':<6} {'H0: r<=':<12} {'Trace Stat':<15} {'95% Crit':<15} {'Result':<10}")
print("-" * 60)
for r in range(len(SELECTED_VARS)):
    reject = "REJECT" if trace_stats[r] > trace_crit_95[r] else "Accept"
    print(f"{r:<6} {'r<='+str(r):<12} {trace_stats[r]:<15.2f} {trace_crit_95[r]:<15.2f} {reject:<10}")

# Estimate VECM for each rank
print("\n" + "=" * 80)
print("ESTIMATING VECM FOR EACH RANK")
print("=" * 80)

for rank in range(1, len(SELECTED_VARS) + 1):
    print(f"\n--- Rank {rank} ---")

    try:
        # Fit on full data for information criteria
        vecm_full = VECM(data, k_ar_diff=1, coint_rank=rank, deterministic='nc')
        vecm_full_result = vecm_full.fit()

        # Fit on training data for out-of-sample validation
        vecm_train = VECM(train_data, k_ar_diff=1, coint_rank=rank, deterministic='nc')
        vecm_train_result = vecm_train.fit()

        # Get log-likelihood
        llf = vecm_full_result.llf

        # Calculate information criteria manually
        nobs = len(data)
        neqs = len(SELECTED_VARS)

        # Number of parameters in VECM
        # alpha: neqs * rank
        # beta: neqs * rank (but beta is normalized)
        # gamma: neqs * neqs * 1 (one lag in differences)
        # Total free parameters approximately:
        k_params = neqs * rank + neqs * neqs + rank * (neqs - 1)  # Approximate

        aic = -2 * llf + 2 * k_params
        bic = -2 * llf + k_params * np.log(nobs)
        hqic = -2 * llf + 2 * k_params * np.log(np.log(nobs))
        fpe = np.exp(2 * llf / nobs) * ((nobs + k_params) / (nobs - k_params)) ** neqs

        print(f"LLF:  {llf:.2f}")
        print(f"AIC:  {aic:.2f}")
        print(f"BIC:  {bic:.2f}")
        print(f"HQIC: {hqic:.2f}")

        # Out-of-sample forecast
        forecast = vecm_train_result.predict(steps=5)

        # Calculate prediction errors
        errors = test_data.values - forecast
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))

        # MAPE (avoid division by zero)
        mape_values = []
        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[1]):
                actual = test_data.values[i, j]
                if abs(actual) > 0.01:  # Avoid near-zero denominators
                    mape_values.append(abs(errors[i, j] / actual) * 100)
        mape = np.mean(mape_values) if mape_values else np.nan

        print(f"Out-of-sample RMSE: {rmse:.4f}")
        print(f"Out-of-sample MAE:  {mae:.4f}")
        print(f"Out-of-sample MAPE: {mape:.2f}%")

        # Residual diagnostics (stationarity)
        residuals = vecm_full_result.resid
        stationary_count = 0
        for var_idx in range(residuals.shape[1]):
            adf_result = adfuller(residuals[:, var_idx], autolag='AIC')
            if adf_result[1] < 0.05:  # p-value < 0.05
                stationary_count += 1

        stationarity_pct = (stationary_count / residuals.shape[1]) * 100
        print(f"Residual stationarity: {stationary_count}/{residuals.shape[1]} variables ({stationarity_pct:.1f}%)")

        # Store results
        results['rank'].append(rank)
        results['aic'].append(aic)
        results['bic'].append(bic)
        results['hqic'].append(hqic)
        results['fpe'].append(fpe)
        results['llf'].append(llf)
        results['out_of_sample_rmse'].append(rmse)
        results['out_of_sample_mae'].append(mae)
        results['out_of_sample_mape'].append(mape)
        results['residual_stationarity_pct'].append(stationarity_pct)
        results['trace_statistic'].append(trace_stats[rank-1])

    except Exception as e:
        print(f"ERROR estimating rank {rank}: {e}")
        # Store NaN for failed ranks
        results['rank'].append(rank)
        results['aic'].append(np.nan)
        results['bic'].append(np.nan)
        results['hqic'].append(np.nan)
        results['fpe'].append(np.nan)
        results['llf'].append(np.nan)
        results['out_of_sample_rmse'].append(np.nan)
        results['out_of_sample_mae'].append(np.nan)
        results['out_of_sample_mape'].append(np.nan)
        results['residual_stationarity_pct'].append(np.nan)
        results['trace_statistic'].append(trace_stats[rank-1])

# Create results DataFrame
results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)
print("\n" + results_df.to_string(index=False))

# Save to Excel
output_file = OUTPUT_DIR / "vecm_rank_robustness_results.xlsx"
results_df.to_excel(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Identify best rank by each criterion
print("\n" + "=" * 80)
print("BEST RANK BY EACH CRITERION")
print("=" * 80)

# Lower is better for AIC, BIC, HQIC, FPE, RMSE, MAE, MAPE
try:
    best_aic = results_df.loc[results_df['aic'].idxmin()]
    print(f"\nAIC (lower is better):                     Rank {int(best_aic['rank'])} (AIC={best_aic['aic']:.2f})")
except:
    print(f"\nAIC (lower is better):                     No valid results")

try:
    best_bic = results_df.loc[results_df['bic'].idxmin()]
    print(f"BIC (lower is better):                     Rank {int(best_bic['rank'])} (BIC={best_bic['bic']:.2f})")
except:
    print(f"BIC (lower is better):                     No valid results")

try:
    best_hqic = results_df.loc[results_df['hqic'].idxmin()]
    print(f"HQIC (lower is better):                    Rank {int(best_hqic['rank'])} (HQIC={best_hqic['hqic']:.2f})")
except:
    print(f"HQIC (lower is better):                    No valid results")

try:
    best_llf = results_df.loc[results_df['llf'].idxmax()]
    print(f"Log-Likelihood (higher is better):         Rank {int(best_llf['rank'])} (LLF={best_llf['llf']:.2f})")
except:
    print(f"Log-Likelihood (higher is better):         No valid results")

try:
    best_rmse = results_df.loc[results_df['out_of_sample_rmse'].idxmin()]
    print(f"Out-of-sample RMSE (lower is better):      Rank {int(best_rmse['rank'])} (RMSE={best_rmse['out_of_sample_rmse']:.4f})")
except:
    print(f"Out-of-sample RMSE (lower is better):      No valid results")

try:
    best_mae = results_df.loc[results_df['out_of_sample_mae'].idxmin()]
    print(f"Out-of-sample MAE (lower is better):       Rank {int(best_mae['rank'])} (MAE={best_mae['out_of_sample_mae']:.4f})")
except:
    print(f"Out-of-sample MAE (lower is better):       No valid results")

try:
    best_mape = results_df.loc[results_df['out_of_sample_mape'].idxmin()]
    print(f"Out-of-sample MAPE (lower is better):      Rank {int(best_mape['rank'])} (MAPE={best_mape['out_of_sample_mape']:.2f}%)")
except:
    print(f"Out-of-sample MAPE (lower is better):      No valid results")

try:
    best_stationarity = results_df.loc[results_df['residual_stationarity_pct'].idxmax()]
    print(f"Residual stationarity (higher is better):  Rank {int(best_stationarity['rank'])} ({best_stationarity['residual_stationarity_pct']:.1f}%)")
except:
    print(f"Residual stationarity (higher is better):  No valid results")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('VECM Rank Robustness Check: Model Comparison Across Ranks', fontsize=16, fontweight='bold')

# Plot 1: Information Criteria (AIC, BIC, HQIC)
ax1 = axes[0, 0]
ax1.plot(results_df['rank'], results_df['aic'], 'o-', label='AIC', linewidth=2, markersize=8)
ax1.plot(results_df['rank'], results_df['bic'], 's-', label='BIC', linewidth=2, markersize=8)
ax1.plot(results_df['rank'], results_df['hqic'], '^-', label='HQIC', linewidth=2, markersize=8)
ax1.axvline(x=6, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current (rank=6)')
ax1.set_xlabel('Cointegration Rank', fontsize=12, fontweight='bold')
ax1.set_ylabel('Information Criterion', fontsize=12, fontweight='bold')
ax1.set_title('Information Criteria (lower is better)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, 9))

# Plot 2: Out-of-sample RMSE
ax2 = axes[0, 1]
ax2.plot(results_df['rank'], results_df['out_of_sample_rmse'], 'o-', color='navy', linewidth=2, markersize=8)
ax2.axvline(x=6, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current (rank=6)')
ax2.set_xlabel('Cointegration Rank', fontsize=12, fontweight='bold')
ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax2.set_title('Out-of-Sample Forecast Error (lower is better)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, 9))

# Plot 3: Out-of-sample MAPE
ax3 = axes[0, 2]
ax3.plot(results_df['rank'], results_df['out_of_sample_mape'], 'o-', color='darkgreen', linewidth=2, markersize=8)
ax3.axvline(x=6, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current (rank=6)')
ax3.set_xlabel('Cointegration Rank', fontsize=12, fontweight='bold')
ax3.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax3.set_title('Out-of-Sample MAPE (lower is better)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(1, 9))

# Plot 4: Log-Likelihood
ax4 = axes[1, 0]
ax4.plot(results_df['rank'], results_df['llf'], 'o-', color='purple', linewidth=2, markersize=8)
ax4.axvline(x=6, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current (rank=6)')
ax4.set_xlabel('Cointegration Rank', fontsize=12, fontweight='bold')
ax4.set_ylabel('Log-Likelihood', fontsize=12, fontweight='bold')
ax4.set_title('Log-Likelihood (higher is better)', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(1, 9))

# Plot 5: Residual Stationarity
ax5 = axes[1, 1]
ax5.bar(results_df['rank'], results_df['residual_stationarity_pct'], color='teal', alpha=0.7, edgecolor='black')
ax5.axvline(x=6, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current (rank=6)')
ax5.set_xlabel('Cointegration Rank', fontsize=12, fontweight='bold')
ax5.set_ylabel('% Variables Stationary', fontsize=12, fontweight='bold')
ax5.set_title('Residual Stationarity (higher is better)', fontsize=13, fontweight='bold')
ax5.set_ylim([0, 105])
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_xticks(range(1, 9))

# Plot 6: Trace Statistics
ax6 = axes[1, 2]
ax6.bar(results_df['rank'], results_df['trace_statistic'], color='coral', alpha=0.7, edgecolor='black')
ax6.plot(range(1, 9), trace_crit_95, 'k--', linewidth=2, label='95% Critical Value')
ax6.axvline(x=6, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current (rank=6)')
ax6.set_xlabel('Cointegration Rank (H0: r<=rank)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Trace Statistic', fontsize=12, fontweight='bold')
ax6.set_title('Johansen Trace Statistics', fontsize=13, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_xticks(range(1, 9))

plt.tight_layout()
plot_file = OUTPUT_DIR / "vecm_rank_robustness_comparison.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\nComparison plot saved to: {plot_file}")
plt.close()

# Create a ranking table (count how many criteria favor each rank)
print("\n" + "=" * 80)
print("OVERALL RANK PERFORMANCE")
print("=" * 80)

# Count "wins" for each rank
rank_scores = {r: 0 for r in range(1, 9)}

# Award points: 1st place = 3 points, 2nd = 2, 3rd = 1
criteria_to_minimize = ['aic', 'bic', 'hqic', 'out_of_sample_rmse', 'out_of_sample_mae', 'out_of_sample_mape']
criteria_to_maximize = ['llf', 'residual_stationarity_pct']

for criterion in criteria_to_minimize:
    sorted_ranks = results_df.nsmallest(3, criterion)['rank'].values
    if len(sorted_ranks) >= 1:
        rank_scores[int(sorted_ranks[0])] += 3
    if len(sorted_ranks) >= 2:
        rank_scores[int(sorted_ranks[1])] += 2
    if len(sorted_ranks) >= 3:
        rank_scores[int(sorted_ranks[2])] += 1

for criterion in criteria_to_maximize:
    sorted_ranks = results_df.nlargest(3, criterion)['rank'].values
    if len(sorted_ranks) >= 1:
        rank_scores[int(sorted_ranks[0])] += 3
    if len(sorted_ranks) >= 2:
        rank_scores[int(sorted_ranks[1])] += 2
    if len(sorted_ranks) >= 3:
        rank_scores[int(sorted_ranks[2])] += 1

print("\nRank Performance (3 points for 1st, 2 for 2nd, 1 for 3rd across 8 criteria):")
print(f"{'Rank':<10} {'Score':<10}")
print("-" * 20)
for rank in sorted(rank_scores.keys(), key=lambda x: rank_scores[x], reverse=True):
    marker = " <-- CURRENT" if rank == 6 else ""
    print(f"{rank:<10} {rank_scores[rank]:<10}{marker}")

best_overall_rank = max(rank_scores.keys(), key=lambda x: rank_scores[x])
print(f"\nBest overall rank: {best_overall_rank} (score={rank_scores[best_overall_rank]})")
print(f"Current rank: 6 (score={rank_scores[6]})")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if best_overall_rank == 6:
    print("\nCurrent rank=6 is OPTIMAL across criteria. No change needed.")
else:
    diff = abs(rank_scores[best_overall_rank] - rank_scores[6])
    if diff <= 2:
        print(f"\nRank {best_overall_rank} scores slightly better (difference={diff} points).")
        print("Current rank=6 is defensible - difference is marginal.")
    else:
        print(f"\nRank {best_overall_rank} scores notably better (difference={diff} points).")
        print(f"Consider switching to rank={best_overall_rank} for improved model fit.")

print("\nAnalysis complete!")
print("=" * 80)
