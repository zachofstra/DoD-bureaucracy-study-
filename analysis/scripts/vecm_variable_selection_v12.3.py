"""
VECM Variable Selection - v12.3 Dataset
Step 1: Test stationarity of all 19 variables
Step 2: Run Johansen cointegration tests on multiple combinations
Step 3: Recommend optimal variable subset for VECM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("VECM VARIABLE SELECTION ANALYSIS - v12.3 DATASET")
print("Stationarity Testing + Johansen Cointegration")
print("=" * 100)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = 'analysis/complete_normalized_dataset_v12.3.xlsx'
OUTPUT_DIR = 'analysis/VECM_Variable_Selection_v12.3'
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

# =============================================================================
# STEP 1: TEST STATIONARITY OF ALL 19 VARIABLES
# =============================================================================
print("\n[STEP 1] Testing stationarity of all 19 variables...")
print("-" * 100)

df = pd.read_excel(DATA_FILE)

# All 19 variables
all_vars = [
    'Junior_Enlisted_Z',
    'Middle_Enlisted_Z',
    'Senior_Enlisted_Z',
    'Company_Grade_Officers_Z',
    'Field_Grade_Officers_Z',
    'GOFOs_Z',
    'Warrant_Officers_Z',
    'GDP_Growth_Z',
    'Major_Conflict',
    'Policy_Count_Log',
    'Total_Civilians_Z',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z',
    'Democrat Party HOR',
    'Republican Party HOR',
    'Democrat Party Senate',
    'Republican Party Senate',
    'POTUS Democrat Party',
    'POTUS Republican Party'
]

# Filter to available variables
available_vars = [v for v in all_vars if v in df.columns]
data = df[available_vars].dropna()

print(f"\nTotal variables: {len(available_vars)}")
print(f"Observations: {len(data)}")

# Test stationarity in levels
print("\n1A. ADF Tests - LEVELS")
print("-" * 100)

levels_results = []
for var in available_vars:
    adf_result = adfuller(data[var], autolag='AIC')
    levels_results.append({
        'Variable': var,
        'ADF_Stat': adf_result[0],
        'p_value': adf_result[1],
        'Lags': adf_result[2],
        'Stationary': 'Yes' if adf_result[1] < 0.05 else 'No'
    })

levels_df = pd.DataFrame(levels_results)
print(f"\n{'Variable':<35} {'ADF Stat':<10} {'p-value':<10} {'Status':<15}")
print("-" * 100)
for _, row in levels_df.iterrows():
    status = "[STATIONARY]" if row['Stationary'] == 'Yes' else "[NON-STATIONARY]"
    print(f"{row['Variable']:<35} {row['ADF_Stat']:<10.4f} {row['p_value']:<10.4f} {status}")

stationary_count = (levels_df['Stationary'] == 'Yes').sum()
non_stationary_vars = levels_df[levels_df['Stationary'] == 'No']['Variable'].tolist()

print(f"\nSummary:")
print(f"  Stationary in levels: {stationary_count}/{len(available_vars)}")
print(f"  Non-stationary in levels: {len(non_stationary_vars)}/{len(available_vars)}")

# Test stationarity in FIRST DIFFERENCES (for non-stationary variables)
print("\n1B. ADF Tests - FIRST DIFFERENCES (Non-stationary variables)")
print("-" * 100)

diff_results = []
i1_variables = []  # I(1) variables: non-stationary in levels, stationary in differences

for var in non_stationary_vars:
    # First difference
    diff_series = data[var].diff().dropna()
    adf_result = adfuller(diff_series, autolag='AIC')

    is_stationary = adf_result[1] < 0.05
    diff_results.append({
        'Variable': var,
        'ADF_Stat': adf_result[0],
        'p_value': adf_result[1],
        'Lags': adf_result[2],
        'Stationary': 'Yes' if is_stationary else 'No',
        'Integration': 'I(1)' if is_stationary else 'I(2) or higher'
    })

    if is_stationary:
        i1_variables.append(var)

diff_df = pd.DataFrame(diff_results)
print(f"\n{'Variable':<35} {'ADF Stat':<10} {'p-value':<10} {'Integration':<15}")
print("-" * 100)
for _, row in diff_df.iterrows():
    print(f"{row['Variable']:<35} {row['ADF_Stat']:<10.4f} {row['p_value']:<10.4f} {row['Integration']}")

print(f"\nI(1) Variables (suitable for VECM): {len(i1_variables)}")
for i, var in enumerate(i1_variables, 1):
    print(f"  {i}. {var}")

# Save stationarity results
levels_df.to_excel(f'{OUTPUT_DIR}/stationarity_levels.xlsx', index=False)
if len(diff_df) > 0:
    diff_df.to_excel(f'{OUTPUT_DIR}/stationarity_first_differences.xlsx', index=False)

# =============================================================================
# STEP 2: JOHANSEN COINTEGRATION TESTS
# =============================================================================
print("\n" + "=" * 100)
print("[STEP 2] Johansen Cointegration Tests on Different Variable Combinations")
print("=" * 100)

if len(i1_variables) < 2:
    print("\nERROR: Need at least 2 I(1) variables for cointegration testing")
    print(f"Found only {len(i1_variables)} I(1) variable(s)")
    exit(1)

# Prepare data with I(1) variables only
i1_data = data[i1_variables].copy()

print(f"\nTesting combinations of I(1) variables...")
print(f"Total I(1) variables: {len(i1_variables)}")

# Test different subset sizes
johansen_results = []

for subset_size in range(min(6, len(i1_variables)), min(9, len(i1_variables) + 1)):
    print(f"\n{'-'*100}")
    print(f"Testing all combinations of {subset_size} variables")
    print(f"{'-'*100}")

    # Get all combinations of this size
    var_combos = list(combinations(i1_variables, subset_size))

    if len(var_combos) > 20:
        print(f"  Too many combinations ({len(var_combos)}), testing first 20 only...")
        var_combos = var_combos[:20]

    for i, var_subset in enumerate(var_combos, 1):
        var_list = list(var_subset)

        try:
            # Run Johansen test
            test_data = i1_data[var_list].dropna()

            if len(test_data) < 20:
                continue

            # Test with lag order 2 (from previous analysis)
            joh_result = coint_johansen(test_data, det_order=0, k_ar_diff=1)

            # Get trace statistics and critical values
            trace_stats = joh_result.trace_stat
            trace_crit_95 = joh_result.trace_stat_crit_vals[:, 1]  # 95% critical values

            # Determine cointegration rank
            coint_rank = 0
            for r in range(len(trace_stats)):
                if trace_stats[r] > trace_crit_95[r]:
                    coint_rank = r + 1

            # Max eigenvalue test
            max_eig_stats = joh_result.max_eig_stat
            max_eig_crit_95 = joh_result.max_eig_stat_crit_vals[:, 1]

            # Store results
            johansen_results.append({
                'Subset_Size': subset_size,
                'Combination': i,
                'Variables': ', '.join(var_list),
                'Coint_Rank_Trace': coint_rank,
                'Trace_Stat_r0': trace_stats[0],
                'Trace_Crit_95_r0': trace_crit_95[0],
                'Max_Eig_Stat_r0': max_eig_stats[0],
                'Max_Eig_Crit_95_r0': max_eig_crit_95[0],
                'Observations': len(test_data)
            })

            if i <= 5 or coint_rank >= 2:  # Print first 5 or if has cointegration
                print(f"\n  Combo {i}: {subset_size} variables, Rank={coint_rank}")
                print(f"    Variables: {', '.join(var_list[:3])}...")
                print(f"    Trace stat (r=0): {trace_stats[0]:.2f} vs crit {trace_crit_95[0]:.2f}")

        except Exception as e:
            continue

johansen_df = pd.DataFrame(johansen_results)

if len(johansen_df) > 0:
    johansen_df = johansen_df.sort_values('Coint_Rank_Trace', ascending=False)
    johansen_df.to_excel(f'{OUTPUT_DIR}/johansen_tests_all_combinations.xlsx', index=False)

    print("\n" + "=" * 100)
    print("TOP 10 VARIABLE COMBINATIONS BY COINTEGRATION RANK")
    print("=" * 100)

    top_10 = johansen_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"\n{row['Subset_Size']} Variables (Rank={row['Coint_Rank_Trace']}):")
        vars_short = row['Variables'].split(', ')
        for v in vars_short:
            print(f"  - {v}")
        print(f"  Trace stat: {row['Trace_Stat_r0']:.2f} (crit: {row['Trace_Crit_95_r0']:.2f})")

# =============================================================================
# STEP 3: RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 100)
print("[STEP 3] RECOMMENDATIONS FOR VECM")
print("=" * 100)

print(f"\nSTATIONARITY SUMMARY:")
print(f"  Total variables tested: {len(available_vars)}")
print(f"  I(1) variables (suitable for VECM): {len(i1_variables)}")
print(f"  I(0) variables (stationary in levels): {stationary_count}")

if len(johansen_df) > 0:
    best_combo = johansen_df.iloc[0]

    print(f"\nBEST VARIABLE COMBINATION:")
    print(f"  Number of variables: {best_combo['Subset_Size']}")
    print(f"  Cointegration rank: {best_combo['Coint_Rank_Trace']}")
    print(f"  Trace statistic: {best_combo['Trace_Stat_r0']:.2f}")
    print(f"\n  Variables:")
    for var in best_combo['Variables'].split(', '):
        print(f"    - {var}")

    # Save recommended variables
    recommended_vars = best_combo['Variables'].split(', ')
    rec_df = pd.DataFrame({'Variable': recommended_vars})
    rec_df.to_excel(f'{OUTPUT_DIR}/recommended_variables_for_vecm.xlsx', index=False)

    print(f"\nALTERNATIVE COMBINATIONS (High Cointegration):")
    for idx, row in johansen_df.head(5).iloc[1:].iterrows():
        print(f"\n  {row['Subset_Size']} variables (Rank={row['Coint_Rank_Trace']}):")
        print(f"    {row['Variables']}")

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print(f"  1. stationarity_levels.xlsx - ADF tests in levels")
print(f"  2. stationarity_first_differences.xlsx - ADF tests in first differences")
print(f"  3. johansen_tests_all_combinations.xlsx - All Johansen test results")
print(f"  4. recommended_variables_for_vecm.xlsx - Best variable subset")

print("\n" + "=" * 100)
print("NEXT STEPS:")
print("=" * 100)
print(f"  1. Review recommended variable combination above")
print(f"  2. Run VECM with {best_combo['Subset_Size']} variables")
print(f"  3. Estimate cointegration rank = {best_combo['Coint_Rank_Trace']}")
print(f"  4. Interpret long-run equilibrium relationships")
print("=" * 100)
