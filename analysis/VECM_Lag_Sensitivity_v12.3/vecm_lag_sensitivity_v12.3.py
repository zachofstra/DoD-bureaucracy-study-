"""
VECM Lag Sensitivity Analysis - v12.3 Dataset
Estimate VECM at lags 1-6 and compare AIC/BIC/HQIC
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("VECM LAG SENSITIVITY ANALYSIS - v12.3 DATASET")
print("Testing lags 1-6 (where numerically stable)")
print("=" * 100)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = 'analysis/complete_normalized_dataset_v12.3.xlsx'
OUTPUT_DIR = 'analysis/VECM_Lag_Sensitivity_v12.3'
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

# 8 variables from final VECM
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

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/2] Loading data...")

df = pd.read_excel(DATA_FILE)
data = df[SELECTED_VARS].dropna()

print(f"  Observations: {len(data)}")
print(f"  Variables: {len(SELECTED_VARS)}")

# =============================================================================
# TEST LAGS 1-6
# =============================================================================
print("\n[2/2] Estimating VECM at lags 1-6...")
print("-" * 100)

lag_results = []

for lag in range(1, 7):
    print(f"\n  Testing lag {lag} (VAR order {lag+1})...")

    try:
        # Run Johansen test first to get rank
        joh_result = coint_johansen(data, det_order=0, k_ar_diff=lag)

        trace_stats = joh_result.trace_stat
        trace_crit_95 = joh_result.trace_stat_crit_vals[:, 1]

        # Determine cointegration rank at 95%
        coint_rank = 0
        for r in range(len(trace_stats)):
            if trace_stats[r] > trace_crit_95[r]:
                coint_rank = r + 1

        # If full rank, adjust for estimation
        if coint_rank == len(SELECTED_VARS):
            estimation_rank = max(1, coint_rank - 2)
            full_rank_warning = "YES (adjusted to rank-2 for estimation)"
        else:
            estimation_rank = coint_rank
            full_rank_warning = "No"

        # Estimate VECM
        vecm_model = VECM(data, k_ar_diff=lag, coint_rank=estimation_rank, deterministic='nc')
        vecm_result = vecm_model.fit()

        # Calculate information criteria manually
        # Number of parameters: neqs * (neqs * lag + coint_rank) for each equation
        neqs = len(SELECTED_VARS)
        k_params = neqs * (neqs * lag + estimation_rank)
        n_obs = vecm_result.nobs
        llf = vecm_result.llf

        aic = -2 * llf + 2 * k_params
        bic = -2 * llf + k_params * np.log(n_obs)
        hqic = -2 * llf + 2 * k_params * np.log(np.log(n_obs))

        # Extract fit statistics
        lag_results.append({
            'Lag': lag,
            'VAR_Order': lag + 1,
            'Johansen_Rank': coint_rank,
            'Estimation_Rank': estimation_rank,
            'Full_Rank': full_rank_warning,
            'Observations': n_obs,
            'AIC': aic,
            'BIC': bic,
            'HQIC': hqic,
            'Log_Likelihood': llf,
            'Status': 'Success'
        })

        print(f"    [OK] Success")
        print(f"      Johansen rank: {coint_rank}, Estimation rank: {estimation_rank}")
        print(f"      AIC: {aic:.2f}, BIC: {bic:.2f}, HQIC: {hqic:.2f}")

    except np.linalg.LinAlgError as e:
        lag_results.append({
            'Lag': lag,
            'VAR_Order': lag + 1,
            'Johansen_Rank': None,
            'Estimation_Rank': None,
            'Full_Rank': None,
            'Observations': None,
            'AIC': None,
            'BIC': None,
            'HQIC': None,
            'Log_Likelihood': None,
            'Status': f'Numerical Error: {str(e)[:50]}'
        })
        print(f"    [FAIL] Numerical error: {e}")

    except Exception as e:
        lag_results.append({
            'Lag': lag,
            'VAR_Order': lag + 1,
            'Johansen_Rank': None,
            'Estimation_Rank': None,
            'Full_Rank': None,
            'Observations': None,
            'AIC': None,
            'BIC': None,
            'HQIC': None,
            'Log_Likelihood': None,
            'Status': f'Error: {str(e)[:50]}'
        })
        print(f"    [FAIL] Error: {e}")

# =============================================================================
# SAVE AND DISPLAY RESULTS
# =============================================================================
results_df = pd.DataFrame(lag_results)
results_df.to_excel(f'{OUTPUT_DIR}/lag_comparison.xlsx', index=False)

print("\n" + "=" * 100)
print("LAG SENSITIVITY RESULTS")
print("=" * 100)

print(f"\n{'Lag':<6} {'VAR':<6} {'Rank':<6} {'Obs':<6} {'AIC':<12} {'BIC':<12} {'HQIC':<12} {'Status':<20}")
print("-" * 100)

for _, row in results_df.iterrows():
    lag_str = f"{row['Lag']}"
    var_str = f"{row['VAR_Order']}" if row['VAR_Order'] is not None else "N/A"
    rank_str = f"{row['Estimation_Rank']}" if row['Estimation_Rank'] is not None else "N/A"
    obs_str = f"{row['Observations']}" if row['Observations'] is not None else "N/A"
    aic_str = f"{row['AIC']:.2f}" if row['AIC'] is not None else "N/A"
    bic_str = f"{row['BIC']:.2f}" if row['BIC'] is not None else "N/A"
    hqic_str = f"{row['HQIC']:.2f}" if row['HQIC'] is not None else "N/A"
    status_str = row['Status'][:20]

    print(f"{lag_str:<6} {var_str:<6} {rank_str:<6} {obs_str:<6} {aic_str:<12} {bic_str:<12} {hqic_str:<12} {status_str:<20}")

# Identify best lags by each criterion (only successful estimations)
successful = results_df[results_df['Status'] == 'Success']

if len(successful) > 0:
    print("\n" + "=" * 100)
    print("OPTIMAL LAG SELECTION")
    print("=" * 100)

    best_aic = successful.loc[successful['AIC'].idxmin()]
    best_bic = successful.loc[successful['BIC'].idxmin()]
    best_hqic = successful.loc[successful['HQIC'].idxmin()]

    print(f"\n  Best by AIC:  Lag {best_aic['Lag']} (AIC = {best_aic['AIC']:.2f})")
    print(f"  Best by BIC:  Lag {best_bic['Lag']} (BIC = {best_bic['BIC']:.2f})")
    print(f"  Best by HQIC: Lag {best_hqic['Lag']} (HQIC = {best_hqic['HQIC']:.2f})")

    print("\n  RECOMMENDATION:")
    if best_bic['Lag'] == best_hqic['Lag']:
        print(f"    BIC and HQIC both select lag {best_bic['Lag']}")
        print(f"    This is the optimal lag order (BIC penalizes complexity more)")
    else:
        print(f"    BIC selects lag {best_bic['Lag']} (more parsimonious)")
        print(f"    AIC selects lag {best_aic['Lag']} (better fit)")
        print(f"    Recommend lag {best_bic['Lag']} for sample size n={len(data)}")

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print(f"  1. lag_comparison.xlsx - Full lag sensitivity results")

print("\n" + "=" * 100)
print("INTERPRETATION:")
print("=" * 100)
print("""
  - AIC = Akaike Information Criterion (penalizes complexity moderately)
  - BIC = Bayesian Information Criterion (penalizes complexity more heavily)
  - HQIC = Hannan-Quinn IC (between AIC and BIC)

  - Lower values = better fit
  - BIC preferred for small samples (n=38)
  - Lag order = number of first-differenced lags in VECM
  - VAR order = lag order + 1 (in levels)
""")
print("=" * 100)
