"""
Test VECM rank=2 with extended lags (k_ar_diff = 1, 2, 3, 4)
to see if more short-run dynamics help capture GOFOs-Junior Enlisted
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis")
DATA_FILE = BASE_DIR / "complete_normalized_dataset_v12.3.xlsx"

# Load data
df = pd.read_excel(DATA_FILE)

# Select the 8 VECM variables (same as previous analyses)
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

# Prepare data
df_vecm = df[SELECTED_VARS].dropna()
print(f"Dataset: {len(df_vecm)} observations")

# Get empirical GOFOs-JE correlation
empirical_corr = np.corrcoef(
    df_vecm['GOFOs_Z'].values,
    df_vecm['Junior_Enlisted_Z'].values
)[0, 1]

print("\n" + "=" * 80)
print("EMPIRICAL BASELINE")
print("=" * 80)
print(f"GOFOs <=> Junior Enlisted correlation: {empirical_corr:+.3f}")
print(f"Expected relationship: OPPOSITE/DAMPENING (negative influence)")

# Test-train split for out-of-sample validation
train_end = 30  # Use first 30 observations for training
test_start = 30  # Forecast remaining observations

train_data = df_vecm.iloc[:train_end]
test_data = df_vecm.iloc[test_start:]

print(f"\nTrain: {len(train_data)} obs, Test: {len(test_data)} obs")

# Test different lag orders
lags_to_test = [1, 2, 3, 4]
results = []

for lag in lags_to_test:
    print("\n" + "=" * 80)
    print(f"RANK=2, K_AR_DIFF={lag} (lag order)")
    print("=" * 80)

    try:
        # Estimate VECM
        model = VECM(train_data, k_ar_diff=lag, coint_rank=2, deterministic='ci')
        vecm_fit = model.fit()

        # Get alpha and beta matrices
        alpha = vecm_fit.alpha  # (8, 2)
        beta = vecm_fit.beta    # (8, 2)

        # Get gamma matrices (short-run dynamics)
        # With k_ar_diff=lag, we have 'lag' gamma matrices
        gamma_matrices = []
        for i in range(lag):
            gamma_matrices.append(vecm_fit.gamma[i])  # Each is (8, 8)

        # Find indices
        gofo_idx = SELECTED_VARS.index('GOFOs_Z')
        je_idx = SELECTED_VARS.index('Junior_Enlisted_Z')

        # Calculate LONG-RUN GOFOs => Junior Enlisted influence (from alpha*beta)
        longrun_influence = 0
        print(f"\nLONG-RUN cointegration contributions:")
        for r in range(2):
            alpha_je = alpha[je_idx, r]
            beta_gofo = beta[gofo_idx, r]
            contribution = alpha_je * beta_gofo
            longrun_influence += contribution
            print(f"  EC{r+1}: alpha_JE={alpha_je:+.4f} x beta_GOFO={beta_gofo:+.4f} = {contribution:+.4f}")

        print(f"\nTotal LONG-RUN GOFOs => Junior Enlisted: {longrun_influence:+.4f}")
        print(f"Direction: {'AMPLIFYING (+)' if longrun_influence > 0 else 'DAMPENING (-)'}")

        # Calculate SHORT-RUN GOFOs => Junior Enlisted influence (from gamma)
        shortrun_influence = 0
        print(f"\nSHORT-RUN dynamic contributions:")
        for i, gamma in enumerate(gamma_matrices):
            gamma_influence = gamma[je_idx, gofo_idx]  # Direct JE <- GOFO effect
            shortrun_influence += gamma_influence
            print(f"  Lag {i+1}: gamma_JE,GOFO = {gamma_influence:+.4f}")

        print(f"\nTotal SHORT-RUN GOFOs => Junior Enlisted: {shortrun_influence:+.4f}")
        print(f"Direction: {'AMPLIFYING (+)' if shortrun_influence > 0 else 'DAMPENING (-)'}")

        # TOTAL influence (long-run + short-run)
        total_influence = longrun_influence + shortrun_influence
        print(f"\n*** TOTAL INFLUENCE (Long + Short): {total_influence:+.4f} ***")
        print(f"Direction: {'AMPLIFYING (+)' if total_influence > 0 else 'DAMPENING (-)'}")

        # Check if matches empirical
        matches_empirical_longrun = (
            (longrun_influence < 0 and empirical_corr < 0) or
            (longrun_influence > 0 and empirical_corr > 0)
        )
        matches_empirical_total = (
            (total_influence < 0 and empirical_corr < 0) or
            (total_influence > 0 and empirical_corr > 0)
        )

        print(f"\nMatches empirical (long-run): {'YES' if matches_empirical_longrun else 'NO'}")
        print(f"Matches empirical (total): {'YES' if matches_empirical_total else 'NO'}")

        # Out-of-sample forecast
        forecast_steps = len(test_data)
        forecast = vecm_fit.predict(steps=forecast_steps)

        # Calculate MAE for all variables
        mae_per_var = np.abs(forecast - test_data.values).mean(axis=0)
        mae_overall = mae_per_var.mean()

        # GOFOs specific MAE
        mae_gofos = mae_per_var[gofo_idx]
        mae_je = mae_per_var[je_idx]

        print(f"\nOut-of-sample forecast performance:")
        print(f"  Overall MAE: {mae_overall:.4f}")
        print(f"  GOFOs MAE: {mae_gofos:.4f}")
        print(f"  Junior Enlisted MAE: {mae_je:.4f}")

        # Model fit statistics
        aic = vecm_fit.aic
        bic = vecm_fit.bic
        print(f"\nModel fit:")
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  (Lower is better)")

        # Store results
        results.append({
            'lag': lag,
            'longrun_influence': longrun_influence,
            'shortrun_influence': shortrun_influence,
            'total_influence': total_influence,
            'matches_longrun': matches_empirical_longrun,
            'matches_total': matches_empirical_total,
            'mae_overall': mae_overall,
            'mae_gofos': mae_gofos,
            'mae_je': mae_je,
            'aic': aic,
            'bic': bic,
            'num_params': alpha.size + beta.size + sum([g.size for g in gamma_matrices])
        })

    except Exception as e:
        print(f"\nERROR with lag={lag}: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append({
            'lag': lag,
            'error': str(e)
        })

# Summary comparison
print("\n" + "=" * 80)
print("SUMMARY COMPARISON (RANK=2, VARYING LAGS)")
print("=" * 80)
print(f"{'Lag':<5} {'LongRun':<10} {'ShortRun':<10} {'TOTAL':<10} {'Match?':<8} "
      f"{'MAE':<8} {'AIC':<10} {'BIC':<10}")
print("-" * 80)

for r in results:
    if 'total_influence' in r:
        print(f"{r['lag']:<5} {r['longrun_influence']:+.4f}    "
              f"{r['shortrun_influence']:+.4f}    "
              f"{r['total_influence']:+.4f}    "
              f"{'YES' if r['matches_total'] else 'NO':<8} "
              f"{r['mae_overall']:.4f}   "
              f"{r['aic']:<10.1f} {r['bic']:<10.1f}")
    else:
        print(f"{r['lag']:<5} ERROR: {r.get('error', 'Unknown')}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print(f"\nEmpirical correlation: {empirical_corr:+.3f} (NEGATIVE => opposite movement)")
print("\nKey insights:")
print("  - Long-run = cointegration relationship (alpha*beta)")
print("  - Short-run = immediate VAR dynamics (gamma)")
print("  - Total = combined effect")
print("\nLook for:")
print("  1. Match? = YES (total influence sign matches empirical)")
print("  2. Lower MAE = better forecast performance")
print("  3. Lower AIC/BIC = better model fit (penalizes complexity)")
print("\nIf adding lags helps, you'll see:")
print("  - Short-run influence counteracting wrong long-run sign")
print("  - Total influence becoming negative (matching empirical)")
print("  - Improved forecast performance (lower MAE)")
