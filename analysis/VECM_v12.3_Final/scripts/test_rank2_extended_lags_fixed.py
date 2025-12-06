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

        # Check if matches empirical
        matches_empirical_longrun = (
            (longrun_influence < 0 and empirical_corr < 0) or
            (longrun_influence > 0 and empirical_corr > 0)
        )

        print(f"Matches empirical: {'YES *** CORRECT ***' if matches_empirical_longrun else 'NO (wrong)'}")

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
            'matches_empirical': matches_empirical_longrun,
            'mae_overall': mae_overall,
            'mae_gofos': mae_gofos,
            'mae_je': mae_je,
            'aic': aic,
            'bic': bic
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
print(f"{'Lag':<5} {'GOFO=>JE':<12} {'Match?':<12} {'MAE Overall':<12} "
      f"{'MAE GOFOs':<12} {'AIC':<10} {'BIC':<10}")
print("-" * 80)

for r in results:
    if 'longrun_influence' in r:
        match_str = 'YES (CORRECT)' if r['matches_empirical'] else 'NO (wrong)'
        print(f"{r['lag']:<5} {r['longrun_influence']:+.4f}      "
              f"{match_str:<12} "
              f"{r['mae_overall']:.4f}       "
              f"{r['mae_gofos']:.4f}       "
              f"{r['aic']:<10.1f} {r['bic']:<10.1f}")
    else:
        print(f"{r['lag']:<5} ERROR: {r.get('error', 'Unknown')}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print(f"\nEmpirical correlation: {empirical_corr:+.3f} (NEGATIVE => opposite movement)")
print("\nKey insights:")
print("  - Long-run = cointegration relationship (alpha*beta)")
print("  - Match? = whether influence sign matches empirical correlation")
print("\nLook for:")
print("  1. Match? = YES (influence matches empirical direction)")
print("  2. Lower MAE = better forecast performance")
print("  3. Lower AIC/BIC = better model fit (penalizes complexity)")
print("\nBest model will have:")
print("  - Negative influence (matching empirical r=-0.775)")
print("  - Good forecast accuracy")
print("  - Reasonable model complexity")

# Find best model
valid_results = [r for r in results if 'longrun_influence' in r]
if valid_results:
    correct_sign = [r for r in valid_results if r['matches_empirical']]
    if correct_sign:
        best = min(correct_sign, key=lambda x: x['mae_overall'])
        print(f"\n*** RECOMMENDED MODEL ***")
        print(f"Lag = {best['lag']}")
        print(f"GOFOs => JE influence: {best['longrun_influence']:+.4f} (CORRECT sign)")
        print(f"MAE: {best['mae_overall']:.4f}")
        print(f"AIC: {best['aic']:.1f}, BIC: {best['bic']:.1f}")
    else:
        print("\nWARNING: No lag specification correctly captures the relationship!")
