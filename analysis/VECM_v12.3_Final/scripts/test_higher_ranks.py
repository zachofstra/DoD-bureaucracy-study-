"""
Test VECM ranks 4, 5, and 6 to see if more cointegration relationships
can correctly capture the GOFOs-Junior Enlisted relationship
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis")
DATA_FILE = BASE_DIR / "complete_normalized_dataset_v12.3.xlsx"

# Load data
df = pd.read_excel(DATA_FILE)

# Select the 8 VECM variables (same as rank=2 analysis)
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

# Test ranks 2, 3, 4, 5, 6
ranks_to_test = [2, 3, 4, 5, 6]
results = []

for rank in ranks_to_test:
    print("\n" + "=" * 80)
    print(f"RANK = {rank}")
    print("=" * 80)

    try:
        # Estimate VECM
        model = VECM(train_data, k_ar_diff=1, coint_rank=rank, deterministic='ci')
        vecm_fit = model.fit()

        # Get alpha and beta matrices
        alpha = vecm_fit.alpha  # (8, rank)
        beta = vecm_fit.beta    # (8, rank)

        # Find indices
        gofo_idx = SELECTED_VARS.index('GOFOs_Z')
        je_idx = SELECTED_VARS.index('Junior_Enlisted_Z')

        # Calculate GOFOs => Junior Enlisted influence
        gofo_to_je_influence = 0
        print(f"\nCointegration vector contributions:")
        for r in range(rank):
            alpha_je = alpha[je_idx, r]
            beta_gofo = beta[gofo_idx, r]
            contribution = alpha_je * beta_gofo
            gofo_to_je_influence += contribution
            print(f"  EC{r+1}: alpha_JE={alpha_je:+.4f} x beta_GOFO={beta_gofo:+.4f} = {contribution:+.4f}")

        print(f"\nTotal GOFOs => Junior Enlisted: {gofo_to_je_influence:+.4f}")
        print(f"Direction: {'AMPLIFYING (+)' if gofo_to_je_influence > 0 else 'DAMPENING (-)'}")

        # Check if matches empirical
        matches_empirical = (
            (gofo_to_je_influence < 0 and empirical_corr < 0) or
            (gofo_to_je_influence > 0 and empirical_corr > 0)
        )
        print(f"Matches empirical direction: {'YES' if matches_empirical else 'NO'}")

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

        # Store results
        results.append({
            'rank': rank,
            'gofo_to_je_influence': gofo_to_je_influence,
            'matches_empirical': matches_empirical,
            'mae_overall': mae_overall,
            'mae_gofos': mae_gofos,
            'mae_je': mae_je,
            'alpha_shape': alpha.shape,
            'beta_shape': beta.shape
        })

    except Exception as e:
        print(f"\nERROR with rank={rank}: {str(e)}")
        results.append({
            'rank': rank,
            'gofo_to_je_influence': None,
            'matches_empirical': False,
            'mae_overall': None,
            'error': str(e)
        })

# Summary comparison
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)
print(f"{'Rank':<6} {'GOFO=>JE':<12} {'Match?':<8} {'MAE Overall':<12} {'MAE GOFOs':<12} {'MAE JE':<12}")
print("-" * 80)

for r in results:
    if r['gofo_to_je_influence'] is not None:
        print(f"{r['rank']:<6} {r['gofo_to_je_influence']:+.4f}      "
              f"{'YES' if r['matches_empirical'] else 'NO':<8} "
              f"{r['mae_overall']:.4f}       "
              f"{r['mae_gofos']:.4f}       "
              f"{r['mae_je']:.4f}")
    else:
        print(f"{r['rank']:<6} ERROR: {r.get('error', 'Unknown')}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print(f"\nEmpirical correlation: {empirical_corr:+.3f} (NEGATIVE => opposite movement)")
print("\nLook for:")
print("  1. Match? = YES (influence sign matches empirical correlation)")
print("  2. Lower MAE = better forecast performance")
print("\nIf all ranks show 'NO', the VECM framework may be fundamentally inadequate")
print("for these 8 variables with complex directional relationships.")
