"""
Test VECM rank=2, lag=6 to see if extensive short-run dynamics help
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis")
DATA_FILE = BASE_DIR / "complete_normalized_dataset_v12.3.xlsx"

print("=" * 80)
print("TESTING VECM: RANK=2, LAG=6")
print("=" * 80)

# Load data
df = pd.read_excel(DATA_FILE)

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

df_vecm = df[SELECTED_VARS].dropna()
print(f"\nDataset: {len(df_vecm)} observations")

# Get empirical correlation
corr_matrix = df_vecm.corr()
empirical_corr = corr_matrix.loc['GOFOs_Z', 'Junior_Enlisted_Z']

print(f"\nEMPIRICAL BASELINE:")
print(f"GOFOs <=> Junior Enlisted correlation: {empirical_corr:+.3f}")
print(f"Expected model influence: NEGATIVE (dampening)")

# Indices
gofo_idx = SELECTED_VARS.index('GOFOs_Z')
je_idx = SELECTED_VARS.index('Junior_Enlisted_Z')

print(f"\n{'='*80}")
print("FULL DATASET ESTIMATION (38 observations)")
print(f"{'='*80}")

try:
    # Estimate on full dataset
    print("\nEstimating VECM(rank=2, lag=6)...")
    model_full = VECM(df_vecm, k_ar_diff=6, coint_rank=2, deterministic='ci')
    vecm_full = model_full.fit()

    alpha_full = vecm_full.alpha
    beta_full = vecm_full.beta

    print(f"Alpha shape: {alpha_full.shape}")
    print(f"Beta shape: {beta_full.shape}")

    # Calculate GOFOs => JE influence
    gofo_je_full = sum([alpha_full[je_idx, r] * beta_full[gofo_idx, r] for r in range(2)])

    print(f"\nCointegration contributions:")
    for r in range(2):
        alpha_je = alpha_full[je_idx, r]
        beta_gofo = beta_full[gofo_idx, r]
        contribution = alpha_je * beta_gofo
        print(f"  EC{r+1}: alpha_JE={alpha_je:+.4f} x beta_GOFO={beta_gofo:+.4f} = {contribution:+.4f}")

    print(f"\n*** GOFOs => Junior Enlisted: {gofo_je_full:+.4f} ***")
    print(f"Direction: {'DAMPENING (-)' if gofo_je_full < 0 else 'AMPLIFYING (+)'}")
    print(f"Empirical correlation: {empirical_corr:+.3f}")

    matches_full = (gofo_je_full < 0 and empirical_corr < 0) or (gofo_je_full > 0 and empirical_corr > 0)
    print(f"Match: {'YES - CORRECT!' if matches_full else 'NO - WRONG'}")

    # Check all variable relationships for sign correctness
    print(f"\n{'='*80}")
    print("CHECKING ALL VARIABLE RELATIONSHIPS")
    print(f"{'='*80}")

    correct_count = 0
    wrong_count = 0

    for i, target in enumerate(SELECTED_VARS):
        for j, source in enumerate(SELECTED_VARS):
            if i == j:
                continue

            model_influence = sum([alpha_full[i, r] * beta_full[j, r] for r in range(2)])
            empirical_corr_pair = corr_matrix.loc[target, source]

            matches = (model_influence < 0 and empirical_corr_pair < 0) or (model_influence > 0 and empirical_corr_pair > 0)

            if matches or abs(model_influence) < 0.01:  # Consider near-zero as neutral
                correct_count += 1
            else:
                wrong_count += 1
                if abs(model_influence) > 0.5:  # Only report significant mismatches
                    print(f"  {source:30s} => {target:30s}: model={model_influence:+.3f}, empirical r={empirical_corr_pair:+.3f} [WRONG]")

    total_pairs = len(SELECTED_VARS) * (len(SELECTED_VARS) - 1)
    print(f"\nOverall sign correctness: {correct_count}/{total_pairs} ({100*correct_count/total_pairs:.1f}%)")

except Exception as e:
    print(f"\nERROR with full dataset: {str(e)}")
    import traceback
    traceback.print_exc()

# Test with training subset
print(f"\n{'='*80}")
print("TRAINING SUBSET ESTIMATION (30 observations)")
print(f"{'='*80}")

train_end = 30
train_data = df_vecm.iloc[:train_end]
test_data = df_vecm.iloc[train_end:]

try:
    print(f"\nEstimating VECM(rank=2, lag=6) on {len(train_data)} observations...")
    model_train = VECM(train_data, k_ar_diff=6, coint_rank=2, deterministic='ci')
    vecm_train = model_train.fit()

    alpha_train = vecm_train.alpha
    beta_train = vecm_train.beta

    # Calculate GOFOs => JE influence
    gofo_je_train = sum([alpha_train[je_idx, r] * beta_train[gofo_idx, r] for r in range(2)])

    print(f"\nCointegration contributions:")
    for r in range(2):
        alpha_je = alpha_train[je_idx, r]
        beta_gofo = beta_train[gofo_idx, r]
        contribution = alpha_je * beta_gofo
        print(f"  EC{r+1}: alpha_JE={alpha_je:+.4f} x beta_GOFO={beta_gofo:+.4f} = {contribution:+.4f}")

    print(f"\n*** GOFOs => Junior Enlisted: {gofo_je_train:+.4f} ***")
    print(f"Direction: {'DAMPENING (-)' if gofo_je_train < 0 else 'AMPLIFYING (+)'}")

    matches_train = (gofo_je_train < 0 and empirical_corr < 0) or (gofo_je_train > 0 and empirical_corr > 0)
    print(f"Match: {'YES - CORRECT!' if matches_train else 'NO - WRONG'}")

    # Out-of-sample forecast
    print(f"\n{'='*80}")
    print("OUT-OF-SAMPLE FORECAST")
    print(f"{'='*80}")

    forecast = vecm_train.predict(steps=len(test_data))
    mae_per_var = np.abs(forecast - test_data.values).mean(axis=0)
    mae_overall = mae_per_var.mean()

    print(f"\nForecast period: {len(test_data)} observations")
    print(f"Overall MAE: {mae_overall:.4f}")
    print(f"GOFOs MAE: {mae_per_var[gofo_idx]:.4f}")
    print(f"Junior Enlisted MAE: {mae_per_var[je_idx]:.4f}")

except Exception as e:
    print(f"\nERROR with training subset: {str(e)}")
    import traceback
    traceback.print_exc()

# Summary comparison
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print(f"\nLAG=6 RESULTS:")
print(f"  Full dataset (38 obs):     GOFOs=>JE = {gofo_je_full:+.4f} ({'CORRECT' if matches_full else 'WRONG'})")
print(f"  Training subset (30 obs):  GOFOs=>JE = {gofo_je_train:+.4f} ({'CORRECT' if matches_train else 'WRONG'})")
print(f"  Out-of-sample MAE: {mae_overall:.4f}")

print(f"\nCOMPARISON WITH PREVIOUS LAGS:")
print(f"  Lag=1 (full):  GOFOs=>JE = +1.4195 (WRONG),  MAE = 0.3863")
print(f"  Lag=2 (full):  GOFOs=>JE = +0.6219 (WRONG),  MAE = ~0.58")
print(f"  Lag=2 (train): GOFOs=>JE = -0.3660 (CORRECT), MAE = 0.5826")
print(f"  Lag=6 (full):  GOFOs=>JE = {gofo_je_full:+.4f} ({'CORRECT' if matches_full else 'WRONG'}),  MAE = N/A")
print(f"  Lag=6 (train): GOFOs=>JE = {gofo_je_train:+.4f} ({'CORRECT' if matches_train else 'WRONG'}), MAE = {mae_overall:.4f}")

print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")

if matches_full and matches_train:
    print("\nLag=6 SUCCESSFULLY captures the relationship in both full and training data!")
    print("The additional short-run dynamics provide enough flexibility.")
elif matches_train and not matches_full:
    print("\nLag=6 shows the same pattern as lag=2:")
    print("  - Correct sign on training subset")
    print("  - Wrong sign on full dataset")
    print("This confirms structural instability in 2017-2024 period.")
elif matches_full and not matches_train:
    print("\nUnexpected: Full dataset correct but training subset wrong.")
    print("This would suggest recent observations stabilize the relationship.")
else:
    print("\nLag=6 FAILS to capture the relationship in both datasets.")
    print("More lags do not help - the VECM framework may be fundamentally inadequate.")
