"""
VECM rank=2, lag=2 with empirical sign validation and correction
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
from pathlib import Path

# Setup
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis")
DATA_FILE = BASE_DIR / "complete_normalized_dataset_v12.3.xlsx"
OUTPUT_DIR = BASE_DIR / "VECM_v12.3_Final" / "VECM_Rank2_Lag2_FINAL"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("VECM ESTIMATION: RANK=2, LAG=2 WITH SIGN CORRECTION")
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

# Calculate empirical correlations
corr_matrix = df_vecm.corr()

# Estimate VECM
print("\nEstimating VECM (rank=2, k_ar_diff=2)...")
model = VECM(df_vecm, k_ar_diff=2, coint_rank=2, deterministic='ci')
vecm_fit = model.fit()

# Extract alpha and beta
alpha_original = vecm_fit.alpha.copy()
beta_original = vecm_fit.beta.copy()

print(f"\nOriginal matrices:")
print(f"  Alpha: {alpha_original.shape}")
print(f"  Beta: {beta_original.shape}")

# Check GOFOs => JE relationship BEFORE correction
gofo_idx = SELECTED_VARS.index('GOFOs_Z')
je_idx = SELECTED_VARS.index('Junior_Enlisted_Z')

original_influence = sum([alpha_original[je_idx, r] * beta_original[gofo_idx, r] for r in range(2)])
empirical_corr = corr_matrix.loc['GOFOs_Z', 'Junior_Enlisted_Z']

print(f"\n{'='*80}")
print("BEFORE SIGN CORRECTION")
print(f"{'='*80}")
print(f"GOFOs => JE influence: {original_influence:+.4f}")
print(f"Empirical correlation: {empirical_corr:+.3f}")
matches_before = (original_influence < 0 and empirical_corr < 0) or (original_influence > 0 and empirical_corr > 0)
print(f"Match: {'YES' if matches_before else 'NO - NEED CORRECTION'}")

# Apply sign correction using empirical correlations
print(f"\n{'='*80}")
print("APPLYING SIGN CORRECTION")
print(f"{'='*80}")

alpha_corrected = alpha_original.copy()
beta_corrected = beta_original.copy()

for vec_idx in range(2):
    print(f"\nEC{vec_idx+1}:")

    # Find reference variable (largest |beta|)
    ref_idx = np.argmax(np.abs(beta_original[:, vec_idx]))
    ref_var = SELECTED_VARS[ref_idx]
    ref_beta_sign = np.sign(beta_original[ref_idx, vec_idx])

    print(f"  Reference: {ref_var} (beta={beta_original[ref_idx, vec_idx]:+.4f})")

    # Check each variable
    correct_signs = 0
    wrong_signs = 0

    for i, var in enumerate(SELECTED_VARS):
        if i == ref_idx:
            continue

        empirical_corr_to_ref = corr_matrix.loc[ref_var, var]
        beta_sign = np.sign(beta_original[i, vec_idx])

        # Expected sign based on correlation with reference
        if empirical_corr_to_ref > 0:
            expected_sign = ref_beta_sign
        else:
            expected_sign = -ref_beta_sign

        if beta_sign == expected_sign or beta_original[i, vec_idx] == 0:
            correct_signs += 1
        else:
            wrong_signs += 1

    print(f"  Correct signs: {correct_signs}, Wrong signs: {wrong_signs}")

    # Flip if majority wrong
    if wrong_signs > correct_signs:
        print(f"  => FLIPPING vector")
        beta_corrected[:, vec_idx] *= -1
        alpha_corrected[:, vec_idx] *= -1
    else:
        print(f"  => Keeping vector as-is")

# Check GOFOs => JE relationship AFTER correction
corrected_influence = sum([alpha_corrected[je_idx, r] * beta_corrected[gofo_idx, r] for r in range(2)])

print(f"\n{'='*80}")
print("AFTER SIGN CORRECTION")
print(f"{'='*80}")

for r in range(2):
    alpha_je = alpha_corrected[je_idx, r]
    beta_gofo = beta_corrected[gofo_idx, r]
    contribution = alpha_je * beta_gofo
    print(f"EC{r+1}: alpha_JE={alpha_je:+.4f} x beta_GOFO={beta_gofo:+.4f} = {contribution:+.4f}")

print(f"\nGOFOs => JE influence: {corrected_influence:+.4f}")
print(f"Direction: {'DAMPENING (-)' if corrected_influence < 0 else 'AMPLIFYING (+)'}")
print(f"Empirical correlation: {empirical_corr:+.3f}")
matches_after = (corrected_influence < 0 and empirical_corr < 0) or (corrected_influence > 0 and empirical_corr > 0)
print(f"Match: {'YES - CORRECT!' if matches_after else 'NO - STILL WRONG'}")

# Save matrices
print(f"\n{'='*80}")
print("SAVING MATRICES")
print(f"{'='*80}")

alpha_df = pd.DataFrame(alpha_corrected, index=SELECTED_VARS, columns=['EC1', 'EC2'])
alpha_df.to_excel(OUTPUT_DIR / "alpha_matrix_rank2_lag2.xlsx")
print("Saved: alpha_matrix_rank2_lag2.xlsx")

beta_df = pd.DataFrame(beta_corrected, index=SELECTED_VARS, columns=['EC1', 'EC2'])
beta_df.to_excel(OUTPUT_DIR / "beta_matrix_rank2_lag2.xlsx")
print("Saved: beta_matrix_rank2_lag2.xlsx")

# Calculate long-run influence matrix
longrun_influence = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        longrun_influence[i, j] = sum([alpha_corrected[i, r] * beta_corrected[j, r] for r in range(2)])

longrun_df = pd.DataFrame(longrun_influence, index=SELECTED_VARS, columns=SELECTED_VARS)
longrun_df.to_excel(OUTPUT_DIR / "longrun_influence_matrix.xlsx")
print("Saved: longrun_influence_matrix.xlsx")

# Out-of-sample validation
print(f"\n{'='*80}")
print("OUT-OF-SAMPLE VALIDATION")
print(f"{'='*80}")

train_end = 30
train_data = df_vecm.iloc[:train_end]
test_data = df_vecm.iloc[train_end:]

# Estimate on training data (no sign correction for forecast)
model_train = VECM(train_data, k_ar_diff=2, coint_rank=2, deterministic='ci')
vecm_train = model_train.fit()

# Forecast
forecast = vecm_train.predict(steps=len(test_data))
mae_per_var = np.abs(forecast - test_data.values).mean(axis=0)
mae_overall = mae_per_var.mean()

print(f"\nTraining: {len(train_data)} obs, Test: {len(test_data)} obs")
print(f"Overall MAE: {mae_overall:.4f}")
print(f"GOFOs MAE: {mae_per_var[gofo_idx]:.4f}")
print(f"Junior Enlisted MAE: {mae_per_var[je_idx]:.4f}")

# Variable importance
beta_importance = np.abs(beta_df).sum(axis=1)
beta_normalized = (beta_importance / beta_importance.max()) * 100

importance_df = pd.DataFrame({
    'Variable': SELECTED_VARS,
    'Long_Run_Importance': beta_normalized.values
})
importance_df.to_excel(OUTPUT_DIR / "variable_importance.xlsx", index=False)
print("\nSaved: variable_importance.xlsx")

# Summary
summary = {
    'Specification': 'VECM(rank=2, lag=2)',
    'Observations': len(df_vecm),
    'Out_of_sample_MAE': mae_overall,
    'GOFOs_to_JE_original': original_influence,
    'GOFOs_to_JE_corrected': corrected_influence,
    'Empirical_correlation': empirical_corr,
    'Matches_before_correction': matches_before,
    'Matches_after_correction': matches_after
}
summary_df = pd.DataFrame([summary])
summary_df.to_excel(OUTPUT_DIR / "model_summary.xlsx", index=False)
print("Saved: model_summary.xlsx")

print(f"\n{'='*80}")
print("COMPLETE!")
print(f"{'='*80}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nKey result: GOFOs => JE = {corrected_influence:+.4f} ({'CORRECT' if matches_after else 'WRONG'})")
