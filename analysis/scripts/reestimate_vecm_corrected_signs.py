"""
Re-estimate VECM with corrected beta signs
Validate against data and calculate out-of-sample MAPE
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis")
DATA_FILE = BASE_DIR / "complete_normalized_dataset_v12.3.xlsx"
OUTPUT_DIR = BASE_DIR / "VECM_v12.3_Final" / "VECM_Rank2_CORRECTED"
OUTPUT_DIR.mkdir(exist_ok=True)

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
print("RE-ESTIMATING VECM WITH CORRECTED BETA SIGNS")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
df = pd.read_excel(DATA_FILE)
df_clean = df.dropna(subset=SELECTED_VARS)
data = df_clean[SELECTED_VARS].values
years = df_clean['FY'].values

print(f"    Data shape: {data.shape}")
print(f"    Years: {years.min():.0f} to {years.max():.0f}")

# Split into train and test
train_size = int(len(data) * 0.8)  # 80% train, 20% test
train_data = data[:train_size]
test_data = data[train_size:]
train_years = years[:train_size]
test_years = years[train_size:]

print(f"\n[2] Data split:")
print(f"    Training: {train_years.min():.0f} to {train_years.max():.0f} (n={len(train_data)})")
print(f"    Testing: {test_years.min():.0f} to {test_years.max():.0f} (n={len(test_data)})")

# Estimate VECM
print("\n[3] Estimating VECM (rank=2, k_ar_diff=1)...")
model = VECM(train_data, k_ar_diff=1, coint_rank=2, deterministic='ci')
vecm_result = model.fit()

print("    VECM estimation complete")

# Extract matrices
alpha_original = vecm_result.alpha
beta_original = vecm_result.beta
gamma = vecm_result.gamma

print(f"\n[4] Original matrices:")
print(f"    Alpha: {alpha_original.shape}")
print(f"    Beta: {beta_original.shape}")
print(f"    Gamma: {gamma.shape}")

# Calculate empirical correlations for validation
print("\n[5] Checking beta signs against empirical correlations...")
corr_matrix = pd.DataFrame(train_data, columns=SELECTED_VARS).corr()

# For each cointegration vector, check if signs match empirical reality
alpha_corrected = alpha_original.copy()
beta_corrected = beta_original.copy()

for vec_idx in range(2):
    print(f"\n[5.{vec_idx+1}] Checking EC{vec_idx+1}...")

    # Find reference variable (one with largest absolute beta)
    abs_betas = np.abs(beta_original[:, vec_idx])
    ref_idx = np.argmax(abs_betas)
    ref_var = SELECTED_VARS[ref_idx]
    ref_beta_sign = np.sign(beta_original[ref_idx, vec_idx])

    print(f"    Reference variable: {ref_var} (beta={beta_original[ref_idx, vec_idx]:+.4f})")

    # Check how many variables have wrong signs
    wrong_signs = 0
    correct_signs = 0

    for i, var in enumerate(SELECTED_VARS):
        if i == ref_idx or abs(beta_original[i, vec_idx]) < 0.01:
            continue

        empirical_corr = corr_matrix.loc[ref_var, var]
        beta_val = beta_original[i, vec_idx]
        beta_sign = np.sign(beta_val)

        # Expected sign based on correlation
        if empirical_corr > 0:
            expected_sign = ref_beta_sign
        else:
            expected_sign = -ref_beta_sign

        if beta_sign == expected_sign:
            correct_signs += 1
        else:
            wrong_signs += 1
            print(f"      {var}: corr={empirical_corr:+.3f}, beta={beta_val:+.4f} [WRONG SIGN]")

    # If majority of signs are wrong, flip the entire vector
    if wrong_signs > correct_signs:
        print(f"    >>> Flipping EC{vec_idx+1} (wrong_signs={wrong_signs} > correct_signs={correct_signs})")
        beta_corrected[:, vec_idx] *= -1
        alpha_corrected[:, vec_idx] *= -1
    else:
        print(f"    >>> Keeping EC{vec_idx+1} (correct_signs={correct_signs} >= wrong_signs={wrong_signs})")

# Re-verify corrected beta signs
print("\n[6] Verifying corrected beta signs...")
for vec_idx in range(2):
    abs_betas = np.abs(beta_corrected[:, vec_idx])
    ref_idx = np.argmax(abs_betas)
    ref_var = SELECTED_VARS[ref_idx]
    ref_beta_sign = np.sign(beta_corrected[ref_idx, vec_idx])

    print(f"\n[6.{vec_idx+1}] EC{vec_idx+1} verification:")
    print(f"    Reference: {ref_var} (beta={beta_corrected[ref_idx, vec_idx]:+.4f})")

    all_correct = True
    for i, var in enumerate(SELECTED_VARS):
        if i == ref_idx or abs(beta_corrected[i, vec_idx]) < 0.01:
            continue

        empirical_corr = corr_matrix.loc[ref_var, var]
        beta_val = beta_corrected[i, vec_idx]
        beta_sign = np.sign(beta_val)

        if empirical_corr > 0:
            expected_sign = ref_beta_sign
        else:
            expected_sign = -ref_beta_sign

        match = "[OK]" if beta_sign == expected_sign else "[BAD]"
        if beta_sign != expected_sign:
            all_correct = False

        print(f"      {var:30s} corr={empirical_corr:+.3f}  beta={beta_val:+.4f}  {match}")

    if all_correct:
        print(f"    >>> All signs CORRECT for EC{vec_idx+1}!")

# Save corrected matrices
print("\n[7] Saving corrected matrices...")
alpha_df = pd.DataFrame(alpha_corrected, index=SELECTED_VARS, columns=['EC1', 'EC2'])
beta_df = pd.DataFrame(beta_corrected, index=SELECTED_VARS, columns=['EC1', 'EC2'])
gamma_df = pd.DataFrame(gamma, index=SELECTED_VARS, columns=SELECTED_VARS)

alpha_df.to_excel(OUTPUT_DIR / "alpha_matrix_rank2_CORRECTED.xlsx")
beta_df.to_excel(OUTPUT_DIR / "beta_matrix_rank2_CORRECTED.xlsx")
gamma_df.to_excel(OUTPUT_DIR / "gamma_matrix_rank2_CORRECTED.xlsx")

print("    Matrices saved")

# Calculate long-run influence with corrected signs
print("\n[8] Calculating long-run influence with corrected signs...")
longrun_influence = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))
signed_direction = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))

for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        signed_sum = 0
        unsigned_sum = 0
        for r in range(2):
            alpha_i = alpha_corrected[i, r]
            beta_j = beta_corrected[j, r]
            influence = alpha_i * beta_j
            signed_sum += influence
            unsigned_sum += abs(influence)

        longrun_influence[i, j] = unsigned_sum
        signed_direction[i, j] = np.sign(signed_sum)

longrun_df = pd.DataFrame(longrun_influence, index=SELECTED_VARS, columns=SELECTED_VARS)
longrun_df.to_excel(OUTPUT_DIR / "longrun_influence_rank2_CORRECTED.xlsx")

print("    Long-run influence calculated")

# OUT-OF-SAMPLE FORECASTING & MAPE
print("\n[9] Out-of-sample forecasting...")

# Forecast test period
forecast_steps = len(test_data)
forecasts = []

# Rolling forecast: use training data + actual values up to t-1 to predict t
for step in range(forecast_steps):
    # Use all training data plus actual test data up to current step
    historical_data = np.vstack([train_data, test_data[:step]]) if step > 0 else train_data

    # Re-fit model on historical data
    temp_model = VECM(historical_data, k_ar_diff=1, coint_rank=2, deterministic='ci')
    temp_result = temp_model.fit()

    # Forecast 1 step ahead
    forecast = temp_result.predict(steps=1)
    forecasts.append(forecast[0])

forecasts = np.array(forecasts)

print(f"    Forecasted {forecast_steps} steps")

# Calculate MAPE for each variable
print("\n[10] Calculating MAPE (Mean Absolute Percentage Error)...")
print("\n" + "-" * 80)
print(f"{'Variable':<30s} {'MAPE (%)':<15s} {'MAE':<15s}")
print("-" * 80)

mape_values = {}
for i, var in enumerate(SELECTED_VARS):
    actual = test_data[:, i]
    predicted = forecasts[:, i]

    # MAPE (handle division by zero)
    # For z-scored data, MAPE might not be meaningful if actual values cross zero
    # So also calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(actual - predicted))

    # MAPE: only for non-zero actuals
    non_zero_mask = np.abs(actual) > 0.01
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
    else:
        mape = np.nan

    mape_values[var] = {'MAPE': mape, 'MAE': mae}

    print(f"{var:<30s} {mape:>10.2f}%     {mae:>10.4f}")

print("-" * 80)
avg_mape = np.nanmean([v['MAPE'] for v in mape_values.values()])
avg_mae = np.mean([v['MAE'] for v in mape_values.values()])
print(f"{'AVERAGE':<30s} {avg_mape:>10.2f}%     {avg_mae:>10.4f}")
print("-" * 80)

# Save forecast results
forecast_df = pd.DataFrame(
    forecasts,
    columns=SELECTED_VARS,
    index=test_years
)
forecast_df['Type'] = 'Forecast'

actual_df = pd.DataFrame(
    test_data,
    columns=SELECTED_VARS,
    index=test_years
)
actual_df['Type'] = 'Actual'

comparison_df = pd.concat([actual_df, forecast_df])
comparison_df.to_excel(OUTPUT_DIR / "out_of_sample_forecast_comparison.xlsx")

print(f"\n[11] Forecast comparison saved")

# Summary
print("\n" + "=" * 80)
print("WHY DID WE GET THE WRONG SIGNS?")
print("=" * 80)
print("""
The Johansen cointegration test finds eigenvectors (beta) that define equilibrium
relationships. These eigenvectors are arbitrary up to sign - you can multiply the
entire vector by -1 and it's still mathematically valid.

The statsmodels VECM implementation automatically normalizes the beta matrix by
setting one coefficient to 1.0 in each cointegration vector. However, it doesn't
check whether the resulting signs match the empirical correlations in the data.

In our case, the normalization resulted in beta signs that were opposite to the
actual directional relationships, making the heatmaps and interpretations misleading.

The fix: We check each cointegration vector's signs against empirical correlations,
and if the majority of signs are wrong, we flip the entire vector (both beta and
the corresponding alpha column).
""")

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nCorrected VECM saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. alpha_matrix_rank2_CORRECTED.xlsx")
print("  2. beta_matrix_rank2_CORRECTED.xlsx")
print("  3. gamma_matrix_rank2_CORRECTED.xlsx")
print("  4. longrun_influence_rank2_CORRECTED.xlsx")
print("  5. out_of_sample_forecast_comparison.xlsx")
print(f"\nOut-of-sample performance:")
print(f"  Average MAPE: {avg_mape:.2f}%")
print(f"  Average MAE: {avg_mae:.4f}")
print("=" * 80)
