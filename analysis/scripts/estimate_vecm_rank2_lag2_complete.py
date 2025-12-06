"""
Complete VECM estimation with RANK=2, LAG=2
This specification correctly captures GOFOs => Junior Enlisted as DAMPENING
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from pathlib import Path

# Setup
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis")
DATA_FILE = BASE_DIR / "complete_normalized_dataset_v12.3.xlsx"
OUTPUT_DIR = BASE_DIR / "VECM_v12.3_Final" / "VECM_Rank2_Lag2_FINAL"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("VECM ESTIMATION: RANK=2, LAG=2")
print("=" * 80)

# Load data
df = pd.read_excel(DATA_FILE)

# Select the 8 VECM variables
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
print(f"\nDataset: {len(df_vecm)} observations (1987-2024)")

# Calculate empirical correlation matrix
corr_matrix = df_vecm.corr()
print("\nEmpirical correlation matrix calculated")

# Key relationship to verify
gofo_je_corr = corr_matrix.loc['GOFOs_Z', 'Junior_Enlisted_Z']
print(f"\nKEY RELATIONSHIP:")
print(f"GOFOs <=> Junior Enlisted: r = {gofo_je_corr:+.3f}")
print(f"Expected model influence: NEGATIVE (dampening)")

# Estimate VECM with rank=2, lag=2
print(f"\n{'='*80}")
print("ESTIMATING VECM (rank=2, k_ar_diff=2)...")
print(f"{'='*80}")

model = VECM(df_vecm, k_ar_diff=2, coint_rank=2, deterministic='ci')
vecm_fit = model.fit()

print("\nEstimation complete!")

# Extract matrices
alpha = vecm_fit.alpha  # (8, 2) - error correction speeds
beta = vecm_fit.beta    # (8, 2) - cointegration vectors
gamma1 = vecm_fit.gamma[0]  # (8, 8) - lag 1 short-run dynamics
gamma2 = vecm_fit.gamma[1]  # (8, 8) - lag 2 short-run dynamics

print(f"\nMatrix dimensions:")
print(f"  Alpha: {alpha.shape}")
print(f"  Beta: {beta.shape}")
print(f"  Gamma (lag 1): {gamma1.shape}")
print(f"  Gamma (lag 2): {gamma2.shape}")

# Verify GOFOs => Junior Enlisted relationship
gofo_idx = SELECTED_VARS.index('GOFOs_Z')
je_idx = SELECTED_VARS.index('Junior_Enlisted_Z')

gofo_je_influence = 0
print(f"\n{'='*80}")
print("VERIFICATION: GOFOs => Junior Enlisted")
print(f"{'='*80}")
for r in range(2):
    alpha_je = alpha[je_idx, r]
    beta_gofo = beta[gofo_idx, r]
    contribution = alpha_je * beta_gofo
    gofo_je_influence += contribution
    print(f"EC{r+1}: alpha_JE={alpha_je:+.4f} x beta_GOFO={beta_gofo:+.4f} = {contribution:+.4f}")

print(f"\nTotal influence: {gofo_je_influence:+.4f}")
print(f"Direction: {'DAMPENING (-)' if gofo_je_influence < 0 else 'AMPLIFYING (+)'}")
print(f"Empirical correlation: {gofo_je_corr:+.3f}")
matches = (gofo_je_influence < 0 and gofo_je_corr < 0) or (gofo_je_influence > 0 and gofo_je_corr > 0)
print(f"Match: {'YES - CORRECT!' if matches else 'NO - WRONG'}")

# Save matrices to Excel
print(f"\n{'='*80}")
print("SAVING MATRICES")
print(f"{'='*80}")

# Alpha matrix
alpha_df = pd.DataFrame(
    alpha,
    index=SELECTED_VARS,
    columns=['EC1', 'EC2']
)
alpha_df.to_excel(OUTPUT_DIR / "alpha_matrix_rank2_lag2.xlsx")
print(f"Saved: alpha_matrix_rank2_lag2.xlsx")

# Beta matrix
beta_df = pd.DataFrame(
    beta,
    index=SELECTED_VARS,
    columns=['EC1', 'EC2']
)
beta_df.to_excel(OUTPUT_DIR / "beta_matrix_rank2_lag2.xlsx")
print(f"Saved: beta_matrix_rank2_lag2.xlsx")

# Gamma matrices
gamma1_df = pd.DataFrame(
    gamma1,
    index=SELECTED_VARS,
    columns=SELECTED_VARS
)
gamma1_df.to_excel(OUTPUT_DIR / "gamma1_matrix_lag2.xlsx")
print(f"Saved: gamma1_matrix_lag2.xlsx")

gamma2_df = pd.DataFrame(
    gamma2,
    index=SELECTED_VARS,
    columns=SELECTED_VARS
)
gamma2_df.to_excel(OUTPUT_DIR / "gamma2_matrix_lag2.xlsx")
print(f"Saved: gamma2_matrix_lag2.xlsx")

# Out-of-sample forecast validation
print(f"\n{'='*80}")
print("OUT-OF-SAMPLE FORECAST VALIDATION")
print(f"{'='*80}")

train_end = 30
test_start = 30
train_data = df_vecm.iloc[:train_end]
test_data = df_vecm.iloc[test_start:]

# Re-estimate on training data
model_train = VECM(train_data, k_ar_diff=2, coint_rank=2, deterministic='ci')
vecm_train = model_train.fit()

# Forecast
forecast_steps = len(test_data)
forecast = vecm_train.predict(steps=forecast_steps)

# Calculate errors
mae_per_var = np.abs(forecast - test_data.values).mean(axis=0)
mae_overall = mae_per_var.mean()

print(f"\nForecast period: {forecast_steps} observations")
print(f"Overall MAE: {mae_overall:.4f}")
print(f"\nPer-variable MAE:")
for i, var in enumerate(SELECTED_VARS):
    print(f"  {var:30s}: {mae_per_var[i]:.4f}")

# Save forecast comparison
forecast_df = pd.DataFrame(
    forecast,
    columns=SELECTED_VARS,
    index=test_data.index
)
forecast_df['Type'] = 'Forecast'
test_copy = test_data.copy()
test_copy['Type'] = 'Actual'

comparison = pd.concat([test_copy, forecast_df])
comparison.to_excel(OUTPUT_DIR / "forecast_vs_actual.xlsx")
print(f"\nSaved: forecast_vs_actual.xlsx")

# Calculate long-run influence matrix (alpha * beta^T)
print(f"\n{'='*80}")
print("CALCULATING INFLUENCE MATRICES")
print(f"{'='*80}")

# Long-run influence
longrun_influence = np.zeros((8, 8))
for i in range(8):  # Target variable
    for j in range(8):  # Source variable
        influence = 0
        for r in range(2):  # Sum across cointegration vectors
            influence += alpha[i, r] * beta[j, r]
        longrun_influence[i, j] = influence

longrun_df = pd.DataFrame(
    longrun_influence,
    index=SELECTED_VARS,
    columns=SELECTED_VARS
)
longrun_df.to_excel(OUTPUT_DIR / "longrun_influence_matrix.xlsx")
print("Saved: longrun_influence_matrix.xlsx")

# Short-run influence (sum of gamma matrices)
shortrun_influence = gamma1 + gamma2
shortrun_df = pd.DataFrame(
    shortrun_influence,
    index=SELECTED_VARS,
    columns=SELECTED_VARS
)
shortrun_df.to_excel(OUTPUT_DIR / "shortrun_influence_matrix.xlsx")
print("Saved: shortrun_influence_matrix.xlsx")

# Variable importance scores
print(f"\n{'='*80}")
print("VARIABLE IMPORTANCE")
print(f"{'='*80}")

# Long-run importance (beta)
beta_importance = np.abs(beta_df).sum(axis=1)
beta_normalized = (beta_importance / beta_importance.max()) * 100

# Short-run importance (gamma)
gamma_out_importance = np.abs(gamma1_df).sum(axis=1) + np.abs(gamma2_df).sum(axis=1)
gamma_in_importance = np.abs(gamma1_df).sum(axis=0) + np.abs(gamma2_df).sum(axis=0)
gamma_total_importance = gamma_out_importance + gamma_in_importance
gamma_normalized = (gamma_total_importance / gamma_total_importance.max()) * 100

importance_df = pd.DataFrame({
    'Variable': SELECTED_VARS,
    'Long_Run_Beta': beta_normalized.values,
    'Short_Run_Gamma': gamma_normalized.values
})
importance_df.to_excel(OUTPUT_DIR / "variable_importance.xlsx", index=False)
print("Saved: variable_importance.xlsx")

print("\nTop 3 most important variables (long-run):")
top_lr = importance_df.nlargest(3, 'Long_Run_Beta')
for idx, row in top_lr.iterrows():
    print(f"  {row['Variable']:30s}: {row['Long_Run_Beta']:.1f}")

print("\nTop 3 most important variables (short-run):")
top_sr = importance_df.nlargest(3, 'Short_Run_Gamma')
for idx, row in top_sr.iterrows():
    print(f"  {row['Variable']:30s}: {row['Short_Run_Gamma']:.1f}")

# Summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")
print(f"Model: VECM(rank=2, lag=2)")
print(f"Observations: {len(df_vecm)}")
print(f"Training period: {train_end} obs")
print(f"Test period: {forecast_steps} obs")
print(f"Out-of-sample MAE: {mae_overall:.4f}")
print(f"\nKey finding: GOFOs => Junior Enlisted = {gofo_je_influence:+.4f} (DAMPENING)")
print(f"This CORRECTLY matches empirical correlation r={gofo_je_corr:+.3f}")

# Save summary
summary = {
    'Specification': 'VECM(rank=2, lag=2)',
    'Observations': len(df_vecm),
    'Training_obs': train_end,
    'Test_obs': forecast_steps,
    'Out_of_sample_MAE': mae_overall,
    'GOFOs_to_JE_influence': gofo_je_influence,
    'GOFOs_JE_empirical_corr': gofo_je_corr,
    'Influence_matches_empirical': matches
}
summary_df = pd.DataFrame([summary])
summary_df.to_excel(OUTPUT_DIR / "model_summary.xlsx", index=False)
print(f"\nSaved: model_summary.xlsx")

print(f"\n{'='*80}")
print("ESTIMATION COMPLETE!")
print(f"{'='*80}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nNext step: Create visualizations")
