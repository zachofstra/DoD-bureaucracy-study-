"""
VECM Re-estimation with Rank=1 (OPTIMAL)
=========================================
Based on robustness check showing rank=1 outperforms rank=6 on:
- Out-of-sample prediction (51% vs 63% MAPE)
- BIC (115 vs 259)
- Overall criteria (17/24 vs 1/24 points)

This script re-estimates the entire VECM analysis with rank=1 and regenerates:
1. Alpha/Beta/Gamma matrices
2. Influence heatmaps (short-run and long-run)
3. Network diagrams
4. Model diagnostics
5. Executive summary outputs
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
OUTPUT_DIR = BASE_DIR / "VECM_Rank1_Final_Executive_Summary"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

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

DISPLAY_NAMES = {
    'Junior_Enlisted_Z': 'Junior\nEnlisted',
    'Company_Grade_Officers_Z': 'Company\nGrade',
    'Field_Grade_Officers_Z': 'Field\nGrade',
    'GOFOs_Z': 'GOFOs',
    'Warrant_Officers_Z': 'Warrant\nOfficers',
    'Policy_Count_Log': 'Policy\nCount',
    'Total_PAS_Z': 'Total\nPAS',
    'FOIA_Simple_Days_Z': 'FOIA\nDays'
}

print("=" * 80)
print("VECM RE-ESTIMATION WITH RANK=1 (OPTIMAL)")
print("=" * 80)
print("\nRank=1 selected based on robustness check:")
print("  - Best out-of-sample prediction (51% MAPE)")
print("  - Lowest BIC (115 vs 259 for rank=6)")
print("  - Won 5/8 evaluation criteria")
print("\n" + "=" * 80)

# Load data
print("\n[1] LOADING DATA...")
data_file = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\complete_normalized_dataset_v12.3.xlsx")
df = pd.read_excel(data_file)
df.columns = df.columns.str.strip()
data = df[SELECTED_VARS].dropna().copy()

print(f"    Data shape: {data.shape[0]} observations x {data.shape[1]} variables")
print(f"    Date range: 1987-2024")

# Estimate VECM with rank=1
print("\n[2] ESTIMATING VECM WITH RANK=1...")
vecm = VECM(data, k_ar_diff=1, coint_rank=1, deterministic='nc')
vecm_result = vecm.fit()

print(f"    Cointegration rank: 1")
print(f"    Log-likelihood: {vecm_result.llf:.2f}")
print(f"    Observations used: {len(data)}")

# Extract matrices
print("\n[3] EXTRACTING COEFFICIENT MATRICES...")

# Alpha (error correction speeds): 8x1
alpha = vecm_result.alpha
alpha_df = pd.DataFrame(alpha,
                        index=SELECTED_VARS,
                        columns=['EC1'])

# Beta (cointegration vector): 8x1
beta = vecm_result.beta
beta_df = pd.DataFrame(beta,
                       index=SELECTED_VARS,
                       columns=['EC1'])

# Gamma (short-run dynamics): 8x8
gamma = vecm_result.gamma
gamma_df = pd.DataFrame(gamma,
                        index=SELECTED_VARS,
                        columns=SELECTED_VARS)

print(f"    Alpha (error correction): {alpha_df.shape}")
print(f"    Beta (cointegration vector): {beta_df.shape}")
print(f"    Gamma (short-run VAR): {gamma_df.shape}")

# Save matrices
print("\n[4] SAVING COEFFICIENT MATRICES...")
alpha_df.to_excel(OUTPUT_DIR / "alpha_matrix_rank1.xlsx")
beta_df.to_excel(OUTPUT_DIR / "beta_matrix_rank1.xlsx")
gamma_df.to_excel(OUTPUT_DIR / "gamma_matrix_rank1.xlsx")
print(f"    Saved to: {OUTPUT_DIR}")

# Calculate long-run influence (rank=1: just alpha * beta for single vector)
print("\n[5] CALCULATING LONG-RUN INFLUENCE...")
longrun_influence = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))

for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        # With rank=1, only one cointegration vector
        alpha_i = alpha_df.iloc[i, 0]  # Error correction speed for variable i
        beta_j = beta_df.iloc[j, 0]    # Weight of variable j in equilibrium
        longrun_influence[i, j] = abs(alpha_i * beta_j)

longrun_df = pd.DataFrame(longrun_influence,
                          index=SELECTED_VARS,
                          columns=SELECTED_VARS)

longrun_df.to_excel(OUTPUT_DIR / "longrun_influence_rank1.xlsx")
print(f"    Long-run influence calculated (unsigned magnitude)")

# Calculate signed direction for coloring
signed_direction = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))
for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        alpha_i = alpha_df.iloc[i, 0]
        beta_j = beta_df.iloc[j, 0]
        signed_direction[i, j] = np.sign(alpha_i * beta_j)

# Create influence heatmaps
print("\n[6] CREATING INFLUENCE HEATMAPS...")

# Short-run heatmap (Gamma)
fig, ax = plt.subplots(figsize=(12, 10))
gamma_values = gamma_df.values
display_names = [DISPLAY_NAMES[var] for var in SELECTED_VARS]

sns.heatmap(gamma_values,
            annot=np.abs(gamma_values),
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Coefficient Value'},
            xticklabels=display_names,
            yticklabels=display_names,
            linewidths=0.5,
            linecolor='gray',
            ax=ax)

ax.set_title('SHORT-RUN DYNAMICS (Gamma)\nYear-to-year VAR effects (rank=1)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('From Variable (t-1)', fontsize=12, fontweight='bold')
ax.set_ylabel('To Variable (equilibrium deviation)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_shortrun_influence_rank1.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Short-run heatmap saved")

# Long-run heatmap (Alpha x Beta) with directional coloring
fig, ax = plt.subplots(figsize=(12, 10))

# Flip sign for coloring to match network diagrams (RED=amplifying, BLUE=dampening)
signed_magnitude_flipped = longrun_influence * signed_direction * (-1)

sns.heatmap(signed_magnitude_flipped,
            annot=longrun_influence,  # Show magnitude
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Direction (RED=Amplifying, BLUE=Dampening)'},
            xticklabels=display_names,
            yticklabels=display_names,
            linewidths=0.5,
            linecolor='gray',
            ax=ax)

ax.set_title('LONG-RUN INFLUENCE (Error Correction)\nMagnitude with directional coloring (rank=1, matches network diagrams)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('From Variable (equilibrium deviation)', fontsize=12, fontweight='bold')
ax.set_ylabel('To Variable (adjustment)', fontsize=12, fontweight='bold')

# Add legend explaining interpretation
legend_text = "Numbers = MAGNITUDE (sum of |alpha x beta|)\nColor = DIRECTION\n  RED = Amplifying (+)\n  BLUE = Dampening (-)"
ax.text(1.15, 0.5, legend_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_longrun_influence_rank1.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Long-run heatmap saved")

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Left: Short-run
ax1 = axes[0]
sns.heatmap(gamma_values,
            annot=np.abs(gamma_values),
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Coefficient Value'},
            xticklabels=display_names,
            yticklabels=display_names,
            linewidths=0.5,
            linecolor='gray',
            ax=ax1)
ax1.set_title('SHORT-RUN DYNAMICS (Gamma)\nYear-to-year effects',
              fontsize=13, fontweight='bold')
ax1.set_xlabel('From Variable (t-1)', fontsize=11, fontweight='bold')
ax1.set_ylabel('To Variable', fontsize=11, fontweight='bold')

# Right: Long-run
ax2 = axes[1]
sns.heatmap(signed_magnitude_flipped,
            annot=longrun_influence,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Direction (RED=+, BLUE=-)'},
            xticklabels=display_names,
            yticklabels=display_names,
            linewidths=0.5,
            linecolor='gray',
            ax=ax2)
ax2.set_title('LONG-RUN DYNAMICS (Error Correction)\nMagnitude with directional coloring (MATCHES NETWORK DIAGRAMS)',
              fontsize=13, fontweight='bold')
ax2.set_xlabel('From Variable (deviation)', fontsize=11, fontweight='bold')
ax2.set_ylabel('To Variable (adjustment)', fontsize=11, fontweight='bold')

fig.suptitle('VECM INFLUENCE COMPARISON: Short-Run vs Long-Run (RANK=1, OPTIMAL)\n(Magnitude values, directional coloring - MATCHES NETWORK DIAGRAMS)',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_influence_comparison_rank1.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Comparison plot saved")

# Model diagnostics
print("\n[7] RUNNING MODEL DIAGNOSTICS...")

# Out-of-sample validation
train_data = data.iloc[:-5]
test_data = data.iloc[-5:]

vecm_train = VECM(train_data, k_ar_diff=1, coint_rank=1, deterministic='nc')
vecm_train_result = vecm_train.fit()
forecast = vecm_train_result.predict(steps=5)

errors = test_data.values - forecast
rmse = np.sqrt(np.mean(errors**2))
mae = np.mean(np.abs(errors))

mape_values = []
for i in range(test_data.shape[0]):
    for j in range(test_data.shape[1]):
        actual = test_data.values[i, j]
        if abs(actual) > 0.01:
            mape_values.append(abs(errors[i, j] / actual) * 100)
mape = np.mean(mape_values)

print(f"    Out-of-sample (2020-2024):")
print(f"      RMSE: {rmse:.4f}")
print(f"      MAE:  {mae:.4f}")
print(f"      MAPE: {mape:.2f}%")

# Residual stationarity
residuals = vecm_result.resid
stationary_count = 0
residual_tests = []

for var_idx, var_name in enumerate(SELECTED_VARS):
    adf_result = adfuller(residuals[:, var_idx], autolag='AIC')
    is_stationary = adf_result[1] < 0.05
    if is_stationary:
        stationary_count += 1
    residual_tests.append({
        'Variable': var_name,
        'ADF_Statistic': adf_result[0],
        'p_value': adf_result[1],
        'Stationary': is_stationary
    })

residual_df = pd.DataFrame(residual_tests)
residual_df.to_excel(OUTPUT_DIR / "residual_stationarity_rank1.xlsx", index=False)

print(f"    Residual stationarity: {stationary_count}/{len(SELECTED_VARS)} variables ({100*stationary_count/len(SELECTED_VARS):.1f}%)")

# Granger causality (from long-run influence)
print("\n[8] IDENTIFYING KEY RELATIONSHIPS...")

# Top 5 long-run influences
longrun_flat = []
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:  # Exclude diagonal
            longrun_flat.append({
                'From': var_from,
                'To': var_to,
                'Magnitude': longrun_influence[i, j],
                'Direction': 'Amplifying' if signed_direction[i, j] > 0 else 'Dampening'
            })

longrun_ranked = pd.DataFrame(longrun_flat).sort_values('Magnitude', ascending=False)
longrun_ranked.to_excel(OUTPUT_DIR / "longrun_relationships_ranked_rank1.xlsx", index=False)

print("\n    Top 5 Long-Run Relationships:")
for idx, row in longrun_ranked.head(5).iterrows():
    print(f"      {row['From']:30s} -> {row['To']:30s}: {row['Magnitude']:6.2f} ({row['Direction']})")

# Create model summary
print("\n[9] CREATING MODEL SUMMARY...")

summary_text = f"""
VECM MODEL SUMMARY - RANK=1 (OPTIMAL)
{'='*80}

MODEL SPECIFICATION:
  Cointegration Rank: 1 (ONE equilibrium relationship)
  Lag Order (differences): 1 (k_ar_diff=1)
  Deterministic Terms: None (nc)
  Variables: {len(SELECTED_VARS)}
  Observations: {len(data)} (1987-2024)

WHY RANK=1 (vs previous rank=6):
  - Best out-of-sample prediction: 51.2% MAPE (vs 62.6% for rank=6)
  - Lowest BIC: 115 (vs 259 for rank=6) - penalizes overfitting
  - Won 5/8 criteria in robustness check
  - Simpler interpretation: ONE Iron Cage equilibrium (not six)
  - More parsimonious (Occam's Razor)

MODEL FIT:
  Log-Likelihood: {vecm_result.llf:.2f}
  Out-of-sample RMSE: {rmse:.4f}
  Out-of-sample MAE: {mae:.4f}
  Out-of-sample MAPE: {mape:.2f}%

RESIDUAL DIAGNOSTICS:
  Stationary residuals: {stationary_count}/{len(SELECTED_VARS)} variables ({100*stationary_count/len(SELECTED_VARS):.1f}%)

INTERPRETATION:
  With rank=1, there is ONE dominant cointegration vector binding all 8 variables
  in long-run equilibrium. This represents Weber's Iron Cage - one structural
  relationship that locks the bureaucratic system together.

  Short-run dynamics (Gamma): Year-to-year adjustments
  Long-run dynamics (Alpha x Beta): Error correction toward equilibrium

COEFFICIENT MATRICES:
  - Alpha (8x1): Error correction speeds for each variable
  - Beta (8x1): Weights defining the ONE equilibrium relationship
  - Gamma (8x8): Short-run VAR coefficients

TOP 5 LONG-RUN RELATIONSHIPS:
"""

for idx, row in longrun_ranked.head(5).iterrows():
    summary_text += f"  {idx+1}. {row['From']:30s} -> {row['To']:30s}: {row['Magnitude']:6.2f} ({row['Direction']})\n"

summary_text += f"""

FILES GENERATED:
  - alpha_matrix_rank1.xlsx: Error correction speeds (8x1)
  - beta_matrix_rank1.xlsx: Cointegration vector (8x1)
  - gamma_matrix_rank1.xlsx: Short-run VAR coefficients (8x8)
  - longrun_influence_rank1.xlsx: Long-run influence matrix (8x8)
  - longrun_relationships_ranked_rank1.xlsx: Ranked relationships
  - residual_stationarity_rank1.xlsx: Residual diagnostics
  - vecm_shortrun_influence_rank1.png: Short-run heatmap
  - vecm_longrun_influence_rank1.png: Long-run heatmap
  - vecm_influence_comparison_rank1.png: Side-by-side comparison

{'='*80}
Analysis complete!
"""

with open(OUTPUT_DIR / "vecm_model_summary_rank1.txt", 'w') as f:
    f.write(summary_text)

print(summary_text)

print("\n" + "=" * 80)
print("ALL OUTPUTS SAVED TO:")
print(f"  {OUTPUT_DIR}")
print("=" * 80)
