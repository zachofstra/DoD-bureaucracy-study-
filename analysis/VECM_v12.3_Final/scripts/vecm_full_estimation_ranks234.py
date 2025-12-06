"""
Full VECM Estimation for Ranks 2-4
====================================
Generate complete outputs for ranks 2, 3, and 4 (all with k_ar_diff=1):
- Alpha/Beta/Gamma matrices
- Influence heatmaps
- Network diagrams
- Model summaries
- Comparison tables
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")

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

DISPLAY_NAMES_LONG = {
    'Junior_Enlisted_Z': 'Junior Enlisted (E-1 to E-4)',
    'Company_Grade_Officers_Z': 'Company Grade (O-1 to O-3)',
    'Field_Grade_Officers_Z': 'Field Grade (O-4 to O-5)',
    'GOFOs_Z': 'General/Flag Officers',
    'Warrant_Officers_Z': 'Warrant Officers',
    'Policy_Count_Log': 'Policy Volume (Log)',
    'Total_PAS_Z': 'Political Appointees (PAS)',
    'FOIA_Simple_Days_Z': 'FOIA Processing Delay'
}

print("=" * 80)
print("FULL VECM ESTIMATION: RANKS 2, 3, 4")
print("=" * 80)
print("\nAll use k_ar_diff=1 (validated as optimal)")
print("\n" + "=" * 80)

# Load data
data_file = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\complete_normalized_dataset_v12.3.xlsx")
df = pd.read_excel(data_file)
df.columns = df.columns.str.strip()
data = df[SELECTED_VARS].dropna().copy()

train_data = data.iloc[:-5]
test_data = data.iloc[-5:]

print(f"\nData: {data.shape[0]} observations x {data.shape[1]} variables")
print(f"Training: {train_data.shape[0]} observations")
print(f"Test: {test_data.shape[0]} observations")

# Process each rank
for rank in [2, 3, 4]:
    print(f"\n{'='*80}")
    print(f"PROCESSING RANK={rank}")
    print(f"{'='*80}")

    # Create output directory
    OUTPUT_DIR = BASE_DIR / f"VECM_Rank{rank}_Final_Executive_Summary"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Estimate VECM
    print(f"\n[1] Estimating VECM...")
    vecm = VECM(data, k_ar_diff=1, coint_rank=rank, deterministic='nc')
    vecm_result = vecm.fit()

    # Extract matrices
    print(f"[2] Extracting matrices...")
    alpha = vecm_result.alpha  # 8 x rank
    beta = vecm_result.beta    # 8 x rank
    gamma = vecm_result.gamma  # 8 x 8

    alpha_df = pd.DataFrame(alpha, index=SELECTED_VARS,
                           columns=[f'EC{i+1}' for i in range(rank)])
    beta_df = pd.DataFrame(beta, index=SELECTED_VARS,
                          columns=[f'EC{i+1}' for i in range(rank)])
    gamma_df = pd.DataFrame(gamma, index=SELECTED_VARS, columns=SELECTED_VARS)

    # Save matrices
    alpha_df.to_excel(OUTPUT_DIR / f"alpha_matrix_rank{rank}.xlsx")
    beta_df.to_excel(OUTPUT_DIR / f"beta_matrix_rank{rank}.xlsx")
    gamma_df.to_excel(OUTPUT_DIR / f"gamma_matrix_rank{rank}.xlsx")

    # Calculate long-run influence (sum of |α × β| across all rank vectors)
    print(f"[3] Calculating long-run influence...")
    longrun_influence = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))
    signed_direction = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))

    for i in range(len(SELECTED_VARS)):
        for j in range(len(SELECTED_VARS)):
            signed_sum = 0
            unsigned_sum = 0
            for r in range(rank):
                alpha_i = alpha_df.iloc[i, r]
                beta_j = beta_df.iloc[j, r]
                influence = alpha_i * beta_j
                signed_sum += influence
                unsigned_sum += abs(influence)

            longrun_influence[i, j] = unsigned_sum
            signed_direction[i, j] = np.sign(signed_sum)

    longrun_df = pd.DataFrame(longrun_influence, index=SELECTED_VARS, columns=SELECTED_VARS)
    longrun_df.to_excel(OUTPUT_DIR / f"longrun_influence_rank{rank}.xlsx")

    # Create heatmaps
    print(f"[4] Creating heatmaps...")

    # Short-run heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    gamma_values = gamma_df.values
    display_names = [DISPLAY_NAMES[var] for var in SELECTED_VARS]

    sns.heatmap(gamma_values, annot=np.abs(gamma_values), fmt='.2f',
                cmap='RdBu_r', center=0, cbar_kws={'label': 'Coefficient Value'},
                xticklabels=display_names, yticklabels=display_names,
                linewidths=0.5, linecolor='gray', ax=ax)

    ax.set_title(f'SHORT-RUN DYNAMICS (Gamma) - Rank={rank}\\nYear-to-year VAR effects',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('From Variable (t-1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('To Variable', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"vecm_shortrun_influence_rank{rank}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Long-run heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    signed_magnitude_flipped = longrun_influence * signed_direction * (-1)

    sns.heatmap(signed_magnitude_flipped, annot=longrun_influence, fmt='.2f',
                cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Direction (RED=Amplifying, BLUE=Dampening)'},
                xticklabels=display_names, yticklabels=display_names,
                linewidths=0.5, linecolor='gray', ax=ax)

    ax.set_title(f'LONG-RUN INFLUENCE (Error Correction) - Rank={rank}\\nMagnitude with directional coloring (sum across {rank} vectors)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('From Variable (equilibrium deviation)', fontsize=12, fontweight='bold')
    ax.set_ylabel('To Variable (adjustment)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"vecm_longrun_influence_rank{rank}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # Short-run
    sns.heatmap(gamma_values, annot=np.abs(gamma_values), fmt='.2f',
                cmap='RdBu_r', center=0, cbar_kws={'label': 'Coefficient'},
                xticklabels=display_names, yticklabels=display_names,
                linewidths=0.5, linecolor='gray', ax=axes[0])
    axes[0].set_title(f'SHORT-RUN (Gamma)', fontsize=13, fontweight='bold')

    # Long-run
    sns.heatmap(signed_magnitude_flipped, annot=longrun_influence, fmt='.2f',
                cmap='RdBu_r', center=0, cbar_kws={'label': 'Direction'},
                xticklabels=display_names, yticklabels=display_names,
                linewidths=0.5, linecolor='gray', ax=axes[1])
    axes[1].set_title(f'LONG-RUN (Error Correction)', fontsize=13, fontweight='bold')

    fig.suptitle(f'VECM INFLUENCE COMPARISON: Rank={rank}, k_ar_diff=1',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"vecm_influence_comparison_rank{rank}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Beta importance chart
    print(f"[5] Creating beta importance chart...")
    fig, ax = plt.subplots(figsize=(14, 10))

    beta_importance = np.abs(beta_df).sum(axis=1).sort_values(ascending=True)
    vars_sorted = beta_importance.index
    values = beta_importance.values
    display_names_sorted = [DISPLAY_NAMES_LONG[v] for v in vars_sorted]

    colors = ['red' if v == 'Total_PAS_Z' else
              'darkblue' if v == 'GOFOs_Z' else
              'navy' if v == 'Field_Grade_Officers_Z' else
              'orange' if v == 'FOIA_Simple_Days_Z' else
              'gray' for v in vars_sorted]

    ax.barh(range(len(vars_sorted)), values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(vars_sorted)))
    ax.set_yticklabels(display_names_sorted, fontsize=11)
    ax.set_xlabel('Total |Beta| Importance (sum across vectors)', fontsize=12, fontweight='bold')
    ax.set_title(f'Beta Importance - Rank={rank}\\n(Red=PAS, Blue=GOFOs, Navy=Field Grade, Orange=FOIA)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    for i, val in enumerate(values):
        ax.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"beta_importance_rank{rank}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Diagnostics
    print(f"[6] Running diagnostics...")

    # Out-of-sample
    vecm_train = VECM(train_data, k_ar_diff=1, coint_rank=rank, deterministic='nc')
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

    # Residual stationarity
    residuals = vecm_result.resid
    residual_tests = []
    for var_idx, var_name in enumerate(SELECTED_VARS):
        adf_result = adfuller(residuals[:, var_idx], autolag='AIC')
        residual_tests.append({
            'Variable': var_name,
            'ADF_Statistic': adf_result[0],
            'p_value': adf_result[1],
            'Stationary': adf_result[1] < 0.05
        })

    residual_df = pd.DataFrame(residual_tests)
    residual_df.to_excel(OUTPUT_DIR / f"residual_stationarity_rank{rank}.xlsx", index=False)

    stationary_count = sum(1 for r in residual_tests if r['Stationary'])

    # Top relationships
    longrun_flat = []
    for i, var_to in enumerate(SELECTED_VARS):
        for j, var_from in enumerate(SELECTED_VARS):
            if i != j:
                longrun_flat.append({
                    'From': var_from,
                    'To': var_to,
                    'Magnitude': longrun_influence[i, j],
                    'Direction': 'Amplifying' if signed_direction[i, j] > 0 else 'Dampening'
                })

    longrun_ranked = pd.DataFrame(longrun_flat).sort_values('Magnitude', ascending=False)
    longrun_ranked.to_excel(OUTPUT_DIR / f"longrun_relationships_ranked_rank{rank}.xlsx", index=False)

    # Model summary
    print(f"[7] Creating model summary...")

    llf = vecm_result.llf
    nobs = len(data) - 1
    neqs = len(SELECTED_VARS)
    k_params = neqs * rank + neqs * neqs + rank * (neqs - 1)
    aic = -2 * llf + 2 * k_params
    bic = -2 * llf + k_params * np.log(nobs)

    summary_text = f"""
VECM MODEL SUMMARY - RANK={rank}
{'='*80}

MODEL SPECIFICATION:
  Cointegration Rank: {rank} ({rank} equilibrium relationship{'s' if rank > 1 else ''})
  Lag Order (differences): 1 (k_ar_diff=1)
  Deterministic Terms: None (nc)
  Variables: {len(SELECTED_VARS)}
  Observations: {len(data)} (1987-2024)

MODEL FIT:
  Log-Likelihood: {llf:.2f}
  AIC: {aic:.2f}
  BIC: {bic:.2f}
  Out-of-sample RMSE: {rmse:.4f}
  Out-of-sample MAE: {mae:.4f}
  Out-of-sample MAPE: {mape:.2f}%

RESIDUAL DIAGNOSTICS:
  Stationary residuals: {stationary_count}/{len(SELECTED_VARS)} variables ({100*stationary_count/len(SELECTED_VARS):.1f}%)

BETA IMPORTANCE (sum of |beta| across {rank} vector{'s' if rank > 1 else ''}):
"""

    for var, importance in beta_importance.items():
        summary_text += f"  {DISPLAY_NAMES_LONG[var]:45s}: {importance:8.2f}\n"

    summary_text += f"""
TOP 5 LONG-RUN RELATIONSHIPS:
"""

    for idx, row in longrun_ranked.head(5).iterrows():
        summary_text += f"  {idx+1}. {row['From']:35s} -> {row['To']:35s}: {row['Magnitude']:6.2f} ({row['Direction']})\n"

    summary_text += f"""
FILES GENERATED:
  - alpha_matrix_rank{rank}.xlsx
  - beta_matrix_rank{rank}.xlsx
  - gamma_matrix_rank{rank}.xlsx
  - longrun_influence_rank{rank}.xlsx
  - longrun_relationships_ranked_rank{rank}.xlsx
  - residual_stationarity_rank{rank}.xlsx
  - vecm_shortrun_influence_rank{rank}.png
  - vecm_longrun_influence_rank{rank}.png
  - vecm_influence_comparison_rank{rank}.png
  - beta_importance_rank{rank}.png

{'='*80}
Analysis complete for rank={rank}!
"""

    with open(OUTPUT_DIR / f"vecm_model_summary_rank{rank}.txt", 'w') as f:
        f.write(summary_text)

    print(summary_text)

print("\n" + "=" * 80)
print("ALL RANKS COMPLETE!")
print("=" * 80)
print("\nOutput directories:")
for rank in [2, 3, 4]:
    print(f"  - VECM_Rank{rank}_Final_Executive_Summary/")
