"""
VECM Rank=2 Analysis - Version 13.4
====================================
Complete VECM rank=2 estimation with new variables including:
- Policy_Count_LogZ (z-scored log transformation)
- Total_Civilians_Z (replaces GOFOs_Z)

Outputs:
- Short-run and long-run network diagrams
- Comparative influence graphic
- Alpha, beta, gamma matrices
- Beta importance chart
- Cointegration vectors visualization
- All diagnostics and summaries
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
OUTPUT_DIR = BASE_DIR / "VECM_Rank2_v13.4_Analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SELECTED_VARS = [
    'Warrant_Officers_Z',
    'Policy_Count_LogZ',
    'Company_Grade_Officers_Z',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z',
    'Junior_Enlisted_Z',
    'Field_Grade_Officers_Z',
    'GOFOs_Z'
]

DISPLAY_NAMES = {
    'Warrant_Officers_Z': 'Warrant\nOfficers',
    'Policy_Count_LogZ': 'Policy\nCount',
    'Company_Grade_Officers_Z': 'Company\nGrade',
    'Total_PAS_Z': 'Total\nPAS',
    'FOIA_Simple_Days_Z': 'FOIA\nDays',
    'Junior_Enlisted_Z': 'Junior\nEnlisted',
    'Field_Grade_Officers_Z': 'Field\nGrade',
    'GOFOs_Z': 'GOFOs'
}

DISPLAY_NAMES_LONG = {
    'Warrant_Officers_Z': 'Warrant Officers',
    'Policy_Count_LogZ': 'Policy Count (Log Z-Score)',
    'Company_Grade_Officers_Z': 'Company Grade (O-1 to O-3)',
    'Total_PAS_Z': 'Political Appointees (PAS)',
    'FOIA_Simple_Days_Z': 'FOIA Processing Delay',
    'Junior_Enlisted_Z': 'Junior Enlisted (E-1 to E-4)',
    'Field_Grade_Officers_Z': 'Field Grade (O-4 to O-5)',
    'GOFOs_Z': 'General/Flag Officers'
}

print("=" * 80)
print("VECM RANK=2 ANALYSIS - VERSION 13.4")
print("=" * 80)
print(f"\nVariables: {len(SELECTED_VARS)}")
for i, var in enumerate(SELECTED_VARS, 1):
    print(f"  {i}. {DISPLAY_NAMES_LONG[var]}")

# Load data
print("\n[1] Loading data...")
data_file = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\complete_normalized_dataset_v13.4.xlsx")
df = pd.read_excel(data_file)
df.columns = df.columns.str.strip()

# Verify all variables exist
missing_vars = [v for v in SELECTED_VARS if v not in df.columns]
if missing_vars:
    print(f"\nERROR: Missing variables in dataset: {missing_vars}")
    print(f"Available variables: {list(df.columns)}")
    exit(1)

data = df[SELECTED_VARS].dropna().copy()
print(f"    Data shape: {data.shape[0]} observations x {data.shape[1]} variables")

# Split train/test
train_data = data.iloc[:-5]
test_data = data.iloc[-5:]
print(f"    Training: {train_data.shape[0]} obs")
print(f"    Test: {test_data.shape[0]} obs")

# Estimate VECM
print("\n[2] Estimating VECM rank=2...")
vecm = VECM(data, k_ar_diff=1, coint_rank=2, deterministic='nc')
vecm_result = vecm.fit()

print(f"    Cointegration rank: 2")
print(f"    Lag order (differences): 1")

# Extract matrices
print("\n[3] Extracting matrices...")
alpha = vecm_result.alpha  # 8 x 2
beta = vecm_result.beta    # 8 x 2
gamma = vecm_result.gamma  # 8 x 8

alpha_df = pd.DataFrame(alpha, index=SELECTED_VARS, columns=['EC1', 'EC2'])
beta_df = pd.DataFrame(beta, index=SELECTED_VARS, columns=['EC1', 'EC2'])
gamma_df = pd.DataFrame(gamma, index=SELECTED_VARS, columns=SELECTED_VARS)

# Save matrices
alpha_df.to_excel(OUTPUT_DIR / "alpha_matrix_rank2.xlsx")
beta_df.to_excel(OUTPUT_DIR / "beta_matrix_rank2.xlsx")
gamma_df.to_excel(OUTPUT_DIR / "gamma_matrix_rank2.xlsx")

print(f"    Alpha (error correction): {alpha_df.shape}")
print(f"    Beta (cointegration vectors): {beta_df.shape}")
print(f"    Gamma (short-run VAR): {gamma_df.shape}")

# Calculate long-run influence
print("\n[4] Calculating long-run influence...")
longrun_influence = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))
signed_direction = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))

for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        signed_sum = 0
        unsigned_sum = 0
        for r in range(2):  # rank=2
            alpha_i = alpha_df.iloc[i, r]
            beta_j = beta_df.iloc[j, r]
            influence = alpha_i * beta_j
            signed_sum += influence
            unsigned_sum += abs(influence)

        longrun_influence[i, j] = unsigned_sum
        signed_direction[i, j] = np.sign(signed_sum)

longrun_df = pd.DataFrame(longrun_influence, index=SELECTED_VARS, columns=SELECTED_VARS)
longrun_df.to_excel(OUTPUT_DIR / "longrun_influence_rank2.xlsx")

print(f"    Long-run influence matrix: {longrun_df.shape}")

# Create heatmaps
print("\n[5] Creating influence heatmaps...")

display_names = [DISPLAY_NAMES[var] for var in SELECTED_VARS]
gamma_values = gamma_df.values
signed_magnitude = longrun_influence * signed_direction  # FIXED: Removed incorrect (-1) flip

# Short-run heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(gamma_values, annot=np.abs(gamma_values), fmt='.2f',
            cmap='RdBu_r', center=0, cbar_kws={'label': 'Coefficient Value'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=ax)
ax.set_title('SHORT-RUN DYNAMICS (Gamma) - Rank=2\nYear-to-year VAR effects',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('From Variable (t-1)', fontsize=12, fontweight='bold')
ax.set_ylabel('To Variable', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_shortrun_influence_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

# Long-run heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(signed_magnitude, annot=longrun_influence, fmt='.2f',
            cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Direction (RED=Amplifying, BLUE=Dampening)'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=ax)
ax.set_title('LONG-RUN INFLUENCE (Error Correction) - Rank=2\nMagnitude with directional coloring (sum across 2 vectors)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('From Variable (equilibrium deviation)', fontsize=12, fontweight='bold')
ax.set_ylabel('To Variable (adjustment)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_longrun_influence_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

# Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Short-run
sns.heatmap(gamma_values, annot=np.abs(gamma_values), fmt='.2f',
            cmap='RdBu_r', center=0, cbar_kws={'label': 'Coefficient'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=axes[0])
axes[0].set_title('SHORT-RUN (Gamma)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('From Variable (t-1)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('To Variable', fontsize=11, fontweight='bold')

# Long-run
sns.heatmap(signed_magnitude, annot=longrun_influence, fmt='.2f',
            cmap='RdBu_r', center=0, cbar_kws={'label': 'Direction'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=axes[1])
axes[1].set_title('LONG-RUN (Error Correction)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('From Variable (equilibrium deviation)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('To Variable (adjustment)', fontsize=11, fontweight='bold')

fig.suptitle('VECM INFLUENCE COMPARISON: Rank=2, k_ar_diff=1\n(v13.4 with Policy_Count_LogZ and GOFOs_Z)',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_influence_comparison_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Heatmaps saved")

# Beta importance
print("\n[6] Creating beta importance chart...")
fig, ax = plt.subplots(figsize=(14, 10))

beta_importance = np.abs(beta_df).sum(axis=1).sort_values(ascending=True)
vars_sorted = beta_importance.index
values = beta_importance.values
display_names_sorted = [DISPLAY_NAMES_LONG[v] for v in vars_sorted]

colors = ['red' if v == 'Total_PAS_Z' else
          'darkblue' if v == 'GOFOs_Z' else
          'navy' if v == 'Field_Grade_Officers_Z' else
          'orange' if v == 'FOIA_Simple_Days_Z' else
          'green' if v == 'Policy_Count_LogZ' else
          'gray' for v in vars_sorted]

ax.barh(range(len(vars_sorted)), values, color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(vars_sorted)))
ax.set_yticklabels(display_names_sorted, fontsize=11)
ax.set_xlabel('Total |Beta| Importance (sum across vectors)', fontsize=12, fontweight='bold')
ax.set_title('Beta Importance - Rank=2\n(Green=Policy, Blue=GOFOs, Navy=Field Grade, Red=PAS, Orange=FOIA)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

for i, val in enumerate(values):
    ax.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "beta_importance_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Beta importance chart saved")

# Cointegration vectors
print("\n[7] Creating cointegration vectors visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

for vec_idx in range(2):
    ax = axes[vec_idx]
    beta_values = beta_df.iloc[:, vec_idx].values

    vars_sorted_idx = np.argsort(np.abs(beta_values))[::-1]
    vars_sorted = [SELECTED_VARS[i] for i in vars_sorted_idx]
    values_sorted = beta_values[vars_sorted_idx]
    labels_sorted = [DISPLAY_NAMES_LONG[v] for v in vars_sorted]

    colors = ['red' if val > 0 else 'blue' for val in values_sorted]

    ax.barh(range(len(vars_sorted)), values_sorted, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(vars_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=10)
    ax.set_xlabel('Beta Coefficient', fontsize=11, fontweight='bold')
    ax.set_title(f'Cointegration Vector {vec_idx+1}\n(Red=Positive, Blue=Negative)',
                 fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1, linestyle='--')
    ax.grid(axis='x', alpha=0.3)

    for i, val in enumerate(values_sorted):
        ax.text(val + (0.02 if val > 0 else -0.02), i, f'{val:.2f}',
                va='center', ha='left' if val > 0 else 'right', fontsize=9)

fig.suptitle('COINTEGRATION VECTORS (Beta) - Rank=2\nTwo Equilibrium Relationships',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_cointegration_vectors_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Cointegration vectors saved")

# Network diagrams
print("\n[8] Creating network diagrams...")

from matplotlib.lines import Line2D

# Long-run network
G_longrun = nx.DiGraph()

beta_importance_vals = np.abs(beta_df).sum(axis=1).values
beta_importance_normalized = (beta_importance_vals / beta_importance_vals.max()) * 3000 + 500

for i, var in enumerate(SELECTED_VARS):
    G_longrun.add_node(var,
                       label=DISPLAY_NAMES_LONG[var],
                       importance=beta_importance_vals[i],
                       size=beta_importance_normalized[i])

threshold = 0.15
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            magnitude = longrun_df.iloc[i, j]
            if magnitude > threshold:
                direction = np.sign(signed_direction[i, j])
                G_longrun.add_edge(var_from, var_to, weight=magnitude, direction=direction)

fig, ax = plt.subplots(figsize=(16, 14))
pos = nx.circular_layout(G_longrun)

node_sizes = [G_longrun.nodes[node]['size'] for node in G_longrun.nodes()]
nx.draw_networkx_nodes(G_longrun, pos, node_size=node_sizes,
                       node_color='yellow', edgecolors='black',
                       linewidths=2.5, alpha=0.9, ax=ax)

edges_amplifying = [(u, v) for u, v, d in G_longrun.edges(data=True) if d['direction'] > 0]
edges_dampening = [(u, v) for u, v, d in G_longrun.edges(data=True) if d['direction'] < 0]

if edges_amplifying or edges_dampening:
    max_weight = max([d['weight'] for u, v, d in G_longrun.edges(data=True)])

    if edges_amplifying:
        weights_amp = [G_longrun[u][v]['weight'] for u, v in edges_amplifying]
        widths_amp = [3 + (w / max_weight) * 5 for w in weights_amp]
        nx.draw_networkx_edges(G_longrun, pos, edgelist=edges_amplifying,
                               width=widths_amp, edge_color='red', alpha=0.7,
                               arrows=True, arrowsize=25, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15',
                               min_source_margin=20, min_target_margin=20, ax=ax)

    if edges_dampening:
        weights_damp = [G_longrun[u][v]['weight'] for u, v in edges_dampening]
        widths_damp = [3 + (w / max_weight) * 5 for w in weights_damp]
        nx.draw_networkx_edges(G_longrun, pos, edgelist=edges_dampening,
                               width=widths_damp, edge_color='blue', alpha=0.7,
                               arrows=True, arrowsize=25, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15',
                               min_source_margin=20, min_target_margin=20, ax=ax)

labels = {node: G_longrun.nodes[node]['label'] for node in G_longrun.nodes()}
nx.draw_networkx_labels(G_longrun, pos, labels=labels,
                        font_size=9, font_weight='bold', ax=ax)

ax.set_title('LONG-RUN EQUILIBRIUM Network (Error Correction) - Rank=2\n(Red=Amplifying, Blue=Dampening; Width=Strength; Node size=Beta importance)',
             fontsize=14, fontweight='bold', pad=20)

legend_elements = [
    Line2D([0], [0], color='red', linewidth=3, label='Amplifying (+)'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dampening (-)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
           markersize=10, markeredgecolor='black', markeredgewidth=2,
           label='Node size = Beta importance', linestyle='None')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_longrun_network_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

# Short-run network
G_shortrun = nx.DiGraph()
for var in SELECTED_VARS:
    G_shortrun.add_node(var, label=DISPLAY_NAMES_LONG[var])

gamma_threshold = 0.15
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            coef = gamma_df.iloc[i, j]
            if abs(coef) > gamma_threshold:
                G_shortrun.add_edge(var_from, var_to, weight=abs(coef), direction=np.sign(coef))

fig, ax = plt.subplots(figsize=(16, 14))
pos_sr = nx.circular_layout(G_shortrun)

nx.draw_networkx_nodes(G_shortrun, pos_sr, node_size=2000,
                       node_color='lightblue', edgecolors='black',
                       linewidths=2.5, alpha=0.9, ax=ax)

edges_pos = [(u, v) for u, v, d in G_shortrun.edges(data=True) if d['direction'] > 0]
edges_neg = [(u, v) for u, v, d in G_shortrun.edges(data=True) if d['direction'] < 0]

if edges_pos or edges_neg:
    max_gamma = max([d['weight'] for u, v, d in G_shortrun.edges(data=True)])

    if edges_pos:
        weights_pos = [G_shortrun[u][v]['weight'] for u, v in edges_pos]
        widths_pos = [3 + (w / max_gamma) * 5 for w in weights_pos]
        nx.draw_networkx_edges(G_shortrun, pos_sr, edgelist=edges_pos,
                               width=widths_pos, edge_color='red', alpha=0.7,
                               arrows=True, arrowsize=25, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15',
                               min_source_margin=20, min_target_margin=20, ax=ax)

    if edges_neg:
        weights_neg = [G_shortrun[u][v]['weight'] for u, v in edges_neg]
        widths_neg = [3 + (w / max_gamma) * 5 for w in weights_neg]
        nx.draw_networkx_edges(G_shortrun, pos_sr, edgelist=edges_neg,
                               width=widths_neg, edge_color='blue', alpha=0.7,
                               arrows=True, arrowsize=25, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15',
                               min_source_margin=20, min_target_margin=20, ax=ax)

labels_sr = {node: G_shortrun.nodes[node]['label'] for node in G_shortrun.nodes()}
nx.draw_networkx_labels(G_shortrun, pos_sr, labels=labels_sr,
                        font_size=9, font_weight='bold', ax=ax)

ax.set_title('SHORT-RUN DYNAMICS Network (Year-to-Year VAR) - Rank=2\n(Red=Amplifying, Blue=Dampening; Width=Coefficient strength)',
             fontsize=14, fontweight='bold', pad=20)

legend_sr = [
    Line2D([0], [0], color='red', linewidth=3, label='Amplifying (+)'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dampening (-)')
]
ax.legend(handles=legend_sr, loc='upper left', fontsize=10)

ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_shortrun_network_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Network diagrams saved")

# Diagnostics
print("\n[9] Running diagnostics...")

# Out-of-sample
vecm_train = VECM(train_data, k_ar_diff=1, coint_rank=2, deterministic='nc')
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
        'Variable': DISPLAY_NAMES_LONG[var_name],
        'ADF_Statistic': adf_result[0],
        'p_value': adf_result[1],
        'Stationary': adf_result[1] < 0.05
    })

residual_df = pd.DataFrame(residual_tests)
residual_df.to_excel(OUTPUT_DIR / "residual_stationarity_rank2.xlsx", index=False)

stationary_count = sum(1 for r in residual_tests if r['Stationary'])

# Top relationships
longrun_flat = []
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            longrun_flat.append({
                'From': DISPLAY_NAMES_LONG[var_from],
                'To': DISPLAY_NAMES_LONG[var_to],
                'Magnitude': longrun_influence[i, j],
                'Direction': 'Amplifying' if signed_direction[i, j] > 0 else 'Dampening'
            })

longrun_ranked = pd.DataFrame(longrun_flat).sort_values('Magnitude', ascending=False)
longrun_ranked.to_excel(OUTPUT_DIR / "longrun_relationships_ranked_rank2.xlsx", index=False)

# Model summary
print("\n[10] Creating model summary...")

llf = vecm_result.llf
nobs = len(data) - 1
neqs = len(SELECTED_VARS)
k_params = neqs * 2 + neqs * neqs + 2 * (neqs - 1)
aic = -2 * llf + 2 * k_params
bic = -2 * llf + k_params * np.log(nobs)

summary_text = f"""
VECM MODEL SUMMARY - RANK=2 - VERSION 13.4
{'='*80}

MODEL SPECIFICATION:
  Cointegration Rank: 2 (2 equilibrium relationships)
  Lag Order (differences): 1 (k_ar_diff=1)
  Deterministic Terms: None (nc)
  Variables: {len(SELECTED_VARS)}
  Observations: {len(data)} (1987-2024)

VARIABLES:
"""

for i, var in enumerate(SELECTED_VARS, 1):
    summary_text += f"  {i}. {DISPLAY_NAMES_LONG[var]}\n"

summary_text += f"""
MODEL FIT:
  Log-Likelihood: {llf:.2f}
  AIC: {aic:.2f}
  BIC: {bic:.2f}
  Out-of-sample RMSE: {rmse:.4f}
  Out-of-sample MAE: {mae:.4f}
  Out-of-sample MAPE: {mape:.2f}%

RESIDUAL DIAGNOSTICS:
  Stationary residuals: {stationary_count}/{len(SELECTED_VARS)} variables ({100*stationary_count/len(SELECTED_VARS):.1f}%)

BETA IMPORTANCE (sum of |beta| across 2 vectors):
"""

for var, importance in beta_importance.items():
    summary_text += f"  {DISPLAY_NAMES_LONG[var]:45s}: {importance:8.2f}\n"

summary_text += f"""
TOP 10 LONG-RUN RELATIONSHIPS:
"""

for idx, row in longrun_ranked.head(10).iterrows():
    summary_text += f"  {idx+1}. {row['From']:40s} -> {row['To']:40s}: {row['Magnitude']:6.2f} ({row['Direction']})\n"

summary_text += f"""
FILES GENERATED:
  - alpha_matrix_rank2.xlsx
  - beta_matrix_rank2.xlsx
  - gamma_matrix_rank2.xlsx
  - longrun_influence_rank2.xlsx
  - longrun_relationships_ranked_rank2.xlsx
  - residual_stationarity_rank2.xlsx
  - vecm_shortrun_influence_rank2.png
  - vecm_longrun_influence_rank2.png
  - vecm_influence_comparison_rank2.png
  - beta_importance_rank2.png
  - vecm_cointegration_vectors_rank2.png
  - vecm_longrun_network_rank2.png
  - vecm_shortrun_network_rank2.png

{'='*80}
Analysis complete for rank=2 (v13.4)!
"""

with open(OUTPUT_DIR / "vecm_model_summary_rank2.txt", 'w') as f:
    f.write(summary_text)

print(summary_text)

print("\n" + "=" * 80)
print("ALL OUTPUTS SAVED TO:")
print(f"  {OUTPUT_DIR}")
print("=" * 80)
