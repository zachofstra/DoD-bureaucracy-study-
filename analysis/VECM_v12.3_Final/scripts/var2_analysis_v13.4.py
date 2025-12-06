"""
VAR(2) Analysis - Version 13.4
===============================
Complete VAR(2) estimation with new variables including:
- Policy_Count_LogZ (z-scored log transformation)
- Total_Civilians_Z (replaces GOFOs_Z)

Outputs:
- Network diagrams for lag 1 and lag 2
- Coefficient heatmaps for lag 1 and lag 2
- Key relationships graphic
- Model diagnostics and summaries
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
OUTPUT_DIR = BASE_DIR / "VAR2_v13.4_Analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SELECTED_VARS = [
    'Warrant_Officers_Z',
    'Policy_Count_LogZ',
    'Company_Grade_Officers_Z',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z',
    'Junior_Enlisted_Z',
    'Field_Grade_Officers_Z',
    'Total_Civilians_Z'
]

DISPLAY_NAMES = {
    'Warrant_Officers_Z': 'Warrant\nOfficers',
    'Policy_Count_LogZ': 'Policy\nCount',
    'Company_Grade_Officers_Z': 'Company\nGrade',
    'Total_PAS_Z': 'Total\nPAS',
    'FOIA_Simple_Days_Z': 'FOIA\nDays',
    'Junior_Enlisted_Z': 'Junior\nEnlisted',
    'Field_Grade_Officers_Z': 'Field\nGrade',
    'Total_Civilians_Z': 'Total\nCivilians'
}

DISPLAY_NAMES_LONG = {
    'Warrant_Officers_Z': 'Warrant Officers',
    'Policy_Count_LogZ': 'Policy Count (Log Z-Score)',
    'Company_Grade_Officers_Z': 'Company Grade (O-1 to O-3)',
    'Total_PAS_Z': 'Political Appointees (PAS)',
    'FOIA_Simple_Days_Z': 'FOIA Processing Delay',
    'Junior_Enlisted_Z': 'Junior Enlisted (E-1 to E-4)',
    'Field_Grade_Officers_Z': 'Field Grade (O-4 to O-5)',
    'Total_Civilians_Z': 'Total Civilian Personnel'
}

print("=" * 80)
print("VAR(2) ANALYSIS - VERSION 13.4")
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

# Estimate VAR(2)
print("\n[2] Estimating VAR(2) model...")
var_model = VAR(data)
var_result = var_model.fit(maxlags=2)

print(f"    AIC: {var_result.aic:.2f}")
print(f"    BIC: {var_result.bic:.2f}")
print(f"    Log-Likelihood: {var_result.llf:.2f}")

# Extract coefficients
print("\n[3] Extracting coefficient matrices...")
lag1_coef = var_result.params.iloc[:len(SELECTED_VARS), :].values.T  # 8x8
lag2_coef = var_result.params.iloc[len(SELECTED_VARS):2*len(SELECTED_VARS), :].values.T  # 8x8

lag1_df = pd.DataFrame(lag1_coef, index=SELECTED_VARS, columns=SELECTED_VARS)
lag2_df = pd.DataFrame(lag2_coef, index=SELECTED_VARS, columns=SELECTED_VARS)

# Save matrices
lag1_df.to_excel(OUTPUT_DIR / "var2_lag1_coefficients.xlsx")
lag2_df.to_excel(OUTPUT_DIR / "var2_lag2_coefficients.xlsx")

print(f"    Lag 1 coefficients: {lag1_df.shape}")
print(f"    Lag 2 coefficients: {lag2_df.shape}")

# Create coefficient heatmaps
print("\n[4] Creating coefficient heatmaps...")

display_names = [DISPLAY_NAMES[var] for var in SELECTED_VARS]

# Lag 1 heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(lag1_coef, annot=True, fmt='.3f',
            cmap='RdBu_r', center=0, cbar_kws={'label': 'Coefficient Value'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=ax)
ax.set_title('VAR(2) LAG 1 COEFFICIENTS\n(From Variable at t-1 → To Variable)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('From Variable (t-1)', fontsize=12, fontweight='bold')
ax.set_ylabel('To Variable (t)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "var2_lag1_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# Lag 2 heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(lag2_coef, annot=True, fmt='.3f',
            cmap='RdBu_r', center=0, cbar_kws={'label': 'Coefficient Value'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=ax)
ax.set_title('VAR(2) LAG 2 COEFFICIENTS\n(From Variable at t-2 → To Variable)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('From Variable (t-2)', fontsize=12, fontweight='bold')
ax.set_ylabel('To Variable (t)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "var2_lag2_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Heatmaps saved")

# Network diagrams
print("\n[5] Creating network diagrams...")

from matplotlib.lines import Line2D

# Lag 1 network
G_lag1 = nx.DiGraph()
for var in SELECTED_VARS:
    G_lag1.add_node(var, label=DISPLAY_NAMES_LONG[var])

threshold = 0.15
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            coef = lag1_coef[i, j]
            if abs(coef) > threshold:
                G_lag1.add_edge(var_from, var_to, weight=abs(coef), direction=np.sign(coef))

fig, ax = plt.subplots(figsize=(16, 12))
pos_lag1 = nx.spring_layout(G_lag1, k=2, iterations=50, seed=42)

nx.draw_networkx_nodes(G_lag1, pos_lag1, node_size=2500,
                       node_color='lightblue', edgecolors='black',
                       linewidths=2, ax=ax)

edges_pos = [(u, v) for u, v, d in G_lag1.edges(data=True) if d['direction'] > 0]
edges_neg = [(u, v) for u, v, d in G_lag1.edges(data=True) if d['direction'] < 0]

if edges_pos or edges_neg:
    max_weight = max([d['weight'] for u, v, d in G_lag1.edges(data=True)])

    if edges_pos:
        weights_pos = [G_lag1[u][v]['weight'] for u, v in edges_pos]
        widths_pos = [w / max_weight * 5 for w in weights_pos]
        nx.draw_networkx_edges(G_lag1, pos_lag1, edgelist=edges_pos,
                               width=widths_pos, edge_color='red', alpha=0.7,
                               arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)

    if edges_neg:
        weights_neg = [G_lag1[u][v]['weight'] for u, v in edges_neg]
        widths_neg = [w / max_weight * 5 for w in weights_neg]
        nx.draw_networkx_edges(G_lag1, pos_lag1, edgelist=edges_neg,
                               width=widths_neg, edge_color='blue', alpha=0.7,
                               arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)

labels_lag1 = {node: G_lag1.nodes[node]['label'] for node in G_lag1.nodes()}
nx.draw_networkx_labels(G_lag1, pos_lag1, labels=labels_lag1,
                        font_size=9, font_weight='bold', ax=ax)

ax.set_title('VAR(2) LAG 1 NETWORK\n(Red=Amplifying, Blue=Dampening; Width=Coefficient strength)',
             fontsize=14, fontweight='bold', pad=20)

legend_elements = [
    Line2D([0], [0], color='red', linewidth=3, label='Amplifying (+)'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dampening (-)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "var2_lag1_network.png", dpi=300, bbox_inches='tight')
plt.close()

# Lag 2 network
G_lag2 = nx.DiGraph()
for var in SELECTED_VARS:
    G_lag2.add_node(var, label=DISPLAY_NAMES_LONG[var])

for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            coef = lag2_coef[i, j]
            if abs(coef) > threshold:
                G_lag2.add_edge(var_from, var_to, weight=abs(coef), direction=np.sign(coef))

fig, ax = plt.subplots(figsize=(16, 12))
pos_lag2 = nx.spring_layout(G_lag2, k=2, iterations=50, seed=42)

nx.draw_networkx_nodes(G_lag2, pos_lag2, node_size=2500,
                       node_color='lightgreen', edgecolors='black',
                       linewidths=2, ax=ax)

edges_pos_2 = [(u, v) for u, v, d in G_lag2.edges(data=True) if d['direction'] > 0]
edges_neg_2 = [(u, v) for u, v, d in G_lag2.edges(data=True) if d['direction'] < 0]

if edges_pos_2 or edges_neg_2:
    max_weight_2 = max([d['weight'] for u, v, d in G_lag2.edges(data=True)])

    if edges_pos_2:
        weights_pos_2 = [G_lag2[u][v]['weight'] for u, v in edges_pos_2]
        widths_pos_2 = [w / max_weight_2 * 5 for w in weights_pos_2]
        nx.draw_networkx_edges(G_lag2, pos_lag2, edgelist=edges_pos_2,
                               width=widths_pos_2, edge_color='red', alpha=0.7,
                               arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)

    if edges_neg_2:
        weights_neg_2 = [G_lag2[u][v]['weight'] for u, v in edges_neg_2]
        widths_neg_2 = [w / max_weight_2 * 5 for w in weights_neg_2]
        nx.draw_networkx_edges(G_lag2, pos_lag2, edgelist=edges_neg_2,
                               width=widths_neg_2, edge_color='blue', alpha=0.7,
                               arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)

labels_lag2 = {node: G_lag2.nodes[node]['label'] for node in G_lag2.nodes()}
nx.draw_networkx_labels(G_lag2, pos_lag2, labels=labels_lag2,
                        font_size=9, font_weight='bold', ax=ax)

ax.set_title('VAR(2) LAG 2 NETWORK\n(Red=Amplifying, Blue=Dampening; Width=Coefficient strength)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "var2_lag2_network.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Network diagrams saved")

# Key relationships
print("\n[6] Identifying key relationships...")

all_relationships = []
for lag, coef_matrix, lag_name in [(1, lag1_coef, 'Lag 1'), (2, lag2_coef, 'Lag 2')]:
    for i, var_to in enumerate(SELECTED_VARS):
        for j, var_from in enumerate(SELECTED_VARS):
            if i != j:
                coef = coef_matrix[i, j]
                if abs(coef) > 0.1:
                    all_relationships.append({
                        'Lag': lag_name,
                        'From': DISPLAY_NAMES_LONG[var_from],
                        'To': DISPLAY_NAMES_LONG[var_to],
                        'Coefficient': coef,
                        'Magnitude': abs(coef),
                        'Direction': 'Amplifying' if coef > 0 else 'Dampening'
                    })

relationships_df = pd.DataFrame(all_relationships).sort_values('Magnitude', ascending=False)
relationships_df.to_excel(OUTPUT_DIR / "var2_key_relationships.xlsx", index=False)

# Key relationships graphic
fig, ax = plt.subplots(figsize=(14, 10))

top_20 = relationships_df.head(20)
y_pos = range(len(top_20))
colors = ['red' if d == 'Amplifying' else 'blue' for d in top_20['Direction']]

bars = ax.barh(y_pos, top_20['Magnitude'], color=colors, alpha=0.7, edgecolor='black')

labels = [f"{row['From'][:20]} → {row['To'][:20]} ({row['Lag']})"
          for _, row in top_20.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Coefficient Magnitude', fontsize=12, fontweight='bold')
ax.set_title('VAR(2) TOP 20 RELATIONSHIPS\n(Red=Amplifying, Blue=Dampening)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

for i, (_, row) in enumerate(top_20.iterrows()):
    ax.text(row['Magnitude'] + 0.02, i, f"{row['Coefficient']:.3f}",
            va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "var2_key_relationships.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"    Top 20 relationships saved")

# Diagnostics
print("\n[7] Running diagnostics...")

# Out-of-sample forecast
var_train = VAR(train_data)
var_train_result = var_train.fit(maxlags=2)
forecast = var_train_result.forecast(train_data.values[-2:], steps=5)

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
residuals = var_result.resid
residual_tests = []
for var_idx, var_name in enumerate(SELECTED_VARS):
    adf_result = adfuller(residuals.iloc[:, var_idx], autolag='AIC')
    residual_tests.append({
        'Variable': DISPLAY_NAMES_LONG[var_name],
        'ADF_Statistic': adf_result[0],
        'p_value': adf_result[1],
        'Stationary': adf_result[1] < 0.05
    })

residual_df = pd.DataFrame(residual_tests)
residual_df.to_excel(OUTPUT_DIR / "var2_residual_stationarity.xlsx", index=False)

stationary_count = sum(1 for r in residual_tests if r['Stationary'])

# Model summary
print("\n[8] Creating model summary...")

summary_text = f"""
VAR(2) MODEL SUMMARY - VERSION 13.4
{'='*80}

MODEL SPECIFICATION:
  Model: Vector Autoregression
  Lag Order: 2
  Variables: {len(SELECTED_VARS)}
  Observations: {len(data)}

VARIABLES:
"""

for i, var in enumerate(SELECTED_VARS, 1):
    summary_text += f"  {i}. {DISPLAY_NAMES_LONG[var]}\n"

summary_text += f"""
MODEL FIT:
  AIC: {var_result.aic:.2f}
  BIC: {var_result.bic:.2f}
  Log-Likelihood: {var_result.llf:.2f}
  Out-of-sample RMSE: {rmse:.4f}
  Out-of-sample MAE: {mae:.4f}
  Out-of-sample MAPE: {mape:.2f}%

RESIDUAL DIAGNOSTICS:
  Stationary residuals: {stationary_count}/{len(SELECTED_VARS)} variables ({100*stationary_count/len(SELECTED_VARS):.1f}%)

TOP 10 RELATIONSHIPS:
"""

for idx, row in relationships_df.head(10).iterrows():
    summary_text += f"  {idx+1}. {row['From'][:30]:30s} -> {row['To'][:30]:30s}: {row['Coefficient']:7.3f} ({row['Direction']}, {row['Lag']})\n"

summary_text += f"""
FILES GENERATED:
  - var2_lag1_coefficients.xlsx
  - var2_lag2_coefficients.xlsx
  - var2_lag1_heatmap.png
  - var2_lag2_heatmap.png
  - var2_lag1_network.png
  - var2_lag2_network.png
  - var2_key_relationships.xlsx
  - var2_key_relationships.png
  - var2_residual_stationarity.xlsx
  - var2_model_summary.txt

{'='*80}
VAR(2) Analysis Complete!
"""

with open(OUTPUT_DIR / "var2_model_summary.txt", 'w') as f:
    f.write(summary_text)

print(summary_text)

print("\n" + "=" * 80)
print("ALL OUTPUTS SAVED TO:")
print(f"  {OUTPUT_DIR}")
print("=" * 80)
