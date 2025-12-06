"""
Add Network Diagrams and Fix Labels for Rank=2
===============================================
1. Generate network diagrams (long-run, short-run, beta vectors)
2. Regenerate influence comparison with proper axis labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
OUTPUT_DIR = BASE_DIR / "VECM_Rank2_Final_Executive_Summary"

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
print("ADDING NETWORK DIAGRAMS AND FIXING LABELS FOR RANK=2")
print("=" * 80)

# Load matrices
print("\n[1] Loading matrices...")
alpha_df = pd.read_excel(OUTPUT_DIR / "alpha_matrix_rank2.xlsx", index_col=0)
beta_df = pd.read_excel(OUTPUT_DIR / "beta_matrix_rank2.xlsx", index_col=0)
gamma_df = pd.read_excel(OUTPUT_DIR / "gamma_matrix_rank2.xlsx", index_col=0)
longrun_df = pd.read_excel(OUTPUT_DIR / "longrun_influence_rank2.xlsx", index_col=0)

print(f"    Alpha: {alpha_df.shape}")
print(f"    Beta: {beta_df.shape}")
print(f"    Gamma: {gamma_df.shape}")

# Recalculate signed direction for coloring
signed_direction = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))
for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        signed_sum = 0
        for r in range(2):  # rank=2
            alpha_i = alpha_df.iloc[i, r]
            beta_j = beta_df.iloc[j, r]
            signed_sum += alpha_i * beta_j
        signed_direction[i, j] = np.sign(signed_sum)

# Fix influence comparison with better labels
print("\n[2] Regenerating influence comparison with axis labels...")

fig, axes = plt.subplots(1, 2, figsize=(24, 10))

display_names = [DISPLAY_NAMES[var] for var in SELECTED_VARS]
gamma_values = gamma_df.values
longrun_values = longrun_df.values
signed_magnitude_flipped = longrun_values * signed_direction * (-1)

# Short-run
sns.heatmap(gamma_values, annot=np.abs(gamma_values), fmt='.2f',
            cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Coefficient Value'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=axes[0])
axes[0].set_title('SHORT-RUN DYNAMICS (Gamma)\nYear-to-year VAR effects',
                  fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('From Variable (t-1)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('To Variable (equilibrium deviation)', fontsize=12, fontweight='bold')

# Long-run
sns.heatmap(signed_magnitude_flipped, annot=longrun_values, fmt='.2f',
            cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Direction (RED=Amplifying, BLUE=Dampening)'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=axes[1])
axes[1].set_title('LONG-RUN INFLUENCE (Error Correction)\nMagnitude with directional coloring (sum across 2 vectors)',
                  fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('From Variable (equilibrium deviation)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('To Variable (adjustment)', fontsize=12, fontweight='bold')

fig.suptitle('VECM INFLUENCE COMPARISON: Short-Run vs Long-Run (RANK=2)\n(Magnitude values with directional coloring)',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_influence_comparison_rank2.png", dpi=300, bbox_inches='tight')
plt.close()
print("    Updated influence comparison saved")

# Network Diagram 1: Long-run equilibrium
print("\n[3] Creating long-run network...")

G_longrun = nx.DiGraph()

# Node sizes based on beta importance
beta_importance = np.abs(beta_df).sum(axis=1).values
beta_importance_normalized = (beta_importance / beta_importance.max()) * 3000 + 500

for i, var in enumerate(SELECTED_VARS):
    G_longrun.add_node(var,
                       label=DISPLAY_NAMES_LONG[var],
                       importance=beta_importance[i],
                       size=beta_importance_normalized[i])

# Add edges
threshold = 0.15
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            magnitude = longrun_df.iloc[i, j]
            if magnitude > threshold:
                direction = np.sign(signed_direction[i, j])
                G_longrun.add_edge(var_from, var_to, weight=magnitude, direction=direction)

# Draw
fig, ax = plt.subplots(figsize=(16, 12))
pos = nx.spring_layout(G_longrun, k=2, iterations=50, seed=42)

node_sizes = [G_longrun.nodes[node]['size'] for node in G_longrun.nodes()]
nx.draw_networkx_nodes(G_longrun, pos, node_size=node_sizes,
                       node_color='yellow', edgecolors='black',
                       linewidths=2, ax=ax)

edges_amplifying = [(u, v) for u, v, d in G_longrun.edges(data=True) if d['direction'] > 0]
edges_dampening = [(u, v) for u, v, d in G_longrun.edges(data=True) if d['direction'] < 0]

weights_amp = [G_longrun[u][v]['weight'] for u, v in edges_amplifying]
weights_damp = [G_longrun[u][v]['weight'] for u, v in edges_dampening]

if weights_amp or weights_damp:
    max_weight = max([d['weight'] for u, v, d in G_longrun.edges(data=True)])
    widths_amp = [w / max_weight * 5 for w in weights_amp]
    widths_damp = [w / max_weight * 5 for w in weights_damp]
else:
    widths_amp, widths_damp = [], []

if edges_amplifying:
    nx.draw_networkx_edges(G_longrun, pos, edgelist=edges_amplifying,
                           width=widths_amp, edge_color='red', alpha=0.7,
                           arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)

if edges_dampening:
    nx.draw_networkx_edges(G_longrun, pos, edgelist=edges_dampening,
                           width=widths_damp, edge_color='blue', alpha=0.7,
                           arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)

labels = {node: G_longrun.nodes[node]['label'] for node in G_longrun.nodes()}
nx.draw_networkx_labels(G_longrun, pos, labels=labels,
                        font_size=9, font_weight='bold', ax=ax)

ax.set_title('LONG-RUN EQUILIBRIUM Network (Error Correction) - Rank=2\n(Red=Amplifying, Blue=Dampening; Width=Strength; Node size=Beta importance)',
             fontsize=14, fontweight='bold', pad=20)

from matplotlib.lines import Line2D
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
print("    Long-run network saved")

# Network Diagram 2: Short-run dynamics
print("\n[4] Creating short-run network...")

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

fig, ax = plt.subplots(figsize=(16, 12))
pos_sr = nx.spring_layout(G_shortrun, k=2, iterations=50, seed=42)

nx.draw_networkx_nodes(G_shortrun, pos_sr, node_size=2000,
                       node_color='lightblue', edgecolors='black',
                       linewidths=2, ax=ax)

edges_pos = [(u, v) for u, v, d in G_shortrun.edges(data=True) if d['direction'] > 0]
edges_neg = [(u, v) for u, v, d in G_shortrun.edges(data=True) if d['direction'] < 0]

weights_pos = [G_shortrun[u][v]['weight'] for u, v in edges_pos]
weights_neg = [G_shortrun[u][v]['weight'] for u, v in edges_neg]

if weights_pos or weights_neg:
    max_gamma = max([d['weight'] for u, v, d in G_shortrun.edges(data=True)])
    widths_pos = [w / max_gamma * 5 for w in weights_pos] if weights_pos else []
    widths_neg = [w / max_gamma * 5 for w in weights_neg] if weights_neg else []

    if edges_pos:
        nx.draw_networkx_edges(G_shortrun, pos_sr, edgelist=edges_pos,
                               width=widths_pos, edge_color='red', alpha=0.7,
                               arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)
    if edges_neg:
        nx.draw_networkx_edges(G_shortrun, pos_sr, edgelist=edges_neg,
                               width=widths_neg, edge_color='blue', alpha=0.7,
                               arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)

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
print("    Short-run network saved")

# Beta vectors visualization
print("\n[5] Creating beta vectors visualization...")

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
print("    Beta vectors visualization saved")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\nFiles added to {OUTPUT_DIR}:")
print("  - vecm_longrun_network_rank2.png")
print("  - vecm_shortrun_network_rank2.png")
print("  - vecm_cointegration_vectors_rank2.png")
print("  - vecm_influence_comparison_rank2.png (updated with axis labels)")
