"""
VECM Network Diagrams - Rank=1
================================
Generate network visualizations showing:
1. Short-run dynamics (Gamma coefficients)
2. Long-run equilibrium (Error correction mechanism)
3. Cointegration structure (Beta vector importance)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
INPUT_DIR = BASE_DIR / "VECM_Rank1_Final_Executive_Summary"
OUTPUT_DIR = INPUT_DIR

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

# Display names for network
DISPLAY_NAMES = {
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
print("GENERATING NETWORK DIAGRAMS FOR RANK=1 MODEL")
print("=" * 80)

# Load matrices
print("\n[1] LOADING COEFFICIENT MATRICES...")
alpha_df = pd.read_excel(INPUT_DIR / "alpha_matrix_rank1.xlsx", index_col=0)
beta_df = pd.read_excel(INPUT_DIR / "beta_matrix_rank1.xlsx", index_col=0)
gamma_df = pd.read_excel(INPUT_DIR / "gamma_matrix_rank1.xlsx", index_col=0)
longrun_df = pd.read_excel(INPUT_DIR / "longrun_influence_rank1.xlsx", index_col=0)

print(f"    Alpha: {alpha_df.shape}")
print(f"    Beta: {beta_df.shape}")
print(f"    Gamma: {gamma_df.shape}")
print(f"    Long-run influence: {longrun_df.shape}")

# =============================================================================
# NETWORK 1: LONG-RUN EQUILIBRIUM (Error Correction)
# =============================================================================
print("\n[2] CREATING LONG-RUN NETWORK...")

G_longrun = nx.DiGraph()

# Add nodes with size based on beta importance (absolute value)
beta_importance = np.abs(beta_df.iloc[:, 0].values)
beta_importance_normalized = (beta_importance / beta_importance.max()) * 3000 + 500

for i, var in enumerate(SELECTED_VARS):
    G_longrun.add_node(var,
                       label=DISPLAY_NAMES[var],
                       importance=beta_importance[i],
                       size=beta_importance_normalized[i])

# Add edges with strength based on long-run influence
# Direction based on sign of alpha * beta
threshold = 0.1  # Only show influences > 0.1

for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:  # No self-loops
            magnitude = longrun_df.iloc[i, j]
            if magnitude > threshold:
                # Calculate direction
                alpha_i = alpha_df.iloc[i, 0]
                beta_j = beta_df.iloc[j, 0]
                direction = np.sign(alpha_i * beta_j)

                G_longrun.add_edge(var_from, var_to,
                                   weight=magnitude,
                                   direction=direction)

# Draw network
fig, ax = plt.subplots(figsize=(16, 12))

pos = nx.spring_layout(G_longrun, k=2, iterations=50, seed=42)

# Draw nodes
node_sizes = [G_longrun.nodes[node]['size'] for node in G_longrun.nodes()]
nx.draw_networkx_nodes(G_longrun, pos,
                       node_size=node_sizes,
                       node_color='yellow',
                       edgecolors='black',
                       linewidths=2,
                       ax=ax)

# Draw edges (colored by direction)
edges_amplifying = [(u, v) for u, v, d in G_longrun.edges(data=True) if d['direction'] > 0]
edges_dampening = [(u, v) for u, v, d in G_longrun.edges(data=True) if d['direction'] < 0]

weights_amplifying = [G_longrun[u][v]['weight'] for u, v in edges_amplifying]
weights_dampening = [G_longrun[u][v]['weight'] for u, v in edges_dampening]

# Normalize widths
max_weight = max([d['weight'] for u, v, d in G_longrun.edges(data=True)])
widths_amplifying = [w / max_weight * 5 for w in weights_amplifying]
widths_dampening = [w / max_weight * 5 for w in weights_dampening]

nx.draw_networkx_edges(G_longrun, pos,
                       edgelist=edges_amplifying,
                       width=widths_amplifying,
                       edge_color='red',
                       alpha=0.7,
                       arrowsize=20,
                       connectionstyle='arc3,rad=0.1',
                       ax=ax)

nx.draw_networkx_edges(G_longrun, pos,
                       edgelist=edges_dampening,
                       width=widths_dampening,
                       edge_color='blue',
                       alpha=0.7,
                       arrowsize=20,
                       connectionstyle='arc3,rad=0.1',
                       ax=ax)

# Draw labels
labels = {node: G_longrun.nodes[node]['label'] for node in G_longrun.nodes()}
nx.draw_networkx_labels(G_longrun, pos,
                        labels=labels,
                        font_size=9,
                        font_weight='bold',
                        ax=ax)

ax.set_title('LONG-RUN EQUILIBRIUM Network (Error Correction Mechanism)\n(Red arrows=Amplifying (+), Blue=Dampening (-); Width=Strength of α×β)',
             fontsize=14, fontweight='bold', pad=20)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=3, label='Amplifying LR influence (+)'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dampening LR influence (-)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
           markersize=10, markeredgecolor='black', markeredgewidth=2,
           label='Node size = Beta importance', linestyle='None')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
          frameon=True, fancybox=True, shadow=True)

ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_longrun_network_rank1.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Long-run network saved")

# =============================================================================
# NETWORK 2: SHORT-RUN DYNAMICS (Gamma)
# =============================================================================
print("\n[3] CREATING SHORT-RUN NETWORK...")

G_shortrun = nx.DiGraph()

# Add nodes (all same size for short-run)
for var in SELECTED_VARS:
    G_shortrun.add_node(var, label=DISPLAY_NAMES[var])

# Add edges based on gamma coefficients
gamma_threshold = 0.15

for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            coef = gamma_df.iloc[i, j]
            if abs(coef) > gamma_threshold:
                G_shortrun.add_edge(var_from, var_to,
                                    weight=abs(coef),
                                    direction=np.sign(coef))

# Draw network
fig, ax = plt.subplots(figsize=(16, 12))

pos_shortrun = nx.spring_layout(G_shortrun, k=2, iterations=50, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G_shortrun, pos_shortrun,
                       node_size=2000,
                       node_color='lightblue',
                       edgecolors='black',
                       linewidths=2,
                       ax=ax)

# Draw edges
edges_positive = [(u, v) for u, v, d in G_shortrun.edges(data=True) if d['direction'] > 0]
edges_negative = [(u, v) for u, v, d in G_shortrun.edges(data=True) if d['direction'] < 0]

weights_positive = [G_shortrun[u][v]['weight'] for u, v in edges_positive]
weights_negative = [G_shortrun[u][v]['weight'] for u, v in edges_negative]

if weights_positive or weights_negative:
    max_gamma = max([d['weight'] for u, v, d in G_shortrun.edges(data=True)])
    widths_positive = [w / max_gamma * 5 for w in weights_positive] if weights_positive else []
    widths_negative = [w / max_gamma * 5 for w in weights_negative] if weights_negative else []
else:
    widths_positive = []
    widths_negative = []

if edges_positive:
    nx.draw_networkx_edges(G_shortrun, pos_shortrun,
                           edgelist=edges_positive,
                           width=widths_positive,
                           edge_color='darkgreen',
                           alpha=0.7,
                           arrowsize=20,
                           connectionstyle='arc3,rad=0.1',
                           ax=ax)

if edges_negative:
    nx.draw_networkx_edges(G_shortrun, pos_shortrun,
                           edgelist=edges_negative,
                           width=widths_negative,
                           edge_color='purple',
                           alpha=0.7,
                           arrowsize=20,
                           connectionstyle='arc3,rad=0.1',
                           ax=ax)

# Draw labels
labels_shortrun = {node: G_shortrun.nodes[node]['label'] for node in G_shortrun.nodes()}
nx.draw_networkx_labels(G_shortrun, pos_shortrun,
                        labels=labels_shortrun,
                        font_size=9,
                        font_weight='bold',
                        ax=ax)

ax.set_title('SHORT-RUN DYNAMICS Network (Year-to-Year VAR)\n(Green=Positive (+), Purple=Negative (-); Width=Coefficient strength)',
             fontsize=14, fontweight='bold', pad=20)

# Legend
legend_elements_sr = [
    Line2D([0], [0], color='darkgreen', linewidth=3, label='Positive SR effect (+)'),
    Line2D([0], [0], color='purple', linewidth=3, label='Negative SR effect (-)')
]
ax.legend(handles=legend_elements_sr, loc='upper left', fontsize=10,
          frameon=True, fancybox=True, shadow=True)

ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_shortrun_network_rank1.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Short-run network saved")

# =============================================================================
# NETWORK 3: COINTEGRATION VECTOR (Beta weights)
# =============================================================================
print("\n[4] CREATING COINTEGRATION VECTOR NETWORK...")

fig, ax = plt.subplots(figsize=(14, 10))

# Create bar chart showing beta weights
beta_values = beta_df.iloc[:, 0].values
colors = ['red' if b > 0 else 'blue' for b in beta_values]

bars = ax.barh(range(len(SELECTED_VARS)), beta_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (var, val) in enumerate(zip(SELECTED_VARS, beta_values)):
    ax.text(val + 0.02 if val > 0 else val - 0.02,
            i,
            f'{val:.3f}',
            va='center',
            ha='left' if val > 0 else 'right',
            fontsize=10,
            fontweight='bold')

ax.set_yticks(range(len(SELECTED_VARS)))
ax.set_yticklabels([DISPLAY_NAMES[var] for var in SELECTED_VARS], fontsize=11)
ax.set_xlabel('Beta Coefficient (Weight in Equilibrium)', fontsize=12, fontweight='bold')
ax.set_title('COINTEGRATION VECTOR (Beta) - RANK=1\nOne Equilibrium Relationship Binding All Variables\n(Red=Positive, Blue=Negative)',
             fontsize=14, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linewidth=1, linestyle='--')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_cointegration_network_rank1.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Cointegration vector diagram saved")

print("\n" + "=" * 80)
print("NETWORK DIAGRAMS COMPLETE")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)
print("\nFiles created:")
print("  - vecm_longrun_network_rank1.png")
print("  - vecm_shortrun_network_rank1.png")
print("  - vecm_cointegration_network_rank1.png")
