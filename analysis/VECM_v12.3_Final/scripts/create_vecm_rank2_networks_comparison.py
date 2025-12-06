"""
Create VECM Rank=2 Network Diagrams with Visible Arrows
=========================================================
1. Regenerate long-run and short-run networks with visible arrows
2. Create side-by-side comparison graphic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from matplotlib.lines import Line2D

# Paths
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
INPUT_DIR = BASE_DIR / "VECM_Rank2_Final_Executive_Summary"
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

DISPLAY_NAMES_LONG = {
    'Junior_Enlisted_Z': 'Junior Enlisted\n(E-1 to E-4)',
    'Company_Grade_Officers_Z': 'Company Grade\n(O-1 to O-3)',
    'Field_Grade_Officers_Z': 'Field Grade\n(O-4 to O-5)',
    'GOFOs_Z': 'General/Flag\nOfficers',
    'Warrant_Officers_Z': 'Warrant\nOfficers',
    'Policy_Count_Log': 'Policy Volume\n(Log)',
    'Total_PAS_Z': 'Political\nAppointees (PAS)',
    'FOIA_Simple_Days_Z': 'FOIA Processing\nDelay'
}

print("=" * 80)
print("CREATING VECM RANK=2 NETWORK DIAGRAMS WITH VISIBLE ARROWS")
print("=" * 80)

# Load matrices
print("\n[1] Loading matrices...")
alpha_df = pd.read_excel(INPUT_DIR / "alpha_matrix_rank2.xlsx", index_col=0)
beta_df = pd.read_excel(INPUT_DIR / "beta_matrix_rank2.xlsx", index_col=0)
gamma_df = pd.read_excel(INPUT_DIR / "gamma_matrix_rank2.xlsx", index_col=0)
longrun_df = pd.read_excel(INPUT_DIR / "longrun_influence_rank2.xlsx", index_col=0)

print(f"    Alpha: {alpha_df.shape}")
print(f"    Beta: {beta_df.shape}")
print(f"    Gamma: {gamma_df.shape}")
print(f"    Long-run influence: {longrun_df.shape}")

# Calculate signed direction for long-run
print("\n[2] Calculating signed direction for long-run...")
signed_direction = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))
for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        signed_sum = 0
        for r in range(2):  # rank=2
            alpha_i = alpha_df.iloc[i, r]
            beta_j = beta_df.iloc[j, r]
            signed_sum += alpha_i * beta_j
        signed_direction[i, j] = np.sign(signed_sum)

print("    Signed direction calculated")

# ============================================================================
# CREATE SIDE-BY-SIDE COMPARISON GRAPHIC
# ============================================================================
print("\n[3] Creating side-by-side comparison graphic...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))

# ----------------------------------------------------------------------------
# LEFT: LONG-RUN NETWORK
# ----------------------------------------------------------------------------
print("    [3a] Creating long-run network...")

G_longrun = nx.DiGraph()

# Add nodes
beta_importance = np.abs(beta_df).sum(axis=1).values
beta_importance_normalized = (beta_importance / beta_importance.max()) * 3000 + 500

for i, var in enumerate(SELECTED_VARS):
    G_longrun.add_node(var, importance=beta_importance[i], size=beta_importance_normalized[i])

# Add edges
threshold = 0.15
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            magnitude = longrun_df.iloc[i, j]
            if magnitude > threshold:
                direction = np.sign(signed_direction[i, j])
                G_longrun.add_edge(var_from, var_to, weight=magnitude, direction=direction)

# Layout
pos = nx.circular_layout(G_longrun)

# Draw nodes
node_sizes = [G_longrun.nodes[node]['size'] for node in G_longrun.nodes()]
nx.draw_networkx_nodes(G_longrun, pos, node_size=node_sizes,
                       node_color='yellow', edgecolors='black',
                       linewidths=2.5, alpha=0.9, ax=ax1)

# Draw edges
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
                               min_source_margin=25, min_target_margin=25, ax=ax1)

    if edges_dampening:
        weights_damp = [G_longrun[u][v]['weight'] for u, v in edges_dampening]
        widths_damp = [3 + (w / max_weight) * 5 for w in weights_damp]
        nx.draw_networkx_edges(G_longrun, pos, edgelist=edges_dampening,
                               width=widths_damp, edge_color='blue', alpha=0.7,
                               arrows=True, arrowsize=25, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15',
                               min_source_margin=25, min_target_margin=25, ax=ax1)

# Draw labels
labels = {node: DISPLAY_NAMES_LONG[node] for node in G_longrun.nodes()}
nx.draw_networkx_labels(G_longrun, pos, labels=labels,
                        font_size=9, font_weight='bold', ax=ax1)

ax1.set_title('LONG-RUN EQUILIBRIUM\n(Error Correction)\n\nRed=Amplifying, Blue=Dampening',
             fontsize=13, fontweight='bold', pad=15)
ax1.axis('off')

# ----------------------------------------------------------------------------
# RIGHT: SHORT-RUN NETWORK
# ----------------------------------------------------------------------------
print("    [3b] Creating short-run network...")

G_shortrun = nx.DiGraph()

# Add nodes
for var in SELECTED_VARS:
    G_shortrun.add_node(var)

# Add edges
gamma_threshold = 0.15
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            coef = gamma_df.iloc[i, j]
            if abs(coef) > gamma_threshold:
                G_shortrun.add_edge(var_from, var_to, weight=abs(coef), direction=np.sign(coef))

# Layout (same positions for consistency)
pos_sr = nx.circular_layout(G_shortrun)

# Draw nodes
nx.draw_networkx_nodes(G_shortrun, pos_sr, node_size=2000,
                       node_color='lightblue', edgecolors='black',
                       linewidths=2.5, alpha=0.9, ax=ax2)

# Draw edges
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
                               min_source_margin=20, min_target_margin=20, ax=ax2)

    if edges_neg:
        weights_neg = [G_shortrun[u][v]['weight'] for u, v in edges_neg]
        widths_neg = [3 + (w / max_gamma) * 5 for w in weights_neg]
        nx.draw_networkx_edges(G_shortrun, pos_sr, edgelist=edges_neg,
                               width=widths_neg, edge_color='blue', alpha=0.7,
                               arrows=True, arrowsize=25, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15',
                               min_source_margin=20, min_target_margin=20, ax=ax2)

# Draw labels
labels_sr = {node: DISPLAY_NAMES_LONG[node] for node in G_shortrun.nodes()}
nx.draw_networkx_labels(G_shortrun, pos_sr, labels=labels_sr,
                        font_size=9, font_weight='bold', ax=ax2)

ax2.set_title('SHORT-RUN DYNAMICS\n(Year-to-Year VAR)\n\nRed=Amplifying, Blue=Dampening',
             fontsize=13, fontweight='bold', pad=15)
ax2.axis('off')

# Main title
fig.suptitle('VECM Network Comparison: Long-Run Equilibrium vs Short-Run Dynamics (Rank=2)\nArrow width = Relationship strength; Node size (left) = Beta importance',
             fontsize=15, fontweight='bold', y=0.98)

# Shared legend
legend_elements = [
    Line2D([0], [0], color='red', linewidth=4, label='Amplifying (+)', marker='>', markersize=10),
    Line2D([0], [0], color='blue', linewidth=4, label='Dampening (-)', marker='>', markersize=10)
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12,
           bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(OUTPUT_DIR / "vecm_network_comparison_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Side-by-side comparison saved!")

# ============================================================================
# REGENERATE INDIVIDUAL NETWORKS
# ============================================================================
print("\n[4] Regenerating individual long-run network...")

fig, ax = plt.subplots(figsize=(16, 14))

pos_new = nx.circular_layout(G_longrun)

nx.draw_networkx_nodes(G_longrun, pos_new, node_size=node_sizes,
                       node_color='yellow', edgecolors='black',
                       linewidths=2.5, alpha=0.9, ax=ax)

if edges_amplifying:
    nx.draw_networkx_edges(G_longrun, pos_new, edgelist=edges_amplifying,
                           width=widths_amp, edge_color='red', alpha=0.7,
                           arrows=True, arrowsize=25, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=25, min_target_margin=25, ax=ax)

if edges_dampening:
    nx.draw_networkx_edges(G_longrun, pos_new, edgelist=edges_dampening,
                           width=widths_damp, edge_color='blue', alpha=0.7,
                           arrows=True, arrowsize=25, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=25, min_target_margin=25, ax=ax)

nx.draw_networkx_labels(G_longrun, pos_new, labels=labels,
                        font_size=10, font_weight='bold', ax=ax)

ax.set_title('LONG-RUN EQUILIBRIUM Network (Error Correction) - Rank=2\n(Red=Amplifying, Blue=Dampening; Width=Strength; Node size=Beta importance)',
             fontsize=14, fontweight='bold', pad=20)

legend_elements = [
    Line2D([0], [0], color='red', linewidth=3, label='Amplifying (+)'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dampening (-)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
           markersize=10, markeredgecolor='black', markeredgewidth=2,
           label='Node size = Beta importance', linestyle='None')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_longrun_network_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Long-run network saved!")

print("\n[5] Regenerating individual short-run network...")

fig, ax = plt.subplots(figsize=(16, 14))

pos_sr_new = nx.circular_layout(G_shortrun)

nx.draw_networkx_nodes(G_shortrun, pos_sr_new, node_size=2000,
                       node_color='lightblue', edgecolors='black',
                       linewidths=2.5, alpha=0.9, ax=ax)

if edges_pos:
    nx.draw_networkx_edges(G_shortrun, pos_sr_new, edgelist=edges_pos,
                           width=widths_pos, edge_color='red', alpha=0.7,
                           arrows=True, arrowsize=25, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=20, min_target_margin=20, ax=ax)

if edges_neg:
    nx.draw_networkx_edges(G_shortrun, pos_sr_new, edgelist=edges_neg,
                           width=widths_neg, edge_color='blue', alpha=0.7,
                           arrows=True, arrowsize=25, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=20, min_target_margin=20, ax=ax)

nx.draw_networkx_labels(G_shortrun, pos_sr_new, labels=labels_sr,
                        font_size=10, font_weight='bold', ax=ax)

ax.set_title('SHORT-RUN DYNAMICS Network (Year-to-Year VAR) - Rank=2\n(Red=Amplifying, Blue=Dampening; Width=Coefficient strength)',
             fontsize=14, fontweight='bold', pad=20)

legend_sr = [
    Line2D([0], [0], color='red', linewidth=3, label='Amplifying (+)'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dampening (-)')
]
ax.legend(handles=legend_sr, loc='upper left', fontsize=11)

ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_shortrun_network_rank2.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Short-run network saved!")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\nFiles saved to: {OUTPUT_DIR}")
print("\nGenerated:")
print("  1. vecm_network_comparison_rank2.png (SIDE-BY-SIDE)")
print("  2. vecm_longrun_network_rank2.png (updated with visible arrows)")
print("  3. vecm_shortrun_network_rank2.png (updated with visible arrows)")
print("=" * 80)
