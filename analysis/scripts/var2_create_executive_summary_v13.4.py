"""
VAR(2) Executive Summary Visualizations - v13.4
================================================
Creates publication-quality visualizations:
1. Side-by-side coefficient heatmaps (Lag 1 | Lag 2)
2. Key relationships chart (combined strength across both lags)
3. Network diagrams with VISIBLE arrows for Lag 1 and Lag 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis")
INPUT_DIR = BASE_DIR / "VAR2_v13.4_Analysis"
OUTPUT_DIR = INPUT_DIR  # Save in same directory

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

DISPLAY_NAMES_NETWORK = {
    'Warrant_Officers_Z': 'Warrant Officers',
    'Policy_Count_LogZ': 'Policy Volume',
    'Company_Grade_Officers_Z': 'Company Grade (O-1 to O-3)',
    'Total_PAS_Z': 'Political Appointees',
    'FOIA_Simple_Days_Z': 'FOIA Delay',
    'Junior_Enlisted_Z': 'Junior Enlisted (E-1 to E-4)',
    'Field_Grade_Officers_Z': 'Field Grade (O-4 to O-5)',
    'Total_Civilians_Z': 'Civilian Personnel'
}

DISPLAY_NAMES_SHORT = {
    'Warrant_Officers_Z': 'Warrant Officers',
    'Policy_Count_LogZ': 'Policy Volume',
    'Company_Grade_Officers_Z': 'Company Grade (O-1 to O-3)',
    'Total_PAS_Z': 'Political Appointees',
    'FOIA_Simple_Days_Z': 'FOIA Delay',
    'Junior_Enlisted_Z': 'Junior Enlisted (E-1 to E-4)',
    'Field_Grade_Officers_Z': 'Field Grade (O-4 to O-5)',
    'Total_Civilians_Z': 'Civilian Personnel'
}

print("=" * 80)
print("VAR(2) EXECUTIVE SUMMARY VISUALIZATIONS - V13.4")
print("=" * 80)

# Load coefficient matrices
print("\n[1] Loading coefficient matrices...")
lag1_df = pd.read_excel(INPUT_DIR / "var2_lag1_coefficients.xlsx", index_col=0)
lag2_df = pd.read_excel(INPUT_DIR / "var2_lag2_coefficients.xlsx", index_col=0)

print(f"    Lag 1: {lag1_df.shape}")
print(f"    Lag 2: {lag2_df.shape}")

# ============================================================================
# VISUALIZATION 1: SIDE-BY-SIDE COEFFICIENT HEATMAPS
# ============================================================================
print("\n[2] Creating side-by-side coefficient heatmaps...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Prepare display names for axes
display_names = [DISPLAY_NAMES_SHORT[var] for var in SELECTED_VARS]

# Lag 1 heatmap
sns.heatmap(lag1_df.values, annot=True, fmt='.3f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            cbar_kws={'label': 'Coefficient'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='white', ax=ax1)
ax1.set_title('Lag 1 Coefficients (t-1 → t)', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Effect on (→)', fontsize=10)
ax1.set_ylabel('Effect from (←)', fontsize=10)

# Lag 2 heatmap
sns.heatmap(lag2_df.values, annot=True, fmt='.3f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            cbar_kws={'label': 'Coefficient'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='white', ax=ax2)
ax2.set_title('Lag 2 Coefficients (t-2 → t)', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Effect on (→)', fontsize=10)
ax2.set_ylabel('', fontsize=10)  # Remove ylabel from second plot

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "var2_coefficient_heatmaps.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Side-by-side heatmaps saved")

# ============================================================================
# VISUALIZATION 2: KEY RELATIONSHIPS (COMBINED STRENGTH)
# ============================================================================
print("\n[3] Creating key relationships chart...")

# Calculate combined strength: |Lag1| + |Lag2|
relationships = []
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:  # Exclude diagonal
            lag1_coef = lag1_df.iloc[i, j]
            lag2_coef = lag2_df.iloc[i, j]
            combined_strength = abs(lag1_coef) + abs(lag2_coef)

            if combined_strength > 0.3:  # Only show significant relationships
                # Determine overall direction
                if abs(lag1_coef) > abs(lag2_coef):
                    direction = 'Amplifying' if lag1_coef > 0 else 'Dampening'
                else:
                    direction = 'Amplifying' if lag2_coef > 0 else 'Dampening'

                relationships.append({
                    'From': DISPLAY_NAMES_SHORT[var_from],
                    'To': DISPLAY_NAMES_SHORT[var_to],
                    'Combined_Strength': combined_strength,
                    'Direction': direction,
                    'Lag1_Coef': lag1_coef,
                    'Lag2_Coef': lag2_coef
                })

relationships_df = pd.DataFrame(relationships).sort_values('Combined_Strength', ascending=False)

# Plot top 15
fig, ax = plt.subplots(figsize=(14, 10))

top_15 = relationships_df.head(15)
y_pos = range(len(top_15))

colors = ['#E57373' if d == 'Amplifying' else '#64B5F6' for d in top_15['Direction']]

bars = ax.barh(y_pos, top_15['Combined_Strength'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Create labels
labels = [f"{row['From']} → {row['To']}" for _, row in top_15.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Combined Coefficient Strength (|Lag1| + |Lag2|)', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Strongest Dynamic Relationships in VAR(2)\n(Red=Amplifying, Blue=Dampening)',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (_, row) in enumerate(top_15.iterrows()):
    ax.text(row['Combined_Strength'] + 0.1, i, f"{row['Combined_Strength']:.2f}",
            va='center', fontsize=9, fontweight='bold')

ax.invert_yaxis()  # Highest at top
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "var2_key_relationships.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Key relationships chart saved")

# ============================================================================
# VISUALIZATION 3: NETWORK DIAGRAM - LAG 1
# ============================================================================
print("\n[4] Creating Lag 1 network diagram...")

G_lag1 = nx.DiGraph()

# Add nodes
for var in SELECTED_VARS:
    G_lag1.add_node(var)

# Add edges (threshold for visibility)
threshold = 0.15
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            coef = lag1_df.iloc[i, j]
            if abs(coef) > threshold:
                G_lag1.add_edge(var_from, var_to, weight=abs(coef), coef=coef)

# Create figure
fig, ax = plt.subplots(figsize=(16, 14))

# Circular layout
pos = nx.circular_layout(G_lag1)

# Node sizes by degree
degrees = dict(G_lag1.degree())
node_sizes = [1500 + degrees[node] * 500 for node in G_lag1.nodes()]

# Draw nodes
nx.draw_networkx_nodes(G_lag1, pos,
                       node_size=node_sizes,
                       node_color='lightblue',
                       edgecolors='black',
                       linewidths=2.5,
                       alpha=0.9,
                       ax=ax)

# Draw edges by direction
edges_pos = [(u, v) for u, v, d in G_lag1.edges(data=True) if d['coef'] > 0]
edges_neg = [(u, v) for u, v, d in G_lag1.edges(data=True) if d['coef'] < 0]

if edges_pos:
    weights_pos = [G_lag1[u][v]['weight'] for u, v in edges_pos]
    max_weight = max(weights_pos) if weights_pos else 1
    widths_pos = [3 + (w / max_weight) * 5 for w in weights_pos]

    nx.draw_networkx_edges(G_lag1, pos,
                           edgelist=edges_pos,
                           width=widths_pos,
                           edge_color='red',
                           alpha=0.7,
                           arrows=True,
                           arrowsize=25,
                           arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=20,
                           min_target_margin=20,
                           ax=ax)

if edges_neg:
    weights_neg = [G_lag1[u][v]['weight'] for u, v in edges_neg]
    max_weight_neg = max(weights_neg) if weights_neg else 1
    widths_neg = [3 + (w / max_weight_neg) * 5 for w in weights_neg]

    nx.draw_networkx_edges(G_lag1, pos,
                           edgelist=edges_neg,
                           width=widths_neg,
                           edge_color='blue',
                           alpha=0.7,
                           arrows=True,
                           arrowsize=25,
                           arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=20,
                           min_target_margin=20,
                           ax=ax)

# Draw labels
labels = {node: DISPLAY_NAMES_NETWORK[node] for node in G_lag1.nodes()}
nx.draw_networkx_labels(G_lag1, pos, labels=labels,
                        font_size=11, font_weight='bold',
                        font_color='black', ax=ax)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=4, label='Amplifying (+)', marker='>', markersize=8),
    Line2D([0], [0], color='blue', linewidth=4, label='Dampening (-)', marker='>', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
           markersize=12, markeredgecolor='black', markeredgewidth=2,
           label='Node size = Degree', linestyle='None')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, frameon=True, fancybox=True)

ax.set_title('VAR(2) Network: Lag 1 Coefficients (t-1 → t)\n(Red arrows=Amplifying (+), Blue arrows=Dampening (-); Width=Strength)',
             fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "var2_network_lag1.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Lag 1 network saved")

# ============================================================================
# VISUALIZATION 4: NETWORK DIAGRAM - LAG 2
# ============================================================================
print("\n[5] Creating Lag 2 network diagram...")

G_lag2 = nx.DiGraph()

# Add nodes
for var in SELECTED_VARS:
    G_lag2.add_node(var)

# Add edges
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            coef = lag2_df.iloc[i, j]
            if abs(coef) > threshold:
                G_lag2.add_edge(var_from, var_to, weight=abs(coef), coef=coef)

# Create figure
fig, ax = plt.subplots(figsize=(16, 14))

# Circular layout (same seed for consistency)
pos2 = nx.circular_layout(G_lag2)

# Node sizes by degree
degrees2 = dict(G_lag2.degree())
node_sizes2 = [1500 + degrees2[node] * 500 for node in G_lag2.nodes()]

# Draw nodes
nx.draw_networkx_nodes(G_lag2, pos2,
                       node_size=node_sizes2,
                       node_color='lightgreen',
                       edgecolors='black',
                       linewidths=2.5,
                       alpha=0.9,
                       ax=ax)

# Draw edges by direction
edges_pos2 = [(u, v) for u, v, d in G_lag2.edges(data=True) if d['coef'] > 0]
edges_neg2 = [(u, v) for u, v, d in G_lag2.edges(data=True) if d['coef'] < 0]

if edges_pos2:
    weights_pos2 = [G_lag2[u][v]['weight'] for u, v in edges_pos2]
    max_weight2 = max(weights_pos2) if weights_pos2 else 1
    widths_pos2 = [3 + (w / max_weight2) * 5 for w in weights_pos2]

    nx.draw_networkx_edges(G_lag2, pos2,
                           edgelist=edges_pos2,
                           width=widths_pos2,
                           edge_color='red',
                           alpha=0.7,
                           arrows=True,
                           arrowsize=25,
                           arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=20,
                           min_target_margin=20,
                           ax=ax)

if edges_neg2:
    weights_neg2 = [G_lag2[u][v]['weight'] for u, v in edges_neg2]
    max_weight_neg2 = max(weights_neg2) if weights_neg2 else 1
    widths_neg2 = [3 + (w / max_weight_neg2) * 5 for w in weights_neg2]

    nx.draw_networkx_edges(G_lag2, pos2,
                           edgelist=edges_neg2,
                           width=widths_neg2,
                           edge_color='blue',
                           alpha=0.7,
                           arrows=True,
                           arrowsize=25,
                           arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=20,
                           min_target_margin=20,
                           ax=ax)

# Draw labels
labels2 = {node: DISPLAY_NAMES_NETWORK[node] for node in G_lag2.nodes()}
nx.draw_networkx_labels(G_lag2, pos2, labels=labels2,
                        font_size=11, font_weight='bold',
                        font_color='black', ax=ax)

# Legend
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, frameon=True, fancybox=True)

ax.set_title('VAR(2) Network: Lag 2 Coefficients (t-2 → t)\n(Red arrows=Amplifying (+), Blue arrows=Dampening (-); Width=Strength)',
             fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "var2_network_lag2.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Lag 2 network saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY VISUALIZATIONS COMPLETE!")
print("=" * 80)
print(f"\nFiles saved to: {OUTPUT_DIR}")
print("\nGenerated visualizations:")
print("  1. var2_coefficient_heatmaps.png (side-by-side Lag 1 | Lag 2)")
print("  2. var2_key_relationships.png (top 15 by combined strength)")
print("  3. var2_network_lag1.png (circular layout with visible arrows)")
print("  4. var2_network_lag2.png (circular layout with visible arrows)")
print("=" * 80)
