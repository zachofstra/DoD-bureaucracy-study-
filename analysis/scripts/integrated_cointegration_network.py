"""
Integrated Cointegrating Vector Network - All 4 Vectors in One Diagram
Shows the complete equilibrium structure of DoD bureaucracy

Visual encoding:
- All 7 variables as nodes (color-coded by category)
- Edges from all 4 vectors combined
- Edge style distinguishes vectors (solid, dashed, dotted, dash-dot)
- Edge color shows relationship type (red=amplifying, blue=dampening)
- Edge labels show vector number and coefficient
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("CREATING INTEGRATED COINTEGRATION NETWORK - ALL 4 VECTORS")
print("=" * 100)

# =============================================================================
# LOAD BETA MATRIX
# =============================================================================
print("\n[1/3] Loading cointegrating vectors...")

beta_df = pd.read_excel('data/analysis/VECM_LAG2/beta_cointegrating_vectors.xlsx', index_col=0)

# Clean variable names
variables = [v.replace('_Z', '').replace('_', ' ') for v in beta_df.index]
var_mapping = dict(zip(beta_df.index, variables))

print(f"  Variables: {len(beta_df)}")
print(f"  Vectors: {len(beta_df.columns)}")

# =============================================================================
# DEFINE NODE PROPERTIES
# =============================================================================
def get_node_properties(var_name):
    """Get category and color for a variable"""
    if 'Enlisted' in var_name:
        return 'Enlisted Personnel', '#3498db'
    elif 'Officers' in var_name or 'GOFO' in var_name:
        return 'Officer Personnel', '#e74c3c'
    elif 'Civilian' in var_name or 'PAS' in var_name or 'FOIA' in var_name or 'Policy' in var_name:
        return 'Bureaucratic Measures', '#2ecc71'
    else:
        return 'Other', '#95a5a6'

# =============================================================================
# BUILD INTEGRATED NETWORK
# =============================================================================
print("\n[2/3] Building integrated network with all 4 vectors...")

G = nx.MultiDiGraph()

# Add all variables as nodes
for orig_var, clean_var in var_mapping.items():
    category, color = get_node_properties(clean_var)
    G.add_node(clean_var, category=category, color=color)

# Edge styles for each vector
vector_styles = {
    'Vector_1': 'solid',
    'Vector_2': 'dashed',
    'Vector_3': 'dotted',
    'Vector_4': 'dashdot'
}

threshold = 0.15  # Only show significant relationships

edges_added = 0

# Add edges from all 4 vectors
for vec_idx, vec_name in enumerate(beta_df.columns):
    vec_num = vec_idx + 1
    coeffs = beta_df[vec_name]

    # Find significant variables
    significant_vars = []
    for orig_var, clean_var in var_mapping.items():
        coef = coeffs[orig_var]
        if abs(coef) > threshold:
            significant_vars.append((clean_var, coef))

    # Sort by absolute coefficient
    significant_vars.sort(key=lambda x: abs(x[1]), reverse=True)

    if len(significant_vars) > 0:
        # Dominant variable
        dom_var, dom_coef = significant_vars[0]

        # Create edges from dominant to others
        for clean_var, coef in significant_vars[1:]:
            # Determine edge color (amplifying vs dampening)
            if np.sign(dom_coef) == np.sign(coef):
                edge_color = '#e74c3c'  # Red = amplifying
                relationship = '+'
            else:
                edge_color = '#3498db'  # Blue = dampening
                relationship = '−'

            # Edge already exists? Add to multi-edge
            edge_key = f"V{vec_num}"

            G.add_edge(dom_var, clean_var,
                      key=edge_key,
                      vector=vec_num,
                      vector_name=vec_name,
                      color=edge_color,
                      coef=coef,
                      relationship=relationship,
                      style=vector_styles[vec_name],
                      width=abs(coef) * 2.5)
            edges_added += 1

    print(f"  Vector {vec_num}: {len(significant_vars)} variables, {len(significant_vars)-1} edges")

print(f"\n  Total nodes: {G.number_of_nodes()}")
print(f"  Total edges: {G.number_of_edges()}")

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================
print("\n[3/3] Creating integrated network visualization...")

fig, ax = plt.subplots(1, 1, figsize=(24, 20), facecolor='white')

# Layout - circular for clarity
pos = nx.spring_layout(G, k=3, iterations=100, seed=42, scale=2)

# Manually adjust layout for better spacing
# Group by category
enlisted_vars = [v for v in variables if 'Enlisted' in v]
officer_vars = [v for v in variables if 'Officers' in v or 'GOFO' in v]
bureau_vars = [v for v in variables if v not in enlisted_vars and v not in officer_vars]

# Circular layout by category
n = len(variables)
for i, var in enumerate(enlisted_vars):
    angle = 2 * np.pi * i / len(enlisted_vars) - np.pi/2
    pos[var] = (3 * np.cos(angle), 3 * np.sin(angle) + 2)

for i, var in enumerate(officer_vars):
    angle = 2 * np.pi * i / len(officer_vars) + np.pi/6
    pos[var] = (3.5 * np.cos(angle) - 1, 3.5 * np.sin(angle) - 1)

for i, var in enumerate(bureau_vars):
    angle = 2 * np.pi * i / len(bureau_vars) + np.pi
    pos[var] = (3 * np.cos(angle) + 1, 3 * np.sin(angle))

# Draw edges by vector (to maintain style)
for vec_idx, vec_name in enumerate(beta_df.columns):
    vec_num = vec_idx + 1

    # Get edges for this vector
    vector_edges = [(u, v) for u, v, k, d in G.edges(data=True, keys=True) if d.get('vector') == vec_num]

    if vector_edges:
        edge_colors = [G[u][v][f'V{vec_num}']['color'] for u, v in vector_edges]
        edge_widths = [G[u][v][f'V{vec_num}']['width'] for u, v in vector_edges]
        edge_style = vector_styles[vec_name]

        nx.draw_networkx_edges(G, pos,
                              edgelist=vector_edges,
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.6,
                              arrowsize=25,
                              arrowstyle='-|>',
                              style=edge_style,
                              connectionstyle=f'arc3,rad={0.1 + vec_idx*0.05}',
                              node_size=3000,
                              ax=ax)

# Draw edge labels (showing vector number and coefficient)
edge_labels = {}
for u, v, k, d in G.edges(data=True, keys=True):
    vec_num = d['vector']
    coef = d['coef']
    if (u, v) not in edge_labels:
        edge_labels[(u, v)] = f"V{vec_num}:{coef:.2f}"
    else:
        edge_labels[(u, v)] += f"\nV{vec_num}:{coef:.2f}"

nx.draw_networkx_edge_labels(G, pos, edge_labels,
                            font_size=8,
                            font_weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='white',
                                    alpha=0.85,
                                    edgecolor='gray',
                                    linewidth=1),
                            ax=ax)

# Draw nodes
for node in G.nodes():
    node_color = G.nodes[node]['color']
    node_size = 4000

    nx.draw_networkx_nodes(G, pos,
                          nodelist=[node],
                          node_color=node_color,
                          node_size=node_size,
                          edgecolors='black',
                          linewidths=4,
                          ax=ax)

# Node labels
nx.draw_networkx_labels(G, pos,
                       font_size=11,
                       font_weight='bold',
                       ax=ax)

# Title
ax.set_title('Integrated Cointegrating Vector Network - VECM Lag 2\n' +
            'Complete Equilibrium Structure of DoD Bureaucracy (1987-2024)\n' +
            'All 4 Cointegrating Relationships Combined\n\n' +
            'Edge style = Vector number | Edge color = Red (amplifying) / Blue (dampening)\n' +
            'Edge labels = V#:coefficient | Arrows show influence direction',
            fontsize=18, fontweight='bold', pad=30)

# Comprehensive legend
legend_elements = [
    # Node categories
    Patch(facecolor='#3498db', label='Enlisted Personnel', edgecolor='black', linewidth=2),
    Patch(facecolor='#e74c3c', label='Officer Personnel', edgecolor='black', linewidth=2),
    Patch(facecolor='#2ecc71', label='Bureaucratic Measures', edgecolor='black', linewidth=2),
    # Edge colors
    Line2D([0], [0], color='#e74c3c', linewidth=4, label='Amplifying (+) - moves together'),
    Line2D([0], [0], color='#3498db', linewidth=4, label='Dampening (−) - moves inversely'),
    # Vector styles
    Line2D([0], [0], color='black', linewidth=3, linestyle='solid', label='Vector 1: Teeth-to-Tail Shift'),
    Line2D([0], [0], color='black', linewidth=3, linestyle='dashed', label='Vector 2: Bureaucratic Delay'),
    Line2D([0], [0], color='black', linewidth=3, linestyle='dotted', label='Vector 3: Political Appointee Trade-off'),
    Line2D([0], [0], color='black', linewidth=3, linestyle='dashdot', label='Vector 4: Civilianization'),
]

ax.legend(handles=legend_elements,
         loc='upper left',
         fontsize=11,
         framealpha=0.95,
         title='Network Key',
         title_fontsize=12,
         ncol=1)

# Add interpretation box
interpretation = """
INTEGRATED EQUILIBRIUM STRUCTURE:
• 4 cointegrating vectors shown simultaneously
• Each vector has distinct line style
• Multiple edges between nodes = multiple equilibria
• Red edges: Variables amplify together
• Blue edges: Variables dampen each other
• Edge labels: V# = vector number, value = coefficient

KEY INSIGHT:
This integrated view reveals how variables participate
in MULTIPLE equilibria simultaneously - the complete
"Iron Cage" structure with 4 interconnected constraints.
"""

ax.text(0.98, 0.02, interpretation,
       transform=ax.transAxes,
       fontsize=10,
       verticalalignment='bottom',
       horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
       family='monospace')

ax.axis('off')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

plt.tight_layout()

output_path = 'data/analysis/VECM_LAG2/integrated_cointegration_network.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n  [OK] Integrated network saved: {output_path}")

# =============================================================================
# PRINT SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("INTEGRATED NETWORK SUMMARY")
print("=" * 100)

print("\nEdge Count by Vector:")
for vec_idx, vec_name in enumerate(beta_df.columns):
    vec_num = vec_idx + 1
    vec_edges = [(u, v) for u, v, k, d in G.edges(data=True, keys=True) if d.get('vector') == vec_num]
    print(f"  Vector {vec_num} ({vec_name}): {len(vec_edges)} edges")

print(f"\nTotal unique edges: {G.number_of_edges()}")
print(f"Network density: {nx.density(G):.3f}")

# Identify most connected nodes
degree_centrality = nx.degree_centrality(G)
sorted_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

print("\nMost Connected Variables (highest degree centrality):")
for var, centrality in sorted_centrality[:5]:
    in_deg = G.in_degree(var)
    out_deg = G.out_degree(var)
    print(f"  {var:30s} - Centrality: {centrality:.3f} (In: {in_deg}, Out: {out_deg})")

print("\n" + "=" * 100)
print("INTEGRATED COINTEGRATION NETWORK COMPLETE")
print("=" * 100)
print("\nFile generated:")
print("  integrated_cointegration_network.png")
print("\nThis diagram shows:")
print("  - All 4 cointegrating vectors on one network")
print("  - Complete equilibrium structure")
print("  - How variables participate in multiple equilibria")
print("  - The interconnected 'Iron Cage' of bureaucratic constraints")
print("=" * 100)
