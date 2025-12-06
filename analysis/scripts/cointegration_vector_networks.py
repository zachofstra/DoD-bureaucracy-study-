"""
Cointegrating Vector Networks - VECM Lag 2
Creates separate influence network for each of the 4 cointegrating vectors

Each network shows:
- Variables as nodes (color-coded by category)
- Relationships from beta coefficients
- Coefficient values labeled on edges
- Direction and magnitude of equilibrium influence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("CREATING COINTEGRATING VECTOR NETWORKS - 4 EQUILIBRIUM RELATIONSHIPS")
print("=" * 100)

# =============================================================================
# LOAD BETA MATRIX
# =============================================================================
print("\n[1/3] Loading cointegrating vectors (beta matrix)...")

beta_df = pd.read_excel('data/analysis/VECM_LAG2/beta_cointegrating_vectors.xlsx', index_col=0)

print(f"  Variables: {len(beta_df)}")
print(f"  Cointegrating vectors: {len(beta_df.columns)}")

# Clean variable names
variables = [v.replace('_Z', '').replace('_', ' ') for v in beta_df.index]
var_mapping = dict(zip(beta_df.index, variables))

# =============================================================================
# DEFINE NODE CATEGORIES AND COLORS
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
# CREATE NETWORK FOR EACH VECTOR
# =============================================================================
print("\n[2/3] Creating individual vector networks...")

fig, axes = plt.subplots(2, 2, figsize=(24, 20), facecolor='white')
axes = axes.flatten()

for vec_idx in range(len(beta_df.columns)):
    vec_name = beta_df.columns[vec_idx]
    vec_num = vec_idx + 1

    print(f"\n  Processing {vec_name} (Vector {vec_num})...")

    ax = axes[vec_idx]

    # Get coefficients for this vector
    coeffs = beta_df[vec_name]

    # Create directed graph
    G = nx.DiGraph()

    # Add all variables as nodes
    for orig_var, clean_var in var_mapping.items():
        category, color = get_node_properties(clean_var)
        coef = coeffs[orig_var]

        G.add_node(clean_var,
                  category=category,
                  color=color,
                  coefficient=coef)

    # Add edges based on coefficient relationships
    # Connect variables with significant coefficients
    # Edge from var_i to var_j if both have |coef| > 0.1
    threshold = 0.1

    significant_vars = []
    for orig_var, clean_var in var_mapping.items():
        coef = abs(coeffs[orig_var])
        if coef > threshold:
            significant_vars.append((clean_var, coeffs[orig_var]))

    # Sort by absolute coefficient value
    significant_vars.sort(key=lambda x: abs(x[1]), reverse=True)

    # Create edges between significant variables
    for i, (var1, coef1) in enumerate(significant_vars):
        for j, (var2, coef2) in enumerate(significant_vars):
            if i != j:
                # Edge weight based on product of coefficients
                edge_weight = abs(coef1 * coef2)

                # Direction: positive coefficient "drives" the relationship
                if abs(coef1) > abs(coef2):
                    # Edge color based on sign of relationship
                    if np.sign(coef1) == np.sign(coef2):
                        edge_color = '#e74c3c'  # Red = amplifying (same direction)
                        relationship = 'positive'
                    else:
                        edge_color = '#3498db'  # Blue = dampening (opposite direction)
                        relationship = 'negative'

                    # Only add edge if weight is significant
                    if edge_weight > 0.3:
                        G.add_edge(var1, var2,
                                  weight=edge_weight,
                                  color=edge_color,
                                  coef1=coef1,
                                  coef2=coef2,
                                  relationship=relationship)

    # Alternative: Create star network from most dominant variable
    # Find most dominant variable
    most_dominant = significant_vars[0] if significant_vars else None

    if most_dominant:
        dom_var, dom_coef = most_dominant

        # Clear graph and rebuild as star network
        G = nx.DiGraph()

        # Add all significant variables
        for clean_var, coef in significant_vars:
            category, color = get_node_properties(clean_var)
            G.add_node(clean_var,
                      category=category,
                      color=color,
                      coefficient=coef)

        # Add edges from dominant variable to others
        for clean_var, coef in significant_vars[1:]:
            # Relationship type
            if np.sign(dom_coef) == np.sign(coef):
                edge_color = '#e74c3c'  # Red = amplifying (moves together)
                relationship = '+'
            else:
                edge_color = '#3498db'  # Blue = dampening (moves inversely)
                relationship = '−'

            edge_width = abs(coef) * 3

            G.add_edge(dom_var, clean_var,
                      weight=abs(coef),
                      color=edge_color,
                      coef=coef,
                      relationship=relationship,
                      width=edge_width)

    # Layout
    if G.number_of_nodes() > 0:
        # Use spring layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw edges with labels
        edges = G.edges()
        if edges:
            edge_colors = [G[u][v]['color'] for u, v in edges]
            edge_widths = [G[u][v]['width'] if 'width' in G[u][v] else 2 for u, v in edges]

            nx.draw_networkx_edges(G, pos,
                                  edge_color=edge_colors,
                                  width=edge_widths,
                                  alpha=0.7,
                                  arrowsize=30,
                                  arrowstyle='-|>',
                                  connectionstyle='arc3,rad=0.1',
                                  node_size=2000,
                                  ax=ax)

            # Edge labels with coefficients
            edge_labels = {}
            for u, v in edges:
                coef = G[u][v]['coef']
                rel = G[u][v]['relationship']
                edge_labels[(u, v)] = f"{coef:.2f}\n({rel})"

            nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                        font_size=9,
                                        font_weight='bold',
                                        bbox=dict(boxstyle='round,pad=0.3',
                                                facecolor='white',
                                                alpha=0.8),
                                        ax=ax)

        # Draw nodes
        for node in G.nodes():
            node_color = G.nodes[node]['color']
            node_coef = G.nodes[node]['coefficient']
            node_size = 2000 + abs(node_coef) * 2000

            nx.draw_networkx_nodes(G, pos,
                                  nodelist=[node],
                                  node_color=node_color,
                                  node_size=node_size,
                                  edgecolors='black',
                                  linewidths=3,
                                  ax=ax)

        # Node labels
        nx.draw_networkx_labels(G, pos,
                               font_size=10,
                               font_weight='bold',
                               ax=ax)

        # Title
        ax.set_title(f'Vector {vec_num}: {vec_name}\n' +
                    f'Dominant variable: {significant_vars[0][0]} (β = {significant_vars[0][1]:.3f})\n' +
                    f'Red = Amplifying (positive) | Blue = Dampening (inverse)',
                    fontsize=14, fontweight='bold', pad=15)
    else:
        ax.text(0.5, 0.5, f'Vector {vec_num}\nNo significant relationships',
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title(f'Vector {vec_num}: {vec_name}', fontsize=14, fontweight='bold')

    ax.axis('off')

    print(f"    Significant variables: {len(significant_vars)}")
    print(f"    Edges: {G.number_of_edges()}")

# Overall title
fig.suptitle('Cointegrating Vector Networks - VECM Lag 2\n' +
            'DoD Bureaucratic Equilibrium Relationships (1987-2024)\n' +
            'Node size = |coefficient| | Red = Amplifying | Blue = Dampening',
            fontsize=18, fontweight='bold', y=0.98)

# Legend
legend_elements = [
    Patch(facecolor='#3498db', label='Enlisted Personnel', edgecolor='black', linewidth=2),
    Patch(facecolor='#e74c3c', label='Officer Personnel', edgecolor='black', linewidth=2),
    Patch(facecolor='#2ecc71', label='Bureaucratic Measures', edgecolor='black', linewidth=2),
    plt.Line2D([0], [0], color='#e74c3c', linewidth=4, label='Amplifying (+) - moves together'),
    plt.Line2D([0], [0], color='#3498db', linewidth=4, label='Dampening (−) - moves inversely'),
]

fig.legend(handles=legend_elements,
          loc='upper right',
          fontsize=12,
          framealpha=0.95,
          title='Network Key',
          title_fontsize=13,
          bbox_to_anchor=(0.98, 0.96))

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = 'data/analysis/VECM_LAG2/cointegrating_vector_networks.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n  [OK] Combined network diagram saved: {output_path}")

# =============================================================================
# CREATE INDIVIDUAL NETWORKS (ONE PER VECTOR)
# =============================================================================
print("\n[3/3] Creating individual high-resolution networks...")

for vec_idx in range(len(beta_df.columns)):
    vec_name = beta_df.columns[vec_idx]
    vec_num = vec_idx + 1

    # Get coefficients
    coeffs = beta_df[vec_name]

    # Create graph
    G = nx.DiGraph()

    # Get significant variables
    significant_vars = []
    for orig_var, clean_var in var_mapping.items():
        coef = coeffs[orig_var]
        if abs(coef) > 0.1:
            category, color = get_node_properties(clean_var)
            G.add_node(clean_var, category=category, color=color, coefficient=coef)
            significant_vars.append((clean_var, coef))

    significant_vars.sort(key=lambda x: abs(x[1]), reverse=True)

    if len(significant_vars) > 0:
        # Star network from most dominant
        dom_var, dom_coef = significant_vars[0]

        for clean_var, coef in significant_vars[1:]:
            if np.sign(dom_coef) == np.sign(coef):
                edge_color = '#e74c3c'  # Red = amplifying
                relationship = '+'
            else:
                edge_color = '#3498db'  # Blue = dampening
                relationship = '−'

            edge_width = abs(coef) * 4

            G.add_edge(dom_var, clean_var,
                      color=edge_color,
                      coef=coef,
                      relationship=relationship,
                      width=edge_width)

        # Create individual figure
        fig_single = plt.figure(figsize=(16, 14), facecolor='white')
        ax_single = fig_single.add_subplot(111)

        # Layout
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

        # Draw edges
        edges = G.edges()
        if edges:
            edge_colors = [G[u][v]['color'] for u, v in edges]
            edge_widths = [G[u][v]['width'] for u, v in edges]

            nx.draw_networkx_edges(G, pos,
                                  edge_color=edge_colors,
                                  width=edge_widths,
                                  alpha=0.7,
                                  arrowsize=40,
                                  arrowstyle='-|>',
                                  connectionstyle='arc3,rad=0.1',
                                  node_size=3000,
                                  ax=ax_single)

            # Edge labels
            edge_labels = {}
            for u, v in edges:
                coef = G[u][v]['coef']
                edge_labels[(u, v)] = f"{coef:.3f}"

            nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                        font_size=11,
                                        font_weight='bold',
                                        bbox=dict(boxstyle='round,pad=0.4',
                                                facecolor='white',
                                                alpha=0.9,
                                                edgecolor='black',
                                                linewidth=2),
                                        ax=ax_single)

        # Draw nodes
        for node in G.nodes():
            node_color = G.nodes[node]['color']
            node_coef = G.nodes[node]['coefficient']
            node_size = 3000 + abs(node_coef) * 3000

            nx.draw_networkx_nodes(G, pos,
                                  nodelist=[node],
                                  node_color=node_color,
                                  node_size=node_size,
                                  edgecolors='black',
                                  linewidths=4,
                                  ax=ax_single)

        # Node labels
        nx.draw_networkx_labels(G, pos,
                               font_size=13,
                               font_weight='bold',
                               ax=ax_single)

        # Title with interpretation
        interpretation = ""
        if vec_num == 1:
            interpretation = "Teeth-to-Tail Shift: Junior Enlisted baseline with Field Grade Officers growth"
        elif vec_num == 2:
            interpretation = "Bureaucratic Delay: FOIA processing tied to policy complexity"
        elif vec_num == 3:
            interpretation = "Political Appointee Trade-off: PAS vs Field Grade Officers"
        elif vec_num == 4:
            interpretation = "Civilianization: Civilian workforce substitution pattern"

        ax_single.set_title(f'Cointegrating Vector {vec_num}\n' +
                           f'{interpretation}\n\n' +
                           f'Dominant Variable: {dom_var} (β = {dom_coef:.3f})\n' +
                           f'Red edges = Amplifying (moves together) | Blue edges = Dampening (moves inversely)\n' +
                           f'Node size = |coefficient| | Edge thickness = |coefficient|',
                           fontsize=16, fontweight='bold', pad=25)

        ax_single.legend(handles=legend_elements,
                        loc='upper left',
                        fontsize=12,
                        framealpha=0.95,
                        title='Network Key',
                        title_fontsize=13)

        ax_single.axis('off')
        plt.tight_layout()

        output_single = f'data/analysis/VECM_LAG2/cointegrating_vector_{vec_num}_network.png'
        plt.savefig(output_single, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] Vector {vec_num} network saved: {output_single}")

        plt.close(fig_single)

print("\n" + "=" * 100)
print("COINTEGRATING VECTOR NETWORKS COMPLETE")
print("=" * 100)
print("\nFiles generated:")
print("  1. cointegrating_vector_networks.png (all 4 vectors in one figure)")
print("  2-5. cointegrating_vector_[1-4]_network.png (individual high-res)")
print("\nEach network shows:")
print("  - Equilibrium relationship from beta coefficients")
print("  - Star pattern from most dominant variable")
print("  - Coefficient values labeled on edges")
print("  - Green = positive relationship, Red = inverse relationship")
print("  - Node/edge size proportional to coefficient magnitude")
print("=" * 100)
