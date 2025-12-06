"""
Network Diagram for VECM Lag 2 - Error Correction Relationships
Color-coded visualization of how variables adjust to equilibrium deviations

Network Structure:
- Nodes = Variables (color-coded by category)
- Edges = Strong error correction relationships (|alpha| > 0.3)
- Edge color = Direction (blue = negative, red = positive)
- Edge thickness = Strength of adjustment
- Node size = Overall endogeneity (max|alpha| across all ECTs)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("CREATING VECM NETWORK DIAGRAM - ERROR CORRECTION RELATIONSHIPS")
print("=" * 100)

# =============================================================================
# LOAD ALPHA MATRIX
# =============================================================================
print("\n[1/3] Loading error correction coefficients...")

alpha_df = pd.read_excel('data/analysis/VECM_LAG2/alpha_error_correction_coefficients.xlsx', index_col=0)

print(f"  Variables: {len(alpha_df)}")
print(f"  Error correction terms: {len(alpha_df.columns)}")

# =============================================================================
# CREATE NETWORK FROM ALPHA MATRIX
# =============================================================================
print("\n[2/3] Building network structure...")

# Create directed graph
G = nx.DiGraph()

# Add nodes for each variable
variables = alpha_df.index.tolist()
ects = alpha_df.columns.tolist()

# Add variable nodes with categories
for var in variables:
    if 'Enlisted' in var:
        category = 'Enlisted Personnel'
        color = '#3498db'  # Blue
    elif 'Officers' in var or 'GOFO' in var:
        category = 'Officer Personnel'
        color = '#e74c3c'  # Red
    elif 'Civilian' in var or 'PAS' in var or 'FOIA' in var or 'Policy' in var:
        category = 'Bureaucratic Measures'
        color = '#2ecc71'  # Green
    else:
        category = 'Other'
        color = '#95a5a6'  # Gray

    # Node size based on max|alpha|
    max_alpha = alpha_df.loc[var].abs().max()

    G.add_node(var,
               category=category,
               color=color,
               endogeneity=max_alpha)

# Add edges for strong error correction relationships
# Edge from ECT to Variable if |alpha| > threshold
threshold = 0.3

edges_added = 0
for var in variables:
    for ect in ects:
        alpha_val = alpha_df.loc[var, ect]

        if abs(alpha_val) > threshold:
            # Create edge from ECT to variable
            # Edge properties
            edge_color = '#3498db' if alpha_val < 0 else '#e74c3c'  # Blue=negative, Red=positive
            edge_width = abs(alpha_val) * 2

            G.add_edge(ect, var,
                      alpha=alpha_val,
                      color=edge_color,
                      width=edge_width,
                      weight=abs(alpha_val))
            edges_added += 1

print(f"  Nodes (variables): {len([n for n in G.nodes() if n in variables])}")
print(f"  Nodes (ECTs): {len([n for n in G.nodes() if n not in variables])}")
print(f"  Edges (strong error corrections): {edges_added}")

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================
print("\n[3/3] Creating network visualization...")

fig, ax = plt.subplots(1, 1, figsize=(20, 16), facecolor='white')

# Layout - separate ECTs and variables
pos = {}

# Position ECTs on left
n_ects = len([n for n in G.nodes() if n not in variables])
for i, ect in enumerate(ects):
    if ect in G.nodes():
        pos[ect] = (-2, (i - n_ects/2) * 2)

# Position variables on right in categories
enlisted_vars = [v for v in variables if 'Enlisted' in v]
officer_vars = [v for v in variables if 'Officers' in v or 'GOFO' in v]
bureau_vars = [v for v in variables if 'Civilian' in v or 'PAS' in v or 'FOIA' in v or 'Policy' in v]

y_offset = 0
for i, var in enumerate(enlisted_vars):
    pos[var] = (2, y_offset + i * 1.5)
y_offset += len(enlisted_vars) * 1.5 + 1

for i, var in enumerate(officer_vars):
    pos[var] = (2, y_offset + i * 1.5)
y_offset += len(officer_vars) * 1.5 + 1

for i, var in enumerate(bureau_vars):
    pos[var] = (2, y_offset + i * 1.5)

# Draw edges with colors and widths
edges = G.edges()
edge_colors = [G[u][v]['color'] for u, v in edges]
edge_widths = [G[u][v]['width'] for u, v in edges]

nx.draw_networkx_edges(G, pos,
                       edge_color=edge_colors,
                       width=edge_widths,
                       alpha=0.6,
                       arrowsize=20,
                       arrowstyle='->',
                       connectionstyle='arc3,rad=0.1',
                       ax=ax)

# Draw ECT nodes (squares)
ect_nodes = [n for n in G.nodes() if n not in variables]
if ect_nodes:
    nx.draw_networkx_nodes(G, pos,
                          nodelist=ect_nodes,
                          node_color='#f39c12',
                          node_size=3000,
                          node_shape='s',
                          edgecolors='black',
                          linewidths=3,
                          ax=ax)

# Draw variable nodes (circles) with category colors
for var in variables:
    if var in G.nodes():
        node_color = G.nodes[var]['color']
        node_size = 1500 + G.nodes[var]['endogeneity'] * 1500

        nx.draw_networkx_nodes(G, pos,
                              nodelist=[var],
                              node_color=node_color,
                              node_size=node_size,
                              edgecolors='black',
                              linewidths=3,
                              ax=ax)

# Draw labels
# ECT labels
ect_labels = {n: n for n in ect_nodes}
nx.draw_networkx_labels(G, pos,
                       labels=ect_labels,
                       font_size=11,
                       font_weight='bold',
                       font_color='white',
                       ax=ax)

# Variable labels
var_labels = {n: n.replace('_Z', '').replace('_', '\n') for n in variables if n in G.nodes()}
nx.draw_networkx_labels(G, pos,
                       labels=var_labels,
                       font_size=10,
                       font_weight='bold',
                       ax=ax)

# Title
ax.set_title('VECM Error Correction Network - Lag 2 Analysis\n' +
            'DoD Bureaucratic Growth: Self-Regulating Equilibrium Dynamics\n' +
            'Edge thickness = Adjustment strength | Blue = Negative correction | Red = Positive correction',
            fontsize=18, fontweight='bold', pad=30)

# Legend
legend_elements = [
    Patch(facecolor='#3498db', label='Enlisted Personnel', edgecolor='black', linewidth=2),
    Patch(facecolor='#e74c3c', label='Officer Personnel', edgecolor='black', linewidth=2),
    Patch(facecolor='#2ecc71', label='Bureaucratic Measures', edgecolor='black', linewidth=2),
    Patch(facecolor='#f39c12', label='Error Correction Terms (ECT)', edgecolor='black', linewidth=2),
    plt.Line2D([0], [0], color='#3498db', linewidth=4, label='Negative Correction (↓ when above equilibrium)'),
    plt.Line2D([0], [0], color='#e74c3c', linewidth=4, label='Positive Correction (↑ when above equilibrium)'),
]

ax.legend(handles=legend_elements,
         loc='upper left',
         fontsize=12,
         framealpha=0.95,
         title='Network Key',
         title_fontsize=14)

# Add interpretation box
interpretation = """
INTERPRETATION:
• ECTs (squares) represent the 4 equilibrium relationships
• Arrows show which variables adjust to which equilibria
• Thicker arrows = stronger adjustment (higher |alpha|)
• Blue arrows: Variable decreases when equilibrium exceeded
• Red arrows: Variable increases when equilibrium exceeded

KEY FINDING:
All 7 variables show strong error correction (no exogenous drivers).
The system is SELF-REGULATING - deviations from equilibrium
trigger coordinated adjustments across multiple dimensions.
"""

ax.text(0.02, 0.02, interpretation,
       transform=ax.transAxes,
       fontsize=10,
       verticalalignment='bottom',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
       family='monospace')

ax.axis('off')
plt.tight_layout()

# Save
output_path = 'data/analysis/VECM_LAG2/error_correction_network.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  [OK] Network diagram saved to: {output_path}")

# =============================================================================
# CREATE SUMMARY STATISTICS
# =============================================================================
print("\nNetwork Statistics:")
print("  " + "-" * 60)

# Most endogenous variables (largest max|alpha|)
endogeneity = {var: G.nodes[var]['endogeneity'] for var in variables if var in G.nodes()}
endogeneity_sorted = sorted(endogeneity.items(), key=lambda x: x[1], reverse=True)

print("\n  Most Endogenous Variables (strongest error correction):")
for var, endo in endogeneity_sorted[:5]:
    var_clean = var.replace('_Z', '').replace('_', ' ')
    print(f"    {var_clean:35s} Max|alpha| = {endo:.3f}")

# ECT with most connections
ect_connections = {}
for ect in ects:
    if ect in G.nodes():
        ect_connections[ect] = G.out_degree(ect)

print("\n  Error Correction Terms (ECT) - Number of Strong Adjustments:")
for ect in sorted(ect_connections.keys()):
    n_conn = ect_connections[ect]
    print(f"    {ect}: {n_conn} variables adjust strongly to this equilibrium")

# Average adjustment strength
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
if edge_weights:
    print(f"\n  Average adjustment strength: {np.mean(edge_weights):.3f}")
    print(f"  Strongest adjustment: {np.max(edge_weights):.3f}")
    print(f"  Total strong error corrections (|alpha| > 0.3): {len(edge_weights)}")

# =============================================================================
# ALTERNATIVE VISUALIZATION: CIRCULAR LAYOUT
# =============================================================================
print("\nCreating alternative circular network layout...")

fig2, ax2 = plt.subplots(1, 1, figsize=(18, 18), facecolor='white')

# Circular layout with variables on outer circle, ECTs in center
pos_circular = {}

# Place ECTs in center square
n_ects = len(ects)
for i, ect in enumerate(ects):
    if ect in G.nodes():
        angle = 2 * np.pi * i / n_ects
        radius = 1
        pos_circular[ect] = (radius * np.cos(angle), radius * np.sin(angle))

# Place variables on outer circle
n_vars = len(variables)
for i, var in enumerate(variables):
    if var in G.nodes():
        angle = 2 * np.pi * i / n_vars
        radius = 3
        pos_circular[var] = (radius * np.cos(angle), radius * np.sin(angle))

# Draw edges
nx.draw_networkx_edges(G, pos_circular,
                       edge_color=edge_colors,
                       width=edge_widths,
                       alpha=0.5,
                       arrowsize=15,
                       arrowstyle='->',
                       connectionstyle='arc3,rad=0.2',
                       ax=ax2)

# Draw ECT nodes
if ect_nodes:
    nx.draw_networkx_nodes(G, pos_circular,
                          nodelist=ect_nodes,
                          node_color='#f39c12',
                          node_size=4000,
                          node_shape='s',
                          edgecolors='black',
                          linewidths=3,
                          ax=ax2)

# Draw variable nodes
for var in variables:
    if var in G.nodes():
        node_color = G.nodes[var]['color']
        node_size = 2000 + G.nodes[var]['endogeneity'] * 2000

        nx.draw_networkx_nodes(G, pos_circular,
                              nodelist=[var],
                              node_color=node_color,
                              node_size=node_size,
                              edgecolors='black',
                              linewidths=3,
                              ax=ax2)

# Labels
nx.draw_networkx_labels(G, pos_circular,
                       labels=ect_labels,
                       font_size=12,
                       font_weight='bold',
                       font_color='white',
                       ax=ax2)

nx.draw_networkx_labels(G, pos_circular,
                       labels=var_labels,
                       font_size=11,
                       font_weight='bold',
                       ax=ax2)

ax2.set_title('VECM Error Correction Network - Circular Layout\n' +
             'DoD Bureaucratic Equilibrium Dynamics (1987-2024)\n' +
             'Inner squares = Equilibrium constraints | Outer circles = Adjusting variables',
             fontsize=18, fontweight='bold', pad=30)

ax2.legend(handles=legend_elements,
          loc='upper right',
          fontsize=11,
          framealpha=0.95,
          title='Network Key',
          title_fontsize=13)

ax2.axis('off')
ax2.set_xlim(-4.5, 4.5)
ax2.set_ylim(-4.5, 4.5)
plt.tight_layout()

output_path2 = 'data/analysis/VECM_LAG2/error_correction_network_circular.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  [OK] Circular network diagram saved to: {output_path2}")

print("\n" + "=" * 100)
print("NETWORK DIAGRAM COMPLETE")
print("=" * 100)
print("\nFiles generated:")
print("  1. error_correction_network.png (hierarchical layout)")
print("  2. error_correction_network_circular.png (circular layout)")
print("\nBoth diagrams show:")
print("  - Error correction structure from VECM lag 2")
print("  - 4 equilibrium constraints (ECT_1 to ECT_4)")
print("  - How each variable adjusts to each equilibrium")
print("  - Color-coded by variable category")
print("  - Edge thickness = adjustment strength")
print("=" * 100)
