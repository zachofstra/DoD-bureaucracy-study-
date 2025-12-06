"""
Enhanced Network Visualization for FINAL_TOP9_WITH_EXOGENOUS
Creates a cleaner, more interpretable causal network diagram
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Read the Granger causality data
df = pd.read_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/granger_significant.xlsx')
centrality = pd.read_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/network_centrality.xlsx')

# Filter for significance at 5% level
df_sig = df[df['Significant_5pct'] == True].copy()

print(f"Total significant relationships (5% level): {len(df_sig)}")

# Create directed graph
G = nx.DiGraph()

# Add all nodes first
all_vars = list(set(df_sig['Cause'].tolist() + df_sig['Effect'].tolist()))
G.add_nodes_from(all_vars)

# Add edges with weights (F-statistic)
for _, row in df_sig.iterrows():
    G.add_edge(row['Cause'], row['Effect'],
               weight=row['F_statistic'],
               pvalue=row['p_value'])

# Define variable categories for coloring
categories = {
    'Exogenous': ['GDP_Growth', 'Major_Conflict'],
    'Administrative': ['Policy_Count', 'Total_PAS', 'Total_Civilians'],
    'Military_Officers': ['O4_MajorLTCDR_Pct', 'O5_LtColCDR_Pct', 'O6_ColCAPT_Pct'],
    'Military_Enlisted': ['E5_Pct']
}

# Color scheme
colors = {
    'Exogenous': '#e74c3c',           # Red
    'Administrative': '#3498db',       # Blue
    'Military_Officers': '#2ecc71',    # Green
    'Military_Enlisted': '#f39c12'     # Orange
}

# Create node to category mapping
node_to_category = {}
for category, nodes in categories.items():
    for node in nodes:
        node_to_category[node] = category

# Create hierarchical layout
pos = {}

# Layer 1: Exogenous (top)
exog_vars = categories['Exogenous']
for i, var in enumerate(exog_vars):
    if var in G.nodes():
        pos[var] = (i * 3 - 1.5, 3)

# Layer 2: Administrative (middle-top)
admin_vars = categories['Administrative']
for i, var in enumerate(admin_vars):
    if var in G.nodes():
        pos[var] = (i * 2.5 - 2.5, 2)

# Layer 3: Military Officers (middle-bottom)
officer_vars = categories['Military_Officers']
for i, var in enumerate(officer_vars):
    if var in G.nodes():
        pos[var] = (i * 2.5 - 2.5, 1)

# Layer 4: Military Enlisted (bottom)
enlisted_vars = categories['Military_Enlisted']
for i, var in enumerate(enlisted_vars):
    if var in G.nodes():
        pos[var] = (0, 0)

# Create figure
fig, ax = plt.subplots(figsize=(20, 14), facecolor='white')

# Draw edges with varying thickness based on F-statistic
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
max_weight = max(weights)

# Normalize edge widths
edge_widths = [2 + (w / max_weight) * 6 for w in weights]

# Draw edges with curved arrows
for (u, v), width in zip(edges, edge_widths):
    # Get positions
    x1, y1 = pos[u]
    x2, y2 = pos[v]

    # Determine if feedback loop
    is_feedback = G.has_edge(v, u)

    # Calculate control point for curve
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Offset for curved edge (more curve for feedback loops)
    offset = 0.3 if is_feedback else 0.15
    dx = -(y2 - y1)
    dy = x2 - x1
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        dx /= length
        dy /= length

    control_x = mid_x + dx * offset
    control_y = mid_y + dy * offset

    # Create curved arrow
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>',
        mutation_scale=25,
        linewidth=width/2,
        color='gray',
        alpha=0.6,
        connectionstyle=f"arc3,rad={offset}",
        zorder=1
    )
    ax.add_patch(arrow)

# Draw nodes
for node in G.nodes():
    category = node_to_category[node]
    color = colors[category]

    # Node size based on total degree
    degree = G.degree(node)
    node_size = 3000 + degree * 800

    x, y = pos[node]

    # Draw node circle
    circle = plt.Circle((x, y), 0.35, color=color, alpha=0.8, zorder=2,
                       edgecolor='white', linewidth=3)
    ax.add_patch(circle)

    # Add node label with better formatting
    label = node.replace('_', '\n')
    ax.text(x, y, label, fontsize=10, fontweight='bold',
           ha='center', va='center', color='white', zorder=3)

# Add category labels on the left
ax.text(-5.5, 3, 'EXOGENOUS\nFACTORS', fontsize=14, fontweight='bold',
       color=colors['Exogenous'], ha='right', va='center',
       bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors['Exogenous'], linewidth=2))

ax.text(-5.5, 2, 'ADMINISTRATIVE\nBUREAUCRACY', fontsize=14, fontweight='bold',
       color=colors['Administrative'], ha='right', va='center',
       bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors['Administrative'], linewidth=2))

ax.text(-5.5, 1, 'MILITARY\nOFFICERS', fontsize=14, fontweight='bold',
       color=colors['Military_Officers'], ha='right', va='center',
       bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors['Military_Officers'], linewidth=2))

ax.text(-5.5, 0, 'MILITARY\nENLISTED', fontsize=14, fontweight='bold',
       color=colors['Military_Enlisted'], ha='right', va='center',
       bbox=dict(boxstyle='round', facecolor='white', edgecolor=colors['Military_Enlisted'], linewidth=2))

# Add title
ax.text(0, 4.2, 'DoD Bureaucratic Growth Causal Network (1987-2024)',
       fontsize=20, fontweight='bold', ha='center')
ax.text(0, 3.85, '9-Variable VAR Model with Exogenous Controls (GDP + Conflict)',
       fontsize=14, ha='center', style='italic', color='gray')

# Add legend for edge thickness
legend_elements = [
    mpatches.Patch(facecolor='gray', alpha=0.6, label=f'{len(df_sig)} Significant Causal Relationships (p<0.05)'),
    plt.Line2D([0], [0], color='gray', linewidth=4, label='Strong Effect (High F-stat)', alpha=0.6),
    plt.Line2D([0], [0], color='gray', linewidth=2, label='Moderate Effect (Low F-stat)', alpha=0.6),
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
         frameon=True, fancybox=True, shadow=True)

# Add key findings box
findings_text = """KEY FINDINGS:
• Total_Civilians: Most connected (8 edges)
• E5_Pct → Officers: Strong causal driver
• Total_PAS → Total_Civilians: Strongest relationship (F=20.1)
• 6 Feedback loops detected
"""

ax.text(-5.5, -1.2, findings_text, fontsize=11,
       bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2),
       verticalalignment='top', family='monospace')

# Add network statistics box
stats_text = f"""NETWORK STATISTICS:
Nodes: {G.number_of_nodes()}
Edges: {G.number_of_edges()}
Density: {nx.density(G):.3f}
Avg Path Length: {nx.average_shortest_path_length(G.to_undirected()):.2f}
"""

ax.text(5.5, -1.2, stats_text, fontsize=11,
       bbox=dict(boxstyle='round', facecolor='lightcyan', edgecolor='blue', linewidth=2),
       verticalalignment='top', family='monospace')

# Set axis limits and remove axes
ax.set_xlim(-6, 6)
ax.set_ylim(-2, 4.5)
ax.axis('off')

plt.tight_layout()
plt.savefig('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/network_graph_enhanced.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("\n[OK] Enhanced network diagram saved: network_graph_enhanced.png")

# Create a second visualization: Focused on O4 drivers
fig2, ax2 = plt.subplots(figsize=(16, 12), facecolor='white')

# Get FEVD data for O4
fevd = pd.read_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/fevd_data.xlsx')

# Filter for step 10 (long-run effects)
o4_fevd = fevd[fevd['Step'] == 10]

# Get O4 columns
o4_cols = [col for col in fevd.columns if col.startswith('O4_MajorLTCDR_Pct_from_')]

# Extract variance decomposition
if len(o4_fevd) > 0 and len(o4_cols) > 0:
    # Get variance values and clean up column names
    variances = {}
    for col in o4_cols:
        source_var = col.replace('O4_MajorLTCDR_Pct_from_', '')
        variances[source_var] = o4_fevd[col].iloc[0] * 100  # Convert to percentage

    variances = pd.Series(variances).sort_values(ascending=False)

    # Create bar chart
    colors_fevd = [colors.get(node_to_category.get(var, 'Administrative'), 'gray')
                   for var in variances.index]

    bars = ax2.barh(range(len(variances)), variances.values, color=colors_fevd,
                    alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_yticks(range(len(variances)))
    ax2.set_yticklabels([v.replace('_', ' ') for v in variances.index], fontsize=12)
    ax2.set_xlabel('Variance Explained (%)', fontsize=14, fontweight='bold')
    ax2.set_title('What Drives O4 (Major/LTCDR) Bureaucratic Bloat?\nForecast Error Variance Decomposition (10 steps ahead)',
                 fontsize=16, fontweight='bold', pad=20)

    # Add percentage labels on bars
    for i, (bar, val) in enumerate(zip(bars, variances.values)):
        ax2.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')

    # Add gridlines
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add category indicators
    exog_total = sum([v for k, v in variances.items() if node_to_category.get(k) == 'Exogenous'])
    admin_total = sum([v for k, v in variances.items() if node_to_category.get(k) == 'Administrative'])

    summary_text = f"""VARIANCE BREAKDOWN:
Endogenous Bureaucratic: {admin_total:.1f}%
Exogenous (GDP+Conflict): {exog_total:.1f}%
Self-perpetuation: {variances.get('O4_MajorLTCDR_Pct', 0):.1f}%

TOP 3 DRIVERS:
1. {variances.index[0].replace('_', ' ')}: {variances.values[0]:.1f}%
2. {variances.index[1].replace('_', ' ')}: {variances.values[1]:.1f}%
3. {variances.index[2].replace('_', ' ')}: {variances.values[2]:.1f}%
"""

    ax2.text(0.98, 0.97, summary_text, transform=ax2.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                     edgecolor='orange', linewidth=2),
            family='monospace')

    plt.tight_layout()
    plt.savefig('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/o4_drivers_chart.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    print("[OK] O4 drivers chart saved: o4_drivers_chart.png")

plt.show()

print("\n[OK] All enhanced visualizations created successfully!")
