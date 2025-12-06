"""
Create Signed Network - Simplified Direct Extraction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')

output_dir = 'data/analysis/VECM_7VARS'

print("Creating signed network...")

# Load and prep data (same as VAR(2) analysis)
df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

endog_vars = ['Junior_Enlisted_Z', 'FOIA_Simple_Days_Z', 'Total_PAS_Z',
              'Total_Civilians_Z', 'Policy_Count_Log', 'Field_Grade_Officers_Z', 'GOFOs_Z']
exog_vars = ['GDP_Growth', 'Major_Conflict']

data = df[endog_vars + exog_vars].copy().dropna()
endog_data = data[endog_vars]
exog_data = data[exog_vars]

# Apply differencing
non_stationary = ['FOIA_Simple_Days_Z', 'Total_Civilians_Z', 'Policy_Count_Log',
                  'Field_Grade_Officers_Z', 'GOFOs_Z']
endog_diff = endog_data.copy()
for var in non_stationary:
    endog_diff[var] = endog_diff[var].diff()
endog_diff = endog_diff.dropna()
exog_aligned = exog_data.loc[endog_diff.index]

# Fit VAR(2)
var_model = VAR(endog_diff, exog=exog_aligned)
var_result = var_model.fit(maxlags=2, ic=None)

# Read Granger results
granger_df = pd.read_excel('data/analysis/FINAL_VAR2_WITH_CIVILIANS/granger_by_lag_significant.xlsx')

# Extract coefficients
relationships = []
for _, row in granger_df.iterrows():
    cause = row['Cause']
    effect = row['Effect']
    lag = int(row['Lag'])

    # Build parameter name: "L{lag}.{cause}"
    param_name = f"L{lag}.{cause}"

    # Get coefficient from params[param_name, effect]
    if param_name in var_result.params.index and effect in var_result.params.columns:
        coef = var_result.params.loc[param_name, effect]
        sign = 'Positive' if coef > 0 else 'Negative'

        relationships.append({
            'Cause': cause,
            'Effect': effect,
            'Lag': lag,
            'Coefficient': coef,
            'Sign': sign,
            'F_stat': row['F_stat'],
            'p_value': row['p_value']
        })
        print(f"{cause} -> {effect} (lag{lag}): {sign} ({coef:.3f})")

rel_df = pd.DataFrame(relationships)
print(f"\nExtracted {len(rel_df)} relationships")

rel_df.to_excel(f'{output_dir}/signed_relationships_FINAL.xlsx', index=False)

# Build network
G = nx.DiGraph()
for _, row in rel_df.iterrows():
    G.add_edge(row['Cause'], row['Effect'],
               weight=row['F_stat'], coefficient=row['Coefficient'], sign=row['Sign'])

for var in endog_vars:
    if var not in G:
        G.add_node(var)

# Calculate centrality
degree = dict(G.degree(weight='weight'))

# Plot
fig, ax = plt.subplots(1, 1, figsize=(20, 16), facecolor='white')
pos = nx.spring_layout(G, k=3, iterations=100, seed=42, weight='weight')

# Node colors
node_colors = []
for node in G.nodes():
    if 'Enlisted' in node:
        node_colors.append('#3498db')
    elif 'Officers' in node or 'GOFO' in node:
        node_colors.append('#e74c3c')
    else:
        node_colors.append('#2ecc71')

# Node sizes
node_sizes = [1200 + degree.get(node, 0) * 180 for node in G.nodes()]

# Edge colors and widths
edge_colors = ['#e74c3c' if G[u][v]['sign'] == 'Positive' else '#3498db' for u, v in G.edges()]
edge_widths = [2.5 + (G[u][v]['weight'] / 8) * 6 for u, v in G.edges()]

# Draw
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.8, width=edge_widths,
                       arrows=True, arrowsize=30, ax=ax, connectionstyle='arc3,rad=0.2')
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       alpha=0.95, edgecolors='black', linewidths=3, ax=ax)

labels = {n: n.replace('_Z', '').replace('_', '\n') for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=13, font_weight='bold', ax=ax)

# Edge labels
edge_labels = {(u, v): f"{'+' if G[u][v]['coefficient']>0 else ''}{G[u][v]['coefficient']:.2f}"
               for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10,
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95),
                             ax=ax)

ax.set_title('DoD Bureaucratic Growth: Signed Causal Network (1987-2024)\n' +
            f'{len(rel_df)} Significant Granger Relationships\n' +
            'RED = Positive (+) | BLUE = Negative (-) | Width = F-statistic strength',
            fontsize=20, fontweight='bold', pad=30)
ax.axis('off')

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend = [
    Patch(facecolor='#3498db', label='Enlisted', alpha=0.95),
    Patch(facecolor='#e74c3c', label='Officers', alpha=0.95),
    Patch(facecolor='#2ecc71', label='Bureaucratic Measures', alpha=0.95),
    Line2D([0], [0], color='#e74c3c', lw=5, label='Positive (+)', alpha=0.8),
    Line2D([0], [0], color='#3498db', lw=5, label='Negative (-)', alpha=0.8),
]
ax.legend(handles=legend, loc='upper right', fontsize=15, framealpha=0.98)

plt.tight_layout()
plt.savefig(f'{output_dir}/SIGNED_NETWORK_FINAL.png', dpi=300, bbox_inches='tight')

print(f"\nSaved: {output_dir}/SIGNED_NETWORK_FINAL.png")
print(f"Relationships: {output_dir}/signed_relationships_FINAL.xlsx")
