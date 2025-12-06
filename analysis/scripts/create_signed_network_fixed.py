"""
Create Signed Network Diagram - FIXED VERSION
Properly extracts coefficient signs from VAR model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')

output_dir = 'data/analysis/VECM_7VARS'

print("=" * 100)
print("CREATING SIGNED NETWORK DIAGRAM (FIXED)")
print("=" * 100)

# Load data
df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

endog_vars = [
    'Junior_Enlisted_Z',
    'FOIA_Simple_Days_Z',
    'Total_PAS_Z',
    'Total_Civilians_Z',
    'Policy_Count_Log',
    'Field_Grade_Officers_Z',
    'GOFOs_Z'
]

exog_vars = ['GDP_Growth', 'Major_Conflict']

data = df[endog_vars + exog_vars].copy().dropna()
endog_data = data[endog_vars]
exog_data = data[exog_vars]

# Apply differencing (same as VAR(2) analysis)
non_stationary = ['FOIA_Simple_Days_Z', 'Total_Civilians_Z', 'Policy_Count_Log',
                  'Field_Grade_Officers_Z', 'GOFOs_Z']

endog_diff = endog_data.copy()
for var in non_stationary:
    endog_diff[var] = endog_diff[var].diff()

endog_diff = endog_diff.dropna()
exog_aligned = exog_data.loc[endog_diff.index]

# Fit VAR(2)
print("\nFitting VAR(2) to extract coefficient signs...")
var_model = VAR(endog_diff, exog=exog_aligned)
var_result = var_model.fit(maxlags=2, ic=None)

print("  [OK] VAR(2) fitted")

# Read significant Granger relationships
print("\nReading Granger causality results...")
try:
    granger_df = pd.read_excel('data/analysis/FINAL_VAR2_WITH_CIVILIANS/granger_by_lag_significant.xlsx')
    print(f"  Found {len(granger_df)} significant Granger relationships")
except:
    print("  ERROR: Could not find Granger results")
    exit(1)

# Extract coefficients for each significant relationship
print("\nExtracting coefficient signs...")
relationships = []

for _, row in granger_df.iterrows():
    cause = row['Cause']
    effect = row['Effect']
    lag = int(row['Lag'])
    f_stat = row['F_stat']
    p_val = row['p_value']

    # Get coefficient from VAR params
    param_name = f'{cause}.L{lag}'

    try:
        if param_name in var_result.params.index and effect in var_result.params.columns:
            coef = var_result.params.loc[param_name, effect]
            sign = 'Positive' if coef > 0 else 'Negative'

            relationships.append({
                'Cause': cause,
                'Effect': effect,
                'Lag': lag,
                'Coefficient': coef,
                'Sign': sign,
                'F_stat': f_stat,
                'p_value': p_val
            })

            print(f"  {cause} -> {effect} (lag {lag}): coef={coef:.3f}, sign={sign}")
    except Exception as e:
        print(f"  WARNING: Could not extract {cause} -> {effect}: {e}")

rel_df = pd.DataFrame(relationships)

if len(rel_df) == 0:
    print("\n  ERROR: No relationships extracted!")
    print("\n  Available params:")
    print(var_result.params.index.tolist()[:20])
    print("\n  Trying alternative extraction...")

    # Try extracting ALL significant relationships from Granger tests
    for _, row in granger_df.iterrows():
        cause = row['Cause']
        effect = row['Effect']
        lag = 2  # Focus on lag 2

        param_name = f'{cause}.L{lag}'

        # Get all column names
        available_cols = var_result.params.columns.tolist()

        if param_name in var_result.params.index:
            # Find matching effect column
            matching_effects = [col for col in available_cols if effect.replace('_Z', '') in col.replace('_Z', '')]

            if len(matching_effects) > 0:
                effect_col = matching_effects[0]
                coef = var_result.params.loc[param_name, effect_col]
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

    rel_df = pd.DataFrame(relationships)

print(f"\nTotal relationships with signs: {len(rel_df)}")

if len(rel_df) == 0:
    print("\nERROR: Still no relationships. Aborting.")
    exit(1)

rel_df.to_excel(f'{output_dir}/signed_relationships_fixed.xlsx', index=False)

# Build network
print("\nBuilding network graph...")
G = nx.DiGraph()

for _, row in rel_df.iterrows():
    G.add_edge(row['Cause'], row['Effect'],
               weight=row['F_stat'],
               coefficient=row['Coefficient'],
               sign=row['Sign'],
               lag=row['Lag'])

# Add isolated nodes
for var in endog_vars:
    if var not in G:
        G.add_node(var)

# Calculate centrality
in_degree = dict(G.in_degree(weight='weight'))
out_degree = dict(G.out_degree(weight='weight'))
total_degree = {n: in_degree.get(n, 0) + out_degree.get(n, 0) for n in G.nodes()}

print(f"  Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(20, 16), facecolor='white')

# Better layout
if G.number_of_edges() > 0:
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42, weight='weight')
else:
    pos = nx.circular_layout(G)

# Node colors by category
node_colors = []
for node in G.nodes():
    if 'Enlisted' in node:
        node_colors.append('#3498db')  # Blue
    elif 'Officers' in node or 'GOFO' in node:
        node_colors.append('#e74c3c')  # Red
    elif 'Civilian' in node or 'PAS' in node or 'FOIA' in node or 'Policy' in node:
        node_colors.append('#2ecc71')  # Green
    else:
        node_colors.append('#95a5a6')  # Gray

# Node sizes by centrality (bigger if connected)
node_sizes = [1000 + total_degree.get(node, 0) * 200 for node in G.nodes()]

# Draw edges with colors
if G.number_of_edges() > 0:
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if G[u][v]['sign'] == 'Positive':
            edge_colors.append('#e74c3c')  # Red for positive
        else:
            edge_colors.append('#3498db')  # Blue for negative

        # Width by F-statistic
        edge_widths.append(2 + (G[u][v]['weight'] / 8) * 6)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.8,
                           width=edge_widths, arrows=True, arrowsize=30,
                           ax=ax, connectionstyle='arc3,rad=0.2', node_size=node_sizes)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                       node_size=node_sizes, alpha=0.95,
                       edgecolors='black', linewidths=3, ax=ax)

# Labels
labels = {node: node.replace('_Z', '').replace('_', '\n') for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=13, font_weight='bold', ax=ax)

# Edge labels with coefficients
if G.number_of_edges() > 0 and G.number_of_edges() < 20:  # Only if not too many
    edge_labels = {}
    for u, v in G.edges():
        coef = G[u][v]['coefficient']
        sign_str = '+' if coef > 0 else ''
        edge_labels[(u, v)] = f"{sign_str}{coef:.2f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10,
                                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='black'),
                                 ax=ax)

ax.set_title('DoD Bureaucratic Growth: Signed Causal Network (1987-2024)\n' +
            f'Based on {len(rel_df)} Significant Granger Relationships\n' +
            'RED edges = POSITIVE (+) influence | BLUE edges = NEGATIVE (-) influence\n' +
            'Edge width = Statistical strength (F-statistic)',
            fontsize=20, fontweight='bold', pad=35)
ax.axis('off')

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#3498db', label='Enlisted Personnel', alpha=0.95),
    Patch(facecolor='#e74c3c', label='Officer Personnel', alpha=0.95),
    Patch(facecolor='#2ecc71', label='Bureaucratic Measures', alpha=0.95),
    Line2D([0], [0], color='#e74c3c', linewidth=5, label='Positive Influence (+)', alpha=0.8),
    Line2D([0], [0], color='#3498db', linewidth=5, label='Negative Influence (-)', alpha=0.8),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=15,
         framealpha=0.98, title='Legend', title_fontsize=16,
         edgecolor='black', fancybox=True)

plt.tight_layout()
plt.savefig(f'{output_dir}/SIGNED_NETWORK_FINAL.png', dpi=300, bbox_inches='tight')

print("\n" + "=" * 100)
print(f"SIGNED NETWORK DIAGRAM CREATED: {G.number_of_edges()} relationships visualized")
print("=" * 100)
print(f"\nSaved to: {output_dir}/SIGNED_NETWORK_FINAL.png")
print(f"Relationships table: {output_dir}/signed_relationships_fixed.xlsx")
print("=" * 100)
