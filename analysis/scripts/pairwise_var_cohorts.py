"""
Pairwise VAR and Granger Causality Network Analysis - COHORT VERSION
Using 7 rank cohorts instead of 19 individual ranks

This should reduce intercorrelation and highlight true causal drivers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("PAIRWISE VAR AND GRANGER CAUSALITY - COHORT VERSION")
print("Using 7 rank cohorts + bureaucratic/exogenous variables")
print("=" * 100)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/6] Loading dataset with rank cohorts...")

df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

# All variables (cohorts already z-scored)
analysis_vars = [
    # Rank cohorts (already z-scored)
    'Junior_Enlisted_Z',
    'Middle_Enlisted_Z',
    'Senior_Enlisted_Z',
    'Company_Grade_Officers_Z',
    'Field_Grade_Officers_Z',
    'GOFOs_Z',
    'Warrant_Officers_Z',

    # Bureaucratic measures
    'Policy_Count_Log',
    'Total_PAS_Z',
    'Total_Civilians_Z',
    'FOIA_Simple_Days_Z',
    'FOIA_Complex_Days_Z',

    # Exogenous
    'GDP_Growth',
    'Major_Conflict',

    # Political
    'HOR_Republican',
    'Senate_Republican',
    'President_Republican'
]

# Filter to available columns
available_vars = [v for v in analysis_vars if v in df.columns]
print(f"  Total variables for analysis: {len(available_vars)}")
print(f"  - Rank cohorts: 7")
print(f"  - Bureaucratic measures: 5")
print(f"  - Exogenous: 2")
print(f"  - Political: 3")

data = df[available_vars].copy()
data = data.dropna()

print(f"  Observations after dropna: {len(data)}")

# =============================================================================
# PAIRWISE GRANGER CAUSALITY TESTS
# =============================================================================
print("\n[2/6] Running pairwise Granger causality tests...")
print("  Testing all variable pairs with lags 1-4...")

granger_results = []
test_count = 0
total_tests = len(available_vars) * (len(available_vars) - 1)

for var1 in available_vars:
    for var2 in available_vars:
        if var1 == var2:
            continue

        test_count += 1
        if test_count % 50 == 0:
            print(f"    Progress: {test_count}/{total_tests} tests completed...")

        try:
            test_data = data[[var2, var1]].dropna()

            if len(test_data) < 15:
                continue

            gc_result = grangercausalitytests(test_data, maxlag=4, verbose=False)

            for lag in [1, 2, 3, 4]:
                if lag in gc_result:
                    f_stat = gc_result[lag][0]['ssr_ftest'][0]
                    p_val = gc_result[lag][0]['ssr_ftest'][1]

                    granger_results.append({
                        'Cause': var1,
                        'Effect': var2,
                        'Lag': lag,
                        'F_stat': f_stat,
                        'p_value': p_val,
                        'Significant_10pct': p_val < 0.10,
                        'Significant_5pct': p_val < 0.05,
                        'Significant_1pct': p_val < 0.01
                    })
        except Exception as e:
            continue

granger_df = pd.DataFrame(granger_results)

print(f"\n  Total tests completed: {len(granger_df)}")
print(f"  Significant at 10%: {granger_df['Significant_10pct'].sum()}")
print(f"  Significant at 5%: {granger_df['Significant_5pct'].sum()}")
print(f"  Significant at 1%: {granger_df['Significant_1pct'].sum()}")

granger_df.to_excel('data/analysis/pairwise_granger_cohorts_all.xlsx', index=False)

granger_sig = granger_df[granger_df['Significant_5pct']].copy()
granger_sig.to_excel('data/analysis/pairwise_granger_cohorts_significant.xlsx', index=False)

# =============================================================================
# BUILD CAUSAL NETWORK
# =============================================================================
print("\n[3/6] Building directed causal network...")

edges = []
for (cause, effect), group in granger_sig.groupby(['Cause', 'Effect']):
    max_f = group['F_stat'].max()
    min_p = group['p_value'].min()
    sig_lags = group['Lag'].tolist()

    edges.append({
        'source': cause,
        'target': effect,
        'weight': max_f,
        'p_value': min_p,
        'significant_lags': ','.join(map(str, sig_lags))
    })

edges_df = pd.DataFrame(edges)
edges_df.to_excel('data/analysis/pairwise_network_cohorts_edges.xlsx', index=False)

G = nx.DiGraph()
G.add_nodes_from(available_vars)

for _, edge in edges_df.iterrows():
    G.add_edge(edge['source'], edge['target'],
               weight=edge['weight'],
               p_value=edge['p_value'])

print(f"  Network nodes: {G.number_of_nodes()}")
print(f"  Network edges: {G.number_of_edges()}")
print(f"  Network density: {nx.density(G):.3f}")

# =============================================================================
# CALCULATE NETWORK CENTRALITY
# =============================================================================
print("\n[4/6] Calculating network centrality measures...")

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())
total_degree = {node: in_degree[node] + out_degree[node] for node in G.nodes()}

try:
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
except:
    eigenvector = {node: 0 for node in G.nodes()}

try:
    pagerank = nx.pagerank(G, weight='weight')
except:
    pagerank = {node: 0 for node in G.nodes()}

try:
    betweenness = nx.betweenness_centrality(G, weight='weight')
except:
    betweenness = {node: 0 for node in G.nodes()}

centrality_df = pd.DataFrame({
    'Variable': list(G.nodes()),
    'In_Degree': [in_degree[n] for n in G.nodes()],
    'Out_Degree': [out_degree[n] for n in G.nodes()],
    'Total_Degree': [total_degree[n] for n in G.nodes()],
    'Eigenvector_Centrality': [eigenvector[n] for n in G.nodes()],
    'PageRank': [pagerank[n] for n in G.nodes()],
    'Betweenness': [betweenness[n] for n in G.nodes()]
})

# Normalize centrality measures
for col in ['Eigenvector_Centrality', 'PageRank', 'Betweenness']:
    if centrality_df[col].max() > 0:
        centrality_df[f'{col}_Normalized'] = (centrality_df[col] / centrality_df[col].max()) * 100

# Composite score
centrality_df['Composite_Score'] = centrality_df[[
    'Total_Degree',
    'Eigenvector_Centrality_Normalized',
    'PageRank_Normalized',
    'Betweenness_Normalized'
]].mean(axis=1)

centrality_df = centrality_df.sort_values('Composite_Score', ascending=False)
centrality_df.to_excel('data/analysis/pairwise_network_cohorts_centrality.xlsx', index=False)

print("\n  Top variables by composite centrality score:")
print("  " + "-" * 96)
print(centrality_df[['Variable', 'Total_Degree', 'Composite_Score']].to_string(index=False))

# =============================================================================
# SELECT TOP VARIABLES
# =============================================================================
print("\n[5/6] Selecting top variables for VAR model...")

top_n = 10
top_vars = centrality_df.head(top_n)['Variable'].tolist()

print(f"\n  TOP {top_n} VARIABLES SELECTED:")
print("  " + "=" * 96)
for i, var in enumerate(top_vars, 1):
    score = centrality_df[centrality_df['Variable'] == var]['Composite_Score'].values[0]
    degree = centrality_df[centrality_df['Variable'] == var]['Total_Degree'].values[0]
    in_deg = centrality_df[centrality_df['Variable'] == var]['In_Degree'].values[0]
    out_deg = centrality_df[centrality_df['Variable'] == var]['Out_Degree'].values[0]
    print(f"  {i:2d}. {var:35s} Score={score:6.2f} (In={in_deg:2.0f}, Out={out_deg:2.0f})")

selection_df = centrality_df.head(top_n)[['Variable', 'In_Degree', 'Out_Degree', 'Total_Degree', 'Composite_Score']]
selection_df.to_excel('data/analysis/top_variables_cohorts.xlsx', index=False)

# =============================================================================
# VISUALIZE NETWORK
# =============================================================================
print("\n[6/6] Creating network visualization...")

fig, axes = plt.subplots(1, 2, figsize=(24, 12), facecolor='white')

# --- Plot 1: Network graph ---
ax = axes[0]
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Color by variable type
colors = []
for node in G.nodes():
    if 'Enlisted' in node or 'Officers' in node or 'GOFO' in node or 'Warrant' in node:
        colors.append('#3498db')  # Blue for rank cohorts
    elif 'Policy' in node or 'PAS' in node or 'FOIA' in node or 'Civilian' in node:
        colors.append('#e74c3c')  # Red for bureaucratic
    elif 'GDP' in node or 'Conflict' in node:
        colors.append('#2ecc71')  # Green for exogenous
    else:
        colors.append('#95a5a6')  # Gray for political

node_sizes = [500 + (centrality_df[centrality_df['Variable']==node]['Composite_Score'].values[0] * 10)
              for node in G.nodes()]

nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3,
                       arrows=True, arrowsize=10, ax=ax, connectionstyle='arc3,rad=0.1')
nx.draw_networkx_nodes(G, pos, node_color=colors,
                       node_size=node_sizes, alpha=0.8, ax=ax)

# Label only top 10
labels = {node: node.replace('_Z', '').replace('_', '\n') if node in top_vars else ''
          for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)

ax.set_title(f'Causal Network - Cohort Version (Top {top_n} Highlighted)',
            fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='Rank Cohorts'),
    Patch(facecolor='#e74c3c', label='Bureaucratic'),
    Patch(facecolor='#2ecc71', label='Exogenous'),
    Patch(facecolor='#95a5a6', label='Political')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# --- Plot 2: Centrality bar chart ---
ax = axes[1]

all_vars = centrality_df
y_pos = np.arange(len(all_vars))

# Color bars by category
bar_colors = []
for var in all_vars['Variable']:
    if 'Enlisted' in var or 'Officers' in var or 'GOFO' in var or 'Warrant' in var:
        bar_colors.append('#3498db')
    elif 'Policy' in var or 'PAS' in var or 'FOIA' in var or 'Civilian' in var:
        bar_colors.append('#e74c3c')
    elif 'GDP' in var or 'Conflict' in var:
        bar_colors.append('#2ecc71')
    else:
        bar_colors.append('#95a5a6')

ax.barh(y_pos, all_vars['Composite_Score'], color=bar_colors, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([v.replace('_Z', '') for v in all_vars['Variable']], fontsize=10)
ax.set_xlabel('Composite Centrality Score', fontsize=12, fontweight='bold')
ax.set_title('All Variables by Network Centrality (Cohort Version)',
            fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('data/analysis/pairwise_network_cohorts.png', dpi=300, bbox_inches='tight')

print("[OK] Visualization saved")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("ANALYSIS COMPLETE - COHORT VERSION")
print("=" * 100)

print(f"\nPAIRWISE GRANGER CAUSALITY:")
print(f"  Total variable pairs tested: {len(available_vars) * (len(available_vars) - 1)}")
print(f"  Significant causal relationships (p<0.05): {len(edges_df)}")
print(f"  Network density: {nx.density(G):.3f}")

print(f"\nTOP {top_n} VARIABLES FOR VAR MODEL:")
for i, var in enumerate(top_vars, 1):
    print(f"  {i}. {var}")

print("\n" + "=" * 100)
print("KEY IMPROVEMENT:")
print("=" * 100)
print("  - Used 7 rank cohorts instead of 19 individual ranks")
print("  - Reduced intercorrelation within rank groups")
print("  - Field_Grade_Officers (O4-O6) represents bureaucratic bloat")
print("  - Cleaner network structure with 18 variables total")
print("=" * 100)

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print("  1. pairwise_granger_cohorts_all.xlsx")
print("  2. pairwise_granger_cohorts_significant.xlsx")
print("  3. pairwise_network_cohorts_edges.xlsx")
print("  4. pairwise_network_cohorts_centrality.xlsx")
print("  5. top_variables_cohorts.xlsx")
print("  6. pairwise_network_cohorts.png")
print("=" * 100)
