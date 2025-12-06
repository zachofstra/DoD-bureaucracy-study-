"""
Pairwise VAR and Granger Causality Network Analysis - v12.3
Using updated political party variables (Democrat/Republican separated)

Selects top 6-8 variables for VECM analysis based on network centrality
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
print("PAIRWISE VAR AND GRANGER CAUSALITY - v12.3 DATASET")
print("All variables already normalized (z-scored)")
print("=" * 100)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/6] Loading v12.3 normalized dataset...")

df = pd.read_excel('analysis/complete_normalized_dataset_v12.3.xlsx')

# All 19 variables from v12.3
analysis_vars = [
    'Junior_Enlisted_Z',
    'Middle_Enlisted_Z',
    'Senior_Enlisted_Z',
    'Company_Grade_Officers_Z',
    'Field_Grade_Officers_Z',
    'GOFOs_Z',
    'Warrant_Officers_Z',
    'GDP_Growth_Z',
    'Major_Conflict',
    'Policy_Count_Log',
    'Total_Civilians_Z',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z',
    'Democrat Party HOR',
    'Republican Party HOR',
    'Democrat Party Senate',
    'Republican Party Senate',
    'POTUS Democrat Party',
    'POTUS Republican Party'
]

# Filter to available columns
available_vars = [v for v in analysis_vars if v in df.columns]
print(f"  Total variables for analysis: {len(available_vars)}")

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

granger_df.to_excel('analysis/pairwise_granger_all_v12.3.xlsx', index=False)

granger_sig = granger_df[granger_df['Significant_5pct']].copy()
granger_sig.to_excel('analysis/pairwise_granger_significant_v12.3.xlsx', index=False)

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
edges_df.to_excel('analysis/pairwise_network_edges_v12.3.xlsx', index=False)

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
centrality_df.to_excel('analysis/pairwise_network_centrality_v12.3.xlsx', index=False)

print("\n  All variables ranked by composite centrality score:")
print("  " + "-" * 96)
print(centrality_df[['Variable', 'Total_Degree', 'Composite_Score']].to_string(index=False))

# =============================================================================
# SELECT TOP VARIABLES (6-8 RANGE)
# =============================================================================
print("\n[5/6] Selecting top variables for VECM model...")

# Try different cutoffs (6, 7, 8 variables)
for top_n in [6, 7, 8]:
    top_vars = centrality_df.head(top_n)['Variable'].tolist()

    print(f"\n  TOP {top_n} VARIABLES OPTION:")
    print("  " + "=" * 96)
    for i, var in enumerate(top_vars, 1):
        score = centrality_df[centrality_df['Variable'] == var]['Composite_Score'].values[0]
        degree = centrality_df[centrality_df['Variable'] == var]['Total_Degree'].values[0]
        print(f"  {i:2d}. {var:30s} (Composite Score: {score:6.2f}, Degree: {degree:2.0f})")

# Save top 8 as default
top_n = 8
top_vars = centrality_df.head(top_n)['Variable'].tolist()

selection_df = centrality_df.head(top_n)[['Variable', 'Total_Degree', 'Composite_Score']]
selection_df.to_excel('analysis/top_variables_for_vecm_v12.3.xlsx', index=False)

print(f"\n  DEFAULT SELECTION: Top {top_n} variables saved to top_variables_for_vecm_v12.3.xlsx")

# =============================================================================
# VISUALIZE NETWORK
# =============================================================================
print("\n[6/6] Creating network visualization...")

fig, axes = plt.subplots(1, 2, figsize=(24, 12), facecolor='white')

# Network diagram
ax = axes[0]
pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

node_colors = ['#e74c3c' if node in top_vars else '#95a5a6' for node in G.nodes()]
node_sizes = [1500 if node in top_vars else 400 for node in G.nodes()]

nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3,
                       arrows=True, arrowsize=10, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                       node_size=node_sizes, alpha=0.8, ax=ax)

labels = {node: node if node in top_vars else '' for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)

ax.set_title(f'Granger Causality Network (v12.3) - Top {top_n} Variables Highlighted',
            fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Centrality bar chart
ax = axes[1]
top_15 = centrality_df.head(15)
y_pos = np.arange(len(top_15))

ax.barh(y_pos, top_15['Composite_Score'], color='#3498db', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_15['Variable'], fontsize=10)
ax.set_xlabel('Composite Centrality Score', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Variables by Network Centrality (v12.3)',
            fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Highlight top 8
for i in range(min(top_n, len(top_15))):
    ax.get_children()[i].set_color('#e74c3c')

plt.tight_layout()
plt.savefig('analysis/pairwise_network_analysis_v12.3.png', dpi=300, bbox_inches='tight')

print("  [OK] Visualization saved")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("ANALYSIS COMPLETE - v12.3 DATASET")
print("=" * 100)

print(f"\nPAIRWISE GRANGER CAUSALITY:")
print(f"  Total variables tested: {len(available_vars)}")
print(f"  Total pairwise tests: {len(available_vars) * (len(available_vars) - 1)}")
print(f"  Significant causal relationships (p<0.05): {len(edges_df)}")
print(f"  Network density: {nx.density(G):.3f}")

print(f"\nRECOMMENDED: TOP {top_n} VARIABLES FOR VECM:")
for i, var in enumerate(top_vars, 1):
    print(f"  {i}. {var}")

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print("  1. pairwise_granger_all_v12.3.xlsx - All Granger test results")
print("  2. pairwise_granger_significant_v12.3.xlsx - Significant relationships only")
print("  3. pairwise_network_edges_v12.3.xlsx - Network edge list")
print("  4. pairwise_network_centrality_v12.3.xlsx - Centrality rankings for all variables")
print("  5. top_variables_for_vecm_v12.3.xlsx - Top 8 variables recommended for VECM")
print("  6. pairwise_network_analysis_v12.3.png - Network visualization")

print("\n" + "=" * 100)
print("NEXT STEPS:")
print("=" * 100)
print("  1. Use these 8 variables for VECM analysis")
print("  2. Run Johansen cointegration test (lag sensitivity)")
print("  3. Estimate VECM model with optimal lag order")
print("  4. Run robustness tests and document equations")
print("  5. Compare results with v11.8 VECM")
print("=" * 100)
