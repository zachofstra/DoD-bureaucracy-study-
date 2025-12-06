"""
Pairwise VAR and Granger Causality Network Analysis - FINAL VERSION
Using z-score normalized rank percentages from updated dataset

All rank percentages now z-scored in the dataset itself
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
print("PAIRWISE VAR AND GRANGER CAUSALITY - USING Z-SCORED RANKS")
print("All rank percentages already normalized in dataset")
print("=" * 100)

# =============================================================================
# LOAD DATA (RANKS ALREADY Z-SCORED)
# =============================================================================
print("\n[1/6] Loading normalized dataset...")

df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

# All variables for analysis (ranks already z-scored in file)
analysis_vars = [
    # Administrative/Policy (already normalized)
    'Policy_Count_Log',
    'Total_PAS_Z',
    'Total_Civilians_Z',
    'FOIA_Simple_Days_Z',
    'FOIA_Complex_Days_Z',

    # Rank percentages (NOW Z-SCORED IN FILE)
    'E1_Pct', 'E2_Pct', 'E3_Pct', 'E4_Pct', 'E5_Pct',
    'E6_Pct', 'E7_Pct', 'E8_Pct', 'E9_Pct',
    'O1_Pct', 'O2_Pct', 'O3_Pct', 'O4_MajorLTCDR_Pct',
    'O5_LtColCDR_Pct', 'O6_ColCAPT_Pct', 'O7_Pct', 'O8_Pct',
    'O9_Pct', 'O10_Pct',

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
total_tests = len(list(combinations(available_vars, 2))) * 2

for var1 in available_vars:
    for var2 in available_vars:
        if var1 == var2:
            continue

        test_count += 1
        if test_count % 100 == 0:
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

granger_df.to_excel('data/analysis/pairwise_granger_all_FINAL.xlsx', index=False)

granger_sig = granger_df[granger_df['Significant_5pct']].copy()
granger_sig.to_excel('data/analysis/pairwise_granger_significant_FINAL.xlsx', index=False)

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
edges_df.to_excel('data/analysis/pairwise_network_edges_FINAL.xlsx', index=False)

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
centrality_df.to_excel('data/analysis/pairwise_network_centrality_FINAL.xlsx', index=False)

print("\n  Top 20 variables by composite centrality score:")
print("  " + "-" * 96)
print(centrality_df.head(20)[['Variable', 'Total_Degree', 'Composite_Score']].to_string(index=False))

# =============================================================================
# SELECT TOP VARIABLES
# =============================================================================
print("\n[5/6] Selecting top variables for VAR model...")

top_n = 10
top_vars = centrality_df.head(top_n)['Variable'].tolist()

print(f"\n  TOP {top_n} VARIABLES SELECTED (Z-SCORED RANKS):")
print("  " + "=" * 96)
for i, var in enumerate(top_vars, 1):
    score = centrality_df[centrality_df['Variable'] == var]['Composite_Score'].values[0]
    degree = centrality_df[centrality_df['Variable'] == var]['Total_Degree'].values[0]
    print(f"  {i:2d}. {var:30s} (Composite Score: {score:6.2f}, Degree: {degree:2.0f})")

selection_df = centrality_df.head(top_n)[['Variable', 'Total_Degree', 'Composite_Score']]
selection_df.to_excel('data/analysis/top_variables_for_var_FINAL.xlsx', index=False)

# =============================================================================
# VISUALIZE NETWORK
# =============================================================================
print("\n[6/6] Creating network visualization...")

fig, axes = plt.subplots(1, 2, figsize=(24, 12), facecolor='white')

ax = axes[0]
pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

node_colors = ['#e74c3c' if node in top_vars else '#95a5a6' for node in G.nodes()]
node_sizes = [1000 if node in top_vars else 300 for node in G.nodes()]

nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3,
                       arrows=True, arrowsize=10, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                       node_size=node_sizes, alpha=0.8, ax=ax)

labels = {node: node if node in top_vars else '' for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)

ax.set_title(f'Causal Network (Z-Scored Ranks) - Top {top_n} Variables Highlighted',
            fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

ax = axes[1]
top_20 = centrality_df.head(20)
y_pos = np.arange(len(top_20))

ax.barh(y_pos, top_20['Composite_Score'], color='#3498db', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_20['Variable'], fontsize=9)
ax.set_xlabel('Composite Centrality Score', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Variables by Network Centrality (Z-Scored Ranks)',
            fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

for i in range(min(10, len(top_20))):
    ax.get_children()[i].set_color('#e74c3c')

plt.tight_layout()
plt.savefig('data/analysis/pairwise_network_analysis_FINAL.png', dpi=300, bbox_inches='tight')

print("[OK] Visualization saved")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("ANALYSIS COMPLETE - SUMMARY (Z-SCORED RANKS)")
print("=" * 100)

print(f"\nPAIRWISE GRANGER CAUSALITY:")
print(f"  Total variable pairs tested: {len(available_vars) * (len(available_vars) - 1)}")
print(f"  Significant causal relationships (p<0.05): {len(edges_df)}")
print(f"  Network density: {nx.density(G):.3f}")

print(f"\nTOP {top_n} VARIABLES FOR VAR MODEL:")
for i, var in enumerate(top_vars, 1):
    print(f"  {i}. {var}")

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print("  1. pairwise_granger_all_FINAL.xlsx")
print("  2. pairwise_granger_significant_FINAL.xlsx")
print("  3. pairwise_network_edges_FINAL.xlsx")
print("  4. pairwise_network_centrality_FINAL.xlsx")
print("  5. top_variables_for_var_FINAL.xlsx")
print("  6. pairwise_network_analysis_FINAL.png")

print("\n" + "=" * 100)
print("NORMALIZATION APPLIED:")
print("=" * 100)
print("  [OK] All rank percentages z-score normalized in dataset")
print("  [OK] Levels playing field between enlisted (high %) and officers (low %)")
print("  [OK] Prevents magnitude bias in Granger causality tests")
print("=" * 100)
