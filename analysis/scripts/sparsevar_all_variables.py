"""
SparseVAR Analysis - All 17 Variables with LASSO Regularization

Uses LASSO (L1 regularization) to identify sparse causal network
among all available variables, creating weighted directed graph.

Variables (17 total):
- 7 Rank Cohorts (z-scored)
- 5 Bureaucratic measures
- 2 Exogenous controls
- 3 Political variables

Method:
- Apply LASSO regularization to VAR coefficients
- Cross-validation to select optimal regularization strength
- Extract non-zero coefficients as weighted network edges
- Granger causality tests on identified relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV, Lasso
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("SPARSEVAR ANALYSIS - ALL 17 VARIABLES WITH LASSO REGULARIZATION")
print("=" * 100)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/8] Loading complete dataset...")

df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

# All 17 variables
all_vars = [
    # Rank cohorts (z-scored)
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
available_vars = [v for v in all_vars if v in df.columns]
print(f"  Total variables: {len(available_vars)}")

data = df[available_vars].copy()
data = data.dropna()
print(f"  Observations: {len(data)}")

# =============================================================================
# STATIONARITY TESTING AND TRANSFORMATION
# =============================================================================
print("\n[2/8] Testing stationarity and applying transformations...")
print("  " + "-" * 96)

stationarity_results = []
data_transformed = data.copy()

for var in available_vars:
    # ADF test
    adf_result = adfuller(data[var].dropna(), maxlag=4, regression='ct')
    adf_pval = adf_result[1]

    # KPSS test
    kpss_result = kpss(data[var].dropna(), regression='ct', nlags=4)
    kpss_pval = kpss_result[1]

    # Decision
    if adf_pval < 0.05 and kpss_pval > 0.05:
        decision = "STATIONARY"
        transform = "LEVELS"
    else:
        decision = "NON-STATIONARY"
        transform = "DIFFERENCE"
        data_transformed[var] = data[var].diff()

    stationarity_results.append({
        'Variable': var,
        'ADF_pvalue': adf_pval,
        'KPSS_pvalue': kpss_pval,
        'Decision': decision,
        'Transform': transform
    })

    status = "LEVELS" if transform == "LEVELS" else "DIFF"
    print(f"  {var:35s} ADF={adf_pval:.4f}, KPSS={kpss_pval:.4f} -> {status}")

data_transformed = data_transformed.dropna()
print(f"\n  Observations after differencing: {len(data_transformed)}")

# Save stationarity results
stationarity_df = pd.DataFrame(stationarity_results)
stationarity_df.to_excel('data/analysis/sparsevar_stationarity.xlsx', index=False)

# =============================================================================
# CREATE LAGGED FEATURES FOR VAR(2)
# =============================================================================
print("\n[3/8] Creating lagged features for VAR(2)...")

lag_order = 2
n_vars = len(available_vars)

# Create lagged features
X_lags = []
y_vars = []

for t in range(lag_order, len(data_transformed)):
    # Features: lags 1 and 2 of all variables
    features = []
    for lag in range(1, lag_order + 1):
        features.extend(data_transformed.iloc[t - lag].values)
    X_lags.append(features)

    # Target: current values of all variables
    y_vars.append(data_transformed.iloc[t].values)

X = np.array(X_lags)
Y = np.array(y_vars)

print(f"  Training samples: {X.shape[0]}")
print(f"  Features per sample: {X.shape[1]} ({n_vars} vars x {lag_order} lags)")
print(f"  Target variables: {Y.shape[1]}")

# =============================================================================
# SPARSEVAR WITH LASSO REGULARIZATION
# =============================================================================
print("\n[4/8] Fitting SparseVAR with LASSO regularization...")
print("  Using cross-validation to select optimal alpha...")

# Fit LASSO for each target variable
lasso_models = []
coefficients = np.zeros((n_vars, n_vars * lag_order))

for i, target_var in enumerate(available_vars):
    print(f"  [{i+1}/{n_vars}] {target_var:35s}", end='')

    # Fit LassoCV with cross-validation
    lasso = LassoCV(cv=5, max_iter=5000, random_state=42)
    lasso.fit(X, Y[:, i])

    lasso_models.append(lasso)
    coefficients[i, :] = lasso.coef_

    n_nonzero = np.sum(lasso.coef_ != 0)
    print(f" alpha={lasso.alpha_:8.6f}, non-zero={n_nonzero:3d}/{X.shape[1]}")

print("\n  [OK] SparseVAR fitted successfully")

# =============================================================================
# EXTRACT NETWORK STRUCTURE (with signs preserved)
# =============================================================================
print("\n[5/8] Extracting sparse network structure...")

# Build weighted adjacency matrix (lag 1 only for network)
# Rows = causes, Columns = effects
adjacency = np.zeros((n_vars, n_vars))
adjacency_signed = np.zeros((n_vars, n_vars))

for i in range(n_vars):  # Target variable
    for j in range(n_vars):  # Predictor variable
        # Coefficient for lag 1 of variable j predicting variable i
        coef = coefficients[i, j]
        if coef != 0:
            adjacency[j, i] = abs(coef)  # j causes i, weight is abs(coefficient)
            adjacency_signed[j, i] = coef  # Preserve sign for interpretation

# Create edge list with signs
edges = []
for i in range(n_vars):
    for j in range(n_vars):
        if adjacency[i, j] > 0:
            edges.append({
                'Source': available_vars[i],
                'Target': available_vars[j],
                'Weight': adjacency[i, j],
                'Coefficient': adjacency_signed[i, j],
                'Sign': 'Positive' if adjacency_signed[i, j] > 0 else 'Negative',
                'Lag': 1
            })

edges_df = pd.DataFrame(edges)
edges_df = edges_df.sort_values('Weight', ascending=False)
edges_df.to_excel('data/analysis/sparsevar_network_edges.xlsx', index=False)

print(f"  Total non-zero edges: {len(edges_df)}")
print(f"  Network density: {len(edges_df) / (n_vars * (n_vars - 1)):.3f}")

n_positive = (edges_df['Sign'] == 'Positive').sum()
n_negative = (edges_df['Sign'] == 'Negative').sum()
print(f"  Positive influences: {n_positive} ({n_positive/len(edges_df)*100:.1f}%)")
print(f"  Negative influences: {n_negative} ({n_negative/len(edges_df)*100:.1f}%)")

print("\n  Top 15 strongest weighted edges:")
print("  " + "-" * 96)
for idx, row in edges_df.head(15).iterrows():
    sign_symbol = "+" if row['Sign'] == 'Positive' else "-"
    print(f"    {sign_symbol} {row['Source']:35s} -> {row['Target']:35s} Coef={row['Coefficient']:+.4f}")

# =============================================================================
# NETWORK CENTRALITY ANALYSIS
# =============================================================================
print("\n[6/8] Computing network centrality measures...")

G = nx.DiGraph()
for idx, row in edges_df.iterrows():
    G.add_edge(row['Source'], row['Target'],
               weight=row['Weight'],
               coefficient=row['Coefficient'],
               sign=row['Sign'])

# Ensure all variables are nodes
for var in available_vars:
    if var not in G:
        G.add_node(var)

# Centrality measures
in_degree = dict(G.in_degree(weight='weight'))
out_degree = dict(G.out_degree(weight='weight'))
total_degree = {node: in_degree[node] + out_degree[node] for node in G.nodes()}

try:
    pagerank = nx.pagerank(G, weight='weight')
except:
    pagerank = {node: 1/len(G.nodes()) for node in G.nodes()}

try:
    betweenness = nx.betweenness_centrality(G, weight='weight')
except:
    betweenness = {node: 0 for node in G.nodes()}

centrality_df = pd.DataFrame({
    'Variable': list(G.nodes()),
    'In_Degree_Weighted': [in_degree[n] for n in G.nodes()],
    'Out_Degree_Weighted': [out_degree[n] for n in G.nodes()],
    'Total_Degree_Weighted': [total_degree[n] for n in G.nodes()],
    'PageRank': [pagerank[n] for n in G.nodes()],
    'Betweenness': [betweenness[n] for n in G.nodes()]
})

# Normalize and create composite score
for col in ['PageRank', 'Betweenness']:
    if centrality_df[col].max() > 0:
        centrality_df[f'{col}_Normalized'] = (centrality_df[col] / centrality_df[col].max()) * 100

centrality_df['Composite_Score'] = centrality_df[[
    'Total_Degree_Weighted',
    'PageRank_Normalized',
    'Betweenness_Normalized'
]].mean(axis=1)

centrality_df = centrality_df.sort_values('Composite_Score', ascending=False)
centrality_df.to_excel('data/analysis/sparsevar_centrality.xlsx', index=False)

print("\n  Top variables by composite centrality:")
print("  " + "-" * 96)
print(centrality_df[['Variable', 'Total_Degree_Weighted', 'Composite_Score']].head(10).to_string(index=False))

# =============================================================================
# GRANGER CAUSALITY ON SPARSE NETWORK
# =============================================================================
print("\n[7/8] Running Granger causality tests on sparse network edges...")

granger_results = []

for idx, edge in edges_df.iterrows():
    cause_var = edge['Source']
    effect_var = edge['Target']

    try:
        test_data = data_transformed[[effect_var, cause_var]].dropna()
        if len(test_data) < 15:
            continue

        gc_result = grangercausalitytests(test_data, maxlag=2, verbose=False)

        p_value_lag1 = gc_result[1][0]['ssr_ftest'][1]
        f_stat_lag1 = gc_result[1][0]['ssr_ftest'][0]

        p_value_lag2 = gc_result[2][0]['ssr_ftest'][1]
        f_stat_lag2 = gc_result[2][0]['ssr_ftest'][0]

        granger_results.append({
            'Cause': cause_var,
            'Effect': effect_var,
            'LASSO_Weight': edge['Weight'],
            'Granger_F_Lag1': f_stat_lag1,
            'Granger_p_Lag1': p_value_lag1,
            'Granger_F_Lag2': f_stat_lag2,
            'Granger_p_Lag2': p_value_lag2,
            'Significant_5pct': min(p_value_lag1, p_value_lag2) < 0.05
        })
    except:
        pass

granger_df = pd.DataFrame(granger_results)
granger_df = granger_df.sort_values('LASSO_Weight', ascending=False)
granger_df.to_excel('data/analysis/sparsevar_granger_validation.xlsx', index=False)

n_significant = granger_df['Significant_5pct'].sum()
print(f"  Granger causality tests: {len(granger_df)}")
print(f"  Significant at 5%: {n_significant} ({n_significant/len(granger_df)*100:.1f}%)")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n[8/8] Creating network visualization...")

fig, axes = plt.subplots(1, 2, figsize=(24, 12), facecolor='white')

# --- Plot 1: Full weighted network ---
ax = axes[0]

pos = nx.spring_layout(G, k=3, iterations=50, seed=42, weight='weight')

# Node colors by category
node_colors = []
for node in G.nodes():
    if 'Enlisted' in node:
        node_colors.append('#3498db')  # Blue
    elif 'Officers' in node or 'GOFO' in node or 'Warrant' in node:
        node_colors.append('#e74c3c')  # Red
    elif 'Policy' in node or 'PAS' in node or 'FOIA' in node or 'Civilian' in node:
        node_colors.append('#2ecc71')  # Green
    elif 'GDP' in node or 'Conflict' in node:
        node_colors.append('#f39c12')  # Orange
    else:
        node_colors.append('#95a5a6')  # Gray

# Node sizes based on centrality
node_sizes = [300 + centrality_df[centrality_df['Variable']==node]['Composite_Score'].values[0] * 15
              for node in G.nodes()]

# Edge widths and colors based on weights and signs
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
max_weight = max(edge_weights) if edge_weights else 1
edge_widths = [1 + (w / max_weight) * 4 for w in edge_weights]

# Color edges by sign: Red = positive influence, Blue = negative influence
edge_colors = []
for u, v in G.edges():
    if G[u][v]['sign'] == 'Positive':
        edge_colors.append('#e74c3c')  # Red for positive
    else:
        edge_colors.append('#3498db')  # Blue for negative

nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6,
                       width=edge_widths, arrows=True, arrowsize=15,
                       ax=ax, connectionstyle='arc3,rad=0.15')

nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                       node_size=node_sizes, alpha=0.8, ax=ax)

labels = {node: node.replace('_Z', '').replace('_', '\n') for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)

ax.set_title('SparseVAR Network - All 17 Variables (LASSO Regularized)\nEdge: Red=Positive influence, Blue=Negative influence, Width=Strength',
            fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#3498db', label='Enlisted Cohorts'),
    Patch(facecolor='#e74c3c', label='Officer Cohorts'),
    Patch(facecolor='#2ecc71', label='Bureaucratic'),
    Patch(facecolor='#f39c12', label='Exogenous'),
    Patch(facecolor='#95a5a6', label='Political'),
    Line2D([0], [0], color='#e74c3c', linewidth=3, label='Positive Influence', alpha=0.6),
    Line2D([0], [0], color='#3498db', linewidth=3, label='Negative Influence', alpha=0.6)
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

# --- Plot 2: Centrality rankings ---
ax = axes[1]

top_15 = centrality_df.head(15)
y_pos = np.arange(len(top_15))

bar_colors = []
for var in top_15['Variable']:
    if 'Enlisted' in var:
        bar_colors.append('#3498db')
    elif 'Officers' in var or 'GOFO' in var or 'Warrant' in var:
        bar_colors.append('#e74c3c')
    elif 'Policy' in var or 'PAS' in var or 'FOIA' in var or 'Civilian' in var:
        bar_colors.append('#2ecc71')
    elif 'GDP' in var or 'Conflict' in var:
        bar_colors.append('#f39c12')
    else:
        bar_colors.append('#95a5a6')

ax.barh(y_pos, top_15['Composite_Score'], color=bar_colors, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([v.replace('_Z', '') for v in top_15['Variable']], fontsize=10)
ax.set_xlabel('Composite Centrality Score', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Variables by Network Centrality\n(SparseVAR with LASSO)',
            fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('data/analysis/sparsevar_network_weighted.png', dpi=300, bbox_inches='tight')
print("  [OK] Network visualization saved")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("SPARSEVAR ANALYSIS COMPLETE")
print("=" * 100)

print(f"\nMODEL SPECIFICATION:")
print(f"  Variables: {n_vars}")
print(f"  Lag order: {lag_order}")
print(f"  Observations: {X.shape[0]}")
print(f"  Potential parameters: {n_vars * n_vars * lag_order} ({n_vars} equations x {n_vars * lag_order} features)")

print(f"\nSPARSITY (via LASSO):")
total_possible = n_vars * n_vars
total_nonzero = len(edges_df)
sparsity = (total_possible - total_nonzero) / total_possible * 100
print(f"  Non-zero edges: {total_nonzero} out of {total_possible} possible")
print(f"  Sparsity: {sparsity:.1f}%")

print(f"\nGRANGER VALIDATION:")
print(f"  Edges tested: {len(granger_df)}")
print(f"  Significant (p<0.05): {n_significant} ({n_significant/len(granger_df)*100:.1f}%)")

print(f"\nTOP 5 MOST CENTRAL VARIABLES:")
for i, row in centrality_df.head(5).iterrows():
    print(f"  {i+1}. {row['Variable']:35s} (Score={row['Composite_Score']:.2f})")

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print("  1. sparsevar_stationarity.xlsx - Stationarity tests and transformations")
print("  2. sparsevar_network_edges.xlsx - Weighted network edges from LASSO")
print("  3. sparsevar_centrality.xlsx - Network centrality measures")
print("  4. sparsevar_granger_validation.xlsx - Granger tests on sparse edges")
print("  5. sparsevar_network_weighted.png - Network visualization")
print("=" * 100)

print("\nKEY ADVANTAGE:")
print("  SparseVAR uses LASSO regularization to automatically identify which of the")
print(f"  {n_vars * n_vars * lag_order} possible coefficients are truly important, avoiding overfitting")
print("  while using all 17 variables.")
print("=" * 100)
