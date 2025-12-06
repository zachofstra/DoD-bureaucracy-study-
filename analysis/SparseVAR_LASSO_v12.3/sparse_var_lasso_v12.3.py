"""
Sparse VAR with LASSO - DoD Bureaucratic Growth Analysis v12.3
Uses L1 regularization to identify key variable relationships and network structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("SPARSE VAR WITH LASSO - NETWORK ANALYSIS (v12.3)")
print("=" * 100)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = 'analysis/complete_normalized_dataset_v12.3.xlsx'
OUTPUT_DIR = 'analysis/SparseVAR_LASSO_v12.3'
MAX_LAG = 3  # Test lags 1-3
ALPHA_CV_FOLDS = 5  # Cross-validation folds for LASSO
MIN_COEF_THRESHOLD = 0.05  # Minimum coefficient to show in network

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/6] Loading data...")

df = pd.read_excel(DATA_FILE)

# Get all numeric columns (exclude Year/FY if present)
exclude_cols = ['Year', 'FY', 'Unnamed: 0']
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
variables = [col for col in numeric_cols if col not in exclude_cols]

print(f"  Loaded: {len(df)} observations")
print(f"  Variables: {len(variables)}")
print(f"\n  Variable list:")
for i, var in enumerate(variables, 1):
    print(f"    {i:2d}. {var}")

# Prepare data
data = df[variables].dropna()
n_obs, n_vars = data.shape

print(f"\n  After dropping NaN: {n_obs} observations, {n_vars} variables")

# =============================================================================
# PREPARE LAG MATRICES
# =============================================================================
print(f"\n[2/6] Creating lag matrices (max lag = {MAX_LAG})...")

def create_lag_matrix(data, lag):
    """Create lagged design matrix for VAR."""
    n = len(data)
    X = []
    y = []

    for t in range(lag, n):
        # Lagged values (predictors)
        lag_values = []
        for l in range(1, lag + 1):
            lag_values.extend(data.iloc[t - l].values)
        X.append(lag_values)

        # Current values (targets)
        y.append(data.iloc[t].values)

    return np.array(X), np.array(y)

# Create lag matrices for each lag order
lag_data = {}
for lag in range(1, MAX_LAG + 1):
    X, y = create_lag_matrix(data, lag)
    lag_data[lag] = {'X': X, 'y': y, 'n_samples': len(X)}
    print(f"  Lag {lag}: X shape = {X.shape}, y shape = {y.shape}")

# =============================================================================
# LASSO REGRESSION FOR EACH VARIABLE AND LAG
# =============================================================================
print(f"\n[3/6] Running LASSO regression with {ALPHA_CV_FOLDS}-fold CV...")

results = {}

for lag in range(1, MAX_LAG + 1):
    print(f"\n  === LAG {lag} ===")

    X = lag_data[lag]['X']
    y = lag_data[lag]['y']

    # Store coefficients for each target variable
    coef_matrix = np.zeros((n_vars, n_vars * lag))
    alpha_values = []
    scores = []

    for i, target_var in enumerate(variables):
        # Target variable
        y_target = y[:, i]

        # LASSO with cross-validation to find optimal alpha
        lasso_cv = LassoCV(cv=ALPHA_CV_FOLDS, random_state=42, max_iter=5000)
        lasso_cv.fit(X, y_target)

        # Store results
        coef_matrix[i, :] = lasso_cv.coef_
        alpha_values.append(lasso_cv.alpha_)
        scores.append(lasso_cv.score(X, y_target))

        # Count non-zero coefficients (selected variables)
        n_selected = np.sum(np.abs(lasso_cv.coef_) > 1e-6)

        print(f"    {target_var:30s}: alpha={lasso_cv.alpha_:8.4f}, R2={lasso_cv.score(X, y_target):6.3f}, selected={n_selected:3d}/{len(lasso_cv.coef_):3d}")

    # Reshape coefficient matrix: (n_vars, lag, n_vars)
    coef_tensor = coef_matrix.reshape(n_vars, lag, n_vars)

    results[lag] = {
        'coef_matrix': coef_matrix,
        'coef_tensor': coef_tensor,
        'alpha_values': alpha_values,
        'scores': scores,
        'n_samples': len(X)
    }

    avg_r2 = np.mean(scores)
    print(f"  Average R2: {avg_r2:.3f}")

# =============================================================================
# SELECT BEST LAG ORDER
# =============================================================================
print("\n[4/6] Selecting optimal lag order...")

# Compare average R2 across lags
avg_r2_by_lag = {lag: np.mean(results[lag]['scores']) for lag in range(1, MAX_LAG + 1)}

best_lag = max(avg_r2_by_lag, key=avg_r2_by_lag.get)

print(f"\n  Average R2 by lag order:")
for lag, r2 in avg_r2_by_lag.items():
    marker = " <-- BEST" if lag == best_lag else ""
    print(f"    Lag {lag}: {r2:.4f}{marker}")

print(f"\n  Selected lag order: {best_lag}")

# =============================================================================
# EXTRACT NETWORK STRUCTURE
# =============================================================================
print(f"\n[5/6] Extracting network structure from Lag {best_lag}...")

coef_tensor = results[best_lag]['coef_tensor']

# Aggregate coefficients across all lags (preserve signs for color coding)
network_matrix_abs = np.zeros((n_vars, n_vars))
network_matrix_signed = np.zeros((n_vars, n_vars))

for l in range(best_lag):
    network_matrix_abs += np.abs(coef_tensor[:, l, :].T)  # Transpose: from -> to
    network_matrix_signed += coef_tensor[:, l, :].T  # Keep signs

# Apply threshold
mask = network_matrix_abs > MIN_COEF_THRESHOLD
network_matrix_abs = network_matrix_abs * mask
network_matrix_signed = network_matrix_signed * mask

# Count edges
n_edges = np.sum(network_matrix_abs > 0)
print(f"  Network edges (|coef| > {MIN_COEF_THRESHOLD}): {n_edges}")

# Save network matrix
network_df = pd.DataFrame(network_matrix_abs, index=variables, columns=variables)
network_df.to_excel(f'{OUTPUT_DIR}/network_adjacency_matrix_lag{best_lag}.xlsx')
print(f"  Saved adjacency matrix to network_adjacency_matrix_lag{best_lag}.xlsx")

# Save signed matrix too
network_signed_df = pd.DataFrame(network_matrix_signed, index=variables, columns=variables)
network_signed_df.to_excel(f'{OUTPUT_DIR}/network_adjacency_matrix_signed_lag{best_lag}.xlsx')
print(f"  Saved signed adjacency matrix to network_adjacency_matrix_signed_lag{best_lag}.xlsx")

# =============================================================================
# CREATE NETWORK DIAGRAM
# =============================================================================
print("\n[6/6] Creating network visualization...")

# Create directed graph
G = nx.DiGraph()

# Add nodes
for var in variables:
    G.add_node(var)

# Add edges with sign information
edge_list = []
for i, var_from in enumerate(variables):
    for j, var_to in enumerate(variables):
        weight_abs = network_matrix_abs[j, i]  # j=to, i=from
        weight_signed = network_matrix_signed[j, i]
        if weight_abs > MIN_COEF_THRESHOLD:
            # Determine relationship type
            relationship = 'amplifying' if weight_signed > 0 else 'dampening'
            edge_color = '#e74c3c' if weight_signed > 0 else '#3498db'  # Red for +, Blue for -

            G.add_edge(var_from, var_to,
                      weight=weight_abs,
                      signed_weight=weight_signed,
                      relationship=relationship,
                      color=edge_color)
            edge_list.append({
                'from': var_from,
                'to': var_to,
                'weight': weight_abs,
                'signed_weight': weight_signed,
                'relationship': relationship
            })

print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")

# Save edge list
edge_df = pd.DataFrame(edge_list)
if len(edge_df) > 0:
    edge_df = edge_df.sort_values('weight', ascending=False)
    edge_df.to_excel(f'{OUTPUT_DIR}/network_edges_lag{best_lag}.xlsx', index=False)
    print(f"  Saved edge list to network_edges_lag{best_lag}.xlsx")

# Calculate network metrics
print("\n  Network metrics:")
degree_centrality = nx.degree_centrality(G)
in_degree_centrality = nx.in_degree_centrality(G)
out_degree_centrality = nx.out_degree_centrality(G)

# Top influential variables (out-degree)
sorted_out = sorted(out_degree_centrality.items(), key=lambda x: x[1], reverse=True)
print("\n  Most influential variables (out-degree):")
for var, cent in sorted_out[:5]:
    print(f"    {var:30s}: {cent:.3f}")

# Most influenced variables (in-degree)
sorted_in = sorted(in_degree_centrality.items(), key=lambda x: x[1], reverse=True)
print("\n  Most influenced variables (in-degree):")
for var, cent in sorted_in[:5]:
    print(f"    {var:30s}: {cent:.3f}")

# =============================================================================
# VISUALIZE NETWORK
# =============================================================================
fig, ax = plt.subplots(1, 1, figsize=(20, 16), facecolor='white')

# Layout - spring layout for network structure
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Node colors by category
def get_node_category(var_name):
    """Categorize variables for color coding."""
    var_lower = var_name.lower()
    if 'enlisted' in var_lower and 'junior' in var_lower:
        return 'Junior Enlisted', '#3498db'
    elif 'enlisted' in var_lower and ('senior' in var_lower or 'e6' in var_lower or 'e7' in var_lower or 'e8' in var_lower or 'e9' in var_lower):
        return 'Senior Enlisted', '#2980b9'
    elif 'officer' in var_lower or 'gofo' in var_lower:
        return 'Officers', '#e74c3c'
    elif 'civilian' in var_lower or 'pas' in var_lower:
        return 'Civilians/Political', '#2ecc71'
    elif 'policy' in var_lower or 'foia' in var_lower or 'directive' in var_lower:
        return 'Bureaucratic Measures', '#f39c12'
    elif 'gdp' in var_lower or 'conflict' in var_lower or 'budget' in var_lower:
        return 'External Factors', '#9b59b6'
    elif 'republican' in var_lower or 'democrat' in var_lower or 'hor' in var_lower or 'senate' in var_lower or 'president' in var_lower:
        return 'Political Party', '#9b59b6'
    else:
        return 'Other', '#95a5a6'

node_colors = []
categories_seen = set()
for node in G.nodes():
    category, color = get_node_category(node)
    node_colors.append(color)
    categories_seen.add((category, color))

# Draw edges FIRST (so arrows appear on top of edge but under node labels)
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
edge_colors = [G[u][v]['color'] for u, v in edges]
max_weight = max(weights) if weights else 1

# Normalize weights for visual thickness
edge_widths = [5 * (w / max_weight) for w in weights]  # Increased from 3 to 5

nx.draw_networkx_edges(G, pos,
                       edgelist=edges,
                       width=edge_widths,
                       alpha=0.6,  # Slightly higher alpha for better color visibility
                       edge_color=edge_colors,  # Use colors from graph attributes
                       arrowsize=35,  # Increased from 20 to 35
                       arrowstyle='-|>',  # Triangle arrowhead
                       connectionstyle='arc3,rad=0.15',  # More curve
                       node_size=2500,  # Reduced to show arrows better
                       min_source_margin=15,  # Space from source node
                       min_target_margin=15,  # Space before target node
                       ax=ax)

# Draw nodes on top
nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       node_size=2500,  # Reduced from 3000
                       edgecolors='black',
                       linewidths=2.5,
                       ax=ax)

# Labels
nx.draw_networkx_labels(G, pos,
                       font_size=9,
                       font_weight='bold',
                       ax=ax)

# Title
ax.set_title(f'Sparse VAR Network (LASSO) - DoD Bureaucratic Growth v12.3\n' +
            f'Lag Order: {best_lag} | Coefficient Threshold: {MIN_COEF_THRESHOLD}\n' +
            f'Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}\n' +
            'Arrow thickness = Coefficient magnitude | Direction shows causal influence',
            fontsize=16, fontweight='bold', pad=20)

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = []

# Node categories
for category, color in sorted(categories_seen):
    legend_elements.append(Patch(facecolor=color, label=category, edgecolor='black', linewidth=1.5))

# Add separator
legend_elements.append(Patch(facecolor='white', label='', edgecolor='none'))

# Edge colors
legend_elements.append(Line2D([0], [0], color='#e74c3c', linewidth=4, label='Amplifying (+) - moves together'))
legend_elements.append(Line2D([0], [0], color='#3498db', linewidth=4, label='Dampening (-) - moves inversely'))

ax.legend(handles=legend_elements,
         loc='upper left',
         fontsize=10,
         framealpha=0.95,
         title='Network Key',
         title_fontsize=11)

# Info box
info_text = f"""SPARSE VAR NETWORK ANALYSIS
Regularization: LASSO (L1)
Cross-validation: {ALPHA_CV_FOLDS}-fold
Best lag: {best_lag}
Avg R2: {avg_r2_by_lag[best_lag]:.3f}

Edges shown: |coefficient| > {MIN_COEF_THRESHOLD}
Total relationships: {G.number_of_edges()}

Interpretation:
• Arrow points from cause to effect
• Thickness = strength of relationship
• RED = Amplifying (positive coefficient)
• BLUE = Dampening (negative coefficient)
• LASSO shrinks weak relationships to zero
• Shows sparse, interpretable network
"""

ax.text(0.98, 0.02, info_text,
       transform=ax.transAxes,
       fontsize=10,
       verticalalignment='bottom',
       horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
       family='monospace')

ax.axis('off')
plt.tight_layout()

# Save figure
output_file = f'{OUTPUT_DIR}/sparse_var_network_lag{best_lag}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n  Network diagram saved: {output_file}")

# =============================================================================
# SAVE COEFFICIENT MATRICES
# =============================================================================
print("\n  Saving coefficient matrices...")

for lag in range(1, MAX_LAG + 1):
    # Save each lag's coefficient matrix
    coef_tensor = results[lag]['coef_tensor']

    for l in range(lag):
        coef_df = pd.DataFrame(
            coef_tensor[:, l, :].T,  # Transpose: rows=from, cols=to
            index=variables,
            columns=variables
        )
        coef_df.to_excel(f'{OUTPUT_DIR}/coefficients_lag{lag}_t-{l+1}.xlsx')

    # Save R2 scores
    scores_df = pd.DataFrame({
        'Variable': variables,
        'R_squared': results[lag]['scores'],
        'LASSO_alpha': results[lag]['alpha_values']
    })
    scores_df.to_excel(f'{OUTPUT_DIR}/model_fit_lag{lag}.xlsx', index=False)

print(f"  Coefficient matrices saved for lags 1-{MAX_LAG}")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 100)
print("SPARSE VAR ANALYSIS COMPLETE (v12.3)")
print("=" * 100)

print(f"\nBest lag order: {best_lag}")
print(f"Average R2: {avg_r2_by_lag[best_lag]:.4f}")
print(f"Network density: {nx.density(G):.3f}")
print(f"Number of edges: {G.number_of_edges()}")

print("\nTop 5 most influential variables:")
for i, (var, cent) in enumerate(sorted_out[:5], 1):
    out_deg = G.out_degree(var)
    in_deg = G.in_degree(var)
    print(f"  {i}. {var:30s} - Out: {out_deg:2d}, In: {in_deg:2d}, Centrality: {cent:.3f}")

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print(f"  1. sparse_var_network_lag{best_lag}.png - Network visualization")
print(f"  2. network_adjacency_matrix_lag{best_lag}.xlsx - Adjacency matrix")
print(f"  3. network_edges_lag{best_lag}.xlsx - Edge list with weights")
print(f"  4. coefficients_lag*_t-*.xlsx - Coefficient matrices for each lag")
print(f"  5. model_fit_lag*.xlsx - R2 and alpha values for each variable")
print("=" * 100)

print("\nInterpretation:")
print("  - LASSO identifies SPARSE, interpretable relationships")
print("  - Only statistically important connections shown")
print("  - Network reveals key drivers of bureaucratic growth")
print("  - Compare with VECM to see long-run vs short-run structure")
print("=" * 100)
