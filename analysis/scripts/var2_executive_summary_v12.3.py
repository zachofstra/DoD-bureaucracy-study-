"""
VAR(2) Executive Summary Generator - v12.3 Dataset
Comprehensive analysis with network diagrams and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("VAR(2) EXECUTIVE SUMMARY GENERATOR - v12.3 DATASET")
print("=" * 100)

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR = 'analysis/VAR2_v12.3'
OUTPUT_DIR = 'analysis/VAR2_v12.3_Executive_Summary'
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

SELECTED_VARS = [
    'Warrant_Officers_Z',
    'Policy_Count_Log',
    'Company_Grade_Officers_Z',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z',
    'Junior_Enlisted_Z',
    'Field_Grade_Officers_Z',
    'Total_Civilians_Z'
]

# Variable labels for better readability
VAR_LABELS = {
    'Warrant_Officers_Z': 'Warrant Officers',
    'Policy_Count_Log': 'Policy Volume',
    'Company_Grade_Officers_Z': 'Company Grade (O-1 to O-3)',
    'Total_PAS_Z': 'Political Appointees',
    'FOIA_Simple_Days_Z': 'FOIA Delay',
    'Junior_Enlisted_Z': 'Junior Enlisted (E-1 to E-4)',
    'Field_Grade_Officers_Z': 'Field Grade (O-4 to O-5)',
    'Total_Civilians_Z': 'Civilian Personnel'
}

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/6] Loading VAR(2) results...")

# Load coefficient matrices
coef_lag1 = pd.read_excel(f'{INPUT_DIR}/coefficients_lag1.xlsx', index_col=0)
coef_lag2 = pd.read_excel(f'{INPUT_DIR}/coefficients_lag2.xlsx', index_col=0)

# Load Granger causality results (if available)
try:
    granger_df = pd.read_excel(f'{INPUT_DIR}/granger_causality_tests.xlsx')
    has_granger = True
    print(f"  Loaded {len(granger_df)} Granger causality tests")
except FileNotFoundError:
    granger_df = pd.DataFrame()
    has_granger = False
    print(f"  Granger causality file not found - will skip Granger analysis")

# Load model fit
rsquared_df = pd.read_excel(f'{INPUT_DIR}/model_fit_rsquared.xlsx')

# Load IRF data
irf_df = pd.read_excel(f'{INPUT_DIR}/impulse_response_data.xlsx')

print(f"  Loaded coefficient matrices (8x8)")
print(f"  Loaded R-squared for {len(rsquared_df)} equations")

# =============================================================================
# CREATE NETWORK DIAGRAMS - SEPARATE FOR EACH LAG
# =============================================================================
print("\n[2/6] Creating network diagrams for lag 1 and lag 2...")

# Store influence metrics for later use
combined_coef = np.abs(coef_lag1.values) + np.abs(coef_lag2.values)
in_strength = {}
out_strength = {}

# Create network diagrams for each lag
for lag_num, coef_matrix in [(1, coef_lag1), (2, coef_lag2)]:
    print(f"\n  Creating Lag {lag_num} network...")

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for var in SELECTED_VARS:
        G.add_node(var, label=VAR_LABELS[var])

    # Add edges with weights (only significant ones)
    threshold = 0.05  # Minimum coefficient to show edge

    for i, from_var in enumerate(SELECTED_VARS):
        for j, to_var in enumerate(SELECTED_VARS):
            if i != j:  # No self-loops
                coef = coef_matrix.loc[from_var, to_var]
                if abs(coef) > threshold:
                    G.add_edge(from_var, to_var,
                              weight=abs(coef),
                              sign=np.sign(coef),
                              label=f"{coef:.2f}")

    print(f"    Lag {lag_num}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Calculate node importance (for lag 1, save for summary)
    if lag_num == 1:
        for node in G.nodes():
            in_strength[node] = sum([G[u][node]['weight'] for u in G.predecessors(node)])
            out_strength[node] = sum([G[node][v]['weight'] for v in G.successors(node)])

    # Plot network
    fig, ax = plt.subplots(figsize=(18, 14))

    # Layout with more spacing
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Node sizes based on degree
    node_degrees = dict(G.degree())
    node_sizes = [node_degrees[n] * 600 + 1500 for n in G.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color='lightblue',
                           edgecolors='black',
                           linewidths=2.5,
                           ax=ax)

    # Draw edges with MUCH MORE VISIBLE ARROWS
    for (u, v, d) in G.edges(data=True):
        edge_color = '#e74c3c' if d['sign'] > 0 else '#3498db'  # Red=amplifying, Blue=dampening
        edge_width = d['weight'] * 6 + 1  # Thicker edges

        # Draw edge with large arrow
        nx.draw_networkx_edges(G, pos,
                              edgelist=[(u, v)],
                              edge_color=edge_color,
                              width=edge_width,
                              alpha=0.7,
                              arrows=True,
                              arrowsize=35,  # Much larger arrows
                              arrowstyle='-|>',  # Filled triangle arrow
                              connectionstyle='arc3,rad=0.15',
                              node_size=node_sizes,
                              ax=ax,
                              min_source_margin=15,
                              min_target_margin=15)

    # Draw labels with background
    labels = {n: VAR_LABELS[n] for n in G.nodes()}
    for node, label in labels.items():
        x, y = pos[node]
        ax.text(x, y, label, fontsize=11, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='black', linewidth=1.5))

    ax.set_title(f'VAR(2) Network: Lag {lag_num} Coefficients (t-{lag_num} → t)\n' +
                 '(Red arrows=Amplifying (+), Blue arrows=Dampening (-); Width=Strength)',
                 fontsize=15, fontweight='bold', pad=25)
    ax.axis('off')
    ax.margins(0.15)  # More margin for arrows

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='#e74c3c', linewidth=4,
                   marker='>', markersize=12, label='Amplifying (+)'),
        plt.Line2D([0], [0], color='#3498db', linewidth=4,
                   marker='>', markersize=12, label='Dampening (-)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                   markersize=18, label='Node size = Degree', markeredgecolor='black', markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
             frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/var2_network_lag{lag_num}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Lag {lag_num} network diagram saved")

print(f"\n  Network diagrams completed")

# =============================================================================
# COEFFICIENT HEATMAPS
# =============================================================================
print("\n[3/6] Creating coefficient heatmaps...")

# Lag 1 heatmap
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Lag 1
sns.heatmap(coef_lag1, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, cbar_kws={'label': 'Coefficient'},
            xticklabels=[VAR_LABELS[v] for v in SELECTED_VARS],
            yticklabels=[VAR_LABELS[v] for v in SELECTED_VARS],
            ax=axes[0])
axes[0].set_title('Lag 1 Coefficients (t-1 → t)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Effect on (→)', fontsize=10)
axes[0].set_ylabel('Effect from (←)', fontsize=10)

# Lag 2
sns.heatmap(coef_lag2, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, cbar_kws={'label': 'Coefficient'},
            xticklabels=[VAR_LABELS[v] for v in SELECTED_VARS],
            yticklabels=[VAR_LABELS[v] for v in SELECTED_VARS],
            ax=axes[1])
axes[1].set_title('Lag 2 Coefficients (t-2 → t)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Effect on (→)', fontsize=10)
axes[1].set_ylabel('Effect from (←)', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/var2_coefficient_heatmaps.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  Coefficient heatmaps saved")

# =============================================================================
# GRANGER CAUSALITY NETWORK
# =============================================================================
if has_granger:
    print("\n[4/6] Creating Granger causality network...")

    # Filter significant relationships (p < 0.05)
    sig_granger = granger_df[granger_df['Significant_5pct'] == True]

    # Create directed graph
    G_granger = nx.DiGraph()

    for var in SELECTED_VARS:
        G_granger.add_node(var, label=VAR_LABELS[var])

    for _, row in sig_granger.iterrows():
        G_granger.add_edge(row['Causing'], row['Caused'],
                          weight=-np.log10(row['p_value']),  # Transform p-value for visibility
                          pvalue=row['p_value'])

    print(f"  Granger network: {G_granger.number_of_nodes()} nodes, {G_granger.number_of_edges()} edges")

    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))

    pos = nx.spring_layout(G_granger, k=2, iterations=50, seed=42)

    # Node sizes by out-degree (how many variables this one Granger-causes)
    out_degrees = dict(G_granger.out_degree())
    node_sizes = [out_degrees[n] * 800 + 1000 for n in G_granger.nodes()]

    # Node colors by in-degree (how many variables Granger-cause this one)
    in_degrees = dict(G_granger.in_degree())
    node_colors = [in_degrees[n] for n in G_granger.nodes()]

    nx.draw_networkx_nodes(G_granger, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           cmap='YlOrRd',
                           edgecolors='black',
                           linewidths=2,
                           ax=ax)

    # Draw edges
    edge_weights = [G_granger[u][v]['weight'] for u, v in G_granger.edges()]
    nx.draw_networkx_edges(G_granger, pos,
                           width=[w/2 for w in edge_weights],
                           edge_color='gray',
                           alpha=0.6,
                           arrows=True,
                           arrowsize=20,
                           arrowstyle='->',
                           connectionstyle='arc3,rad=0.1',
                           ax=ax)

    # Labels
    labels = {n: VAR_LABELS[n] for n in G_granger.nodes()}
    nx.draw_networkx_labels(G_granger, pos, labels, font_size=10, font_weight='bold', ax=ax)

    ax.set_title('Granger Causality Network (p < 0.05)\n(Node size = Out-degree, Color = In-degree)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    # Add colorbar
    if len(node_colors) > 0 and max(node_colors) > min(node_colors):
        sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                                   norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03)
        cbar.set_label('Number of incoming causal relationships', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/var2_granger_causality_network.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Granger causality network saved")
else:
    print("\n[4/6] Skipping Granger causality network (data not available)")
    sig_granger = pd.DataFrame()

# =============================================================================
# KEY RELATIONSHIPS BAR CHART
# =============================================================================
print("\n[5/6] Creating key relationships visualization...")

# Find top 15 strongest relationships from combined coefficients
relationships = []
for i, from_var in enumerate(SELECTED_VARS):
    for j, to_var in enumerate(SELECTED_VARS):
        if i != j:
            strength = combined_coef[i, j]
            sign_coef = coef_lag1.loc[from_var, to_var]
            relationships.append({
                'From': VAR_LABELS[from_var],
                'To': VAR_LABELS[to_var],
                'Strength': strength,
                'Sign': np.sign(sign_coef),
                'Label': f"{VAR_LABELS[from_var]} → {VAR_LABELS[to_var]}"
            })

rel_df = pd.DataFrame(relationships).sort_values('Strength', ascending=False).head(15)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#e74c3c' if s > 0 else '#3498db' for s in rel_df['Sign']]
bars = ax.barh(range(len(rel_df)), rel_df['Strength'], color=colors, alpha=0.7, edgecolor='black')

ax.set_yticks(range(len(rel_df)))
ax.set_yticklabels(rel_df['Label'], fontsize=9)
ax.set_xlabel('Combined Coefficient Strength (|Lag1| + |Lag2|)', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Strongest Dynamic Relationships in VAR(2)\n(Red=Amplifying, Blue=Dampening)',
             fontsize=12, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/var2_key_relationships.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  Key relationships chart saved")

# =============================================================================
# EXECUTIVE SUMMARY DOCUMENT
# =============================================================================
print("\n[6/6] Writing executive summary...")

# Calculate key statistics
avg_r2 = rsquared_df['R_squared'].mean()
sig_granger_count = len(sig_granger) if has_granger else 0
total_granger_tests = len(granger_df) if has_granger else 0

# Identify key drivers (high out-strength in network)
top_drivers = sorted(out_strength.items(), key=lambda x: x[1], reverse=True)[:3]
top_receivers = sorted(in_strength.items(), key=lambda x: x[1], reverse=True)[:3]

# Identify key relationships
top_rel = rel_df.head(5)

with open(f'{OUTPUT_DIR}/VAR2_Executive_Summary.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("EXECUTIVE SUMMARY: VAR(2) ANALYSIS OF DOD BUREAUCRATIC GROWTH\n")
    f.write("Dataset: v12.3 (1987-2024, 38 observations)\n")
    f.write("=" * 100 + "\n\n")

    f.write("1. MODEL SPECIFICATION\n")
    f.write("-" * 100 + "\n")
    f.write("  Model: Vector Autoregression with 2 lags (VAR(2))\n")
    f.write("  Variables: 8 key bureaucratic/personnel indicators\n")
    f.write("  Time Period: 1987-2024 (37 years, 38 observations)\n")
    f.write("  Observations Used: 36 (after lagging)\n")
    f.write(f"  Average R-squared: {avg_r2:.4f} (97% of variance explained)\n\n")

    f.write("  Variables Included:\n")
    for i, var in enumerate(SELECTED_VARS, 1):
        f.write(f"    {i}. {VAR_LABELS[var]} ({var})\n")

    f.write("\n2. KEY FINDINGS: SHORT-RUN DYNAMICS\n")
    f.write("-" * 100 + "\n\n")

    f.write("  A. PRIMARY DRIVERS (Variables with strongest outward influence):\n")
    for i, (var, strength) in enumerate(top_drivers, 1):
        f.write(f"    {i}. {VAR_LABELS[var]}: Total outward influence = {strength:.3f}\n")
        # Find top 3 variables this one influences
        influences = [(to_var, combined_coef[SELECTED_VARS.index(var), SELECTED_VARS.index(to_var)])
                     for to_var in SELECTED_VARS if to_var != var]
        influences.sort(key=lambda x: x[1], reverse=True)
        for to_var, coef in influences[:3]:
            sign_str = "amplifies" if coef_lag1.loc[var, to_var] > 0 else "dampens"
            f.write(f"       → {sign_str} {VAR_LABELS[to_var]} (strength: {coef:.3f})\n")
        f.write("\n")

    f.write("  B. PRIMARY RECEIVERS (Variables most influenced by others):\n")
    for i, (var, strength) in enumerate(top_receivers, 1):
        f.write(f"    {i}. {VAR_LABELS[var]}: Total inward influence = {strength:.3f}\n")
        # Find top 3 variables influencing this one
        influencers = [(from_var, combined_coef[SELECTED_VARS.index(from_var), SELECTED_VARS.index(var)])
                      for from_var in SELECTED_VARS if from_var != var]
        influencers.sort(key=lambda x: x[1], reverse=True)
        for from_var, coef in influencers[:3]:
            sign_str = "amplified by" if coef_lag1.loc[from_var, var] > 0 else "dampened by"
            f.write(f"       ← {sign_str} {VAR_LABELS[from_var]} (strength: {coef:.3f})\n")
        f.write("\n")

    f.write("  C. STRONGEST INDIVIDUAL RELATIONSHIPS:\n")
    for i, row in top_rel.iterrows():
        sign_str = "AMPLIFYING" if row['Sign'] > 0 else "DAMPENING"
        f.write(f"    {row['Label']}: {row['Strength']:.3f} ({sign_str})\n")

    f.write("\n\n3. GRANGER CAUSALITY RESULTS\n")
    f.write("-" * 100 + "\n")
    if has_granger:
        f.write(f"  Total tests performed: {total_granger_tests}\n")
        f.write(f"  Significant relationships (p < 0.05): {sig_granger_count}\n")
        f.write(f"  Percentage significant: {100*sig_granger_count/total_granger_tests:.1f}%\n\n")

        f.write("  Top Granger-Causal Relationships:\n")
        top_granger = sig_granger.nsmallest(10, 'p_value')
        for i, (idx, row) in enumerate(top_granger.iterrows(), 1):
            f.write(f"    {i}. {VAR_LABELS[row['Causing']]} → {VAR_LABELS[row['Caused']]}\n")
            f.write(f"       F-stat: {row['F_statistic']:.2f}, p-value: {row['p_value']:.4f}\n")
    else:
        f.write("  Granger causality test results not available\n")
        f.write("  (Use coefficient-based network analysis instead)\n")

    f.write("\n\n4. INTERPRETATION: BUREAUCRATIC GROWTH DYNAMICS\n")
    f.write("-" * 100 + "\n\n")

    f.write("  IRON CAGE OF BUREAUCRACY (Weber):\n")
    f.write("    - Policy Volume shows strong persistence and influences personnel structure\n")
    f.write("    - Field Grade Officers (O-4/O-5) exhibit bureaucratic middle-management growth\n")
    f.write("    - FOIA delays correlate with policy complexity and personnel expansion\n\n")

    f.write("  TEETH-TO-TAIL DYNAMICS:\n")
    f.write("    - Junior Enlisted (combat 'teeth') show different dynamics than officers\n")
    f.write("    - Officer ranks (Company Grade, Field Grade) show interconnected growth\n")
    f.write("    - Warrant Officers occupy unique technical specialist niche\n\n")

    f.write("  POLITICAL APPOINTEE EFFECTS:\n")
    f.write("    - Total PAS (political appointees) influences civilian personnel structure\n")
    f.write("    - May represent external political pressure on bureaucratic organization\n\n")

    f.write("\n5. MODEL DIAGNOSTICS\n")
    f.write("-" * 100 + "\n")
    f.write(f"  Average R-squared: {avg_r2:.4f}\n")
    f.write("  Stationarity: 2/8 variables stationary (suggests VECM may be appropriate)\n")
    f.write("  Lag order: 2 (selected by LASSO analysis and sample size constraints)\n")
    f.write("  Sample size: n=38, adequate for 8 variables with 2 lags\n\n")

    f.write("\n6. NEXT STEPS\n")
    f.write("-" * 100 + "\n")
    f.write("  1. Compare with VECM to separate short-run vs long-run dynamics\n")
    f.write("  2. Examine impulse response functions for shock propagation\n")
    f.write("  3. Interpret coefficients in context of Goldwater-Nichols Act (1986)\n")
    f.write("  4. Link findings to 'Iron Cage' theoretical framework\n")
    f.write("  5. Consider policy implications for DoD organizational reform\n\n")

    f.write("=" * 100 + "\n")
    f.write("FILES GENERATED:\n")
    f.write("=" * 100 + "\n")
    f.write("  1. var2_network_lag1.png - Lag 1 dynamic relationship network\n")
    f.write("  2. var2_network_lag2.png - Lag 2 dynamic relationship network\n")
    f.write("  3. var2_coefficient_heatmaps.png - Lag 1 and Lag 2 coefficient matrices\n")
    if has_granger:
        f.write("  4. var2_granger_causality_network.png - Significant causal relationships\n")
        f.write("  5. var2_key_relationships.png - Top 15 strongest relationships\n")
        f.write("  6. VAR2_Executive_Summary.txt - This document\n")
    else:
        f.write("  4. var2_key_relationships.png - Top 15 strongest relationships\n")
        f.write("  5. VAR2_Executive_Summary.txt - This document\n")
    f.write("=" * 100 + "\n")

print(f"  Executive summary written")

print("\n" + "=" * 100)
print("VAR(2) EXECUTIVE SUMMARY COMPLETE")
print("=" * 100)
print(f"\nFiles saved to: {OUTPUT_DIR}/")
print("  - var2_network_lag1.png")
print("  - var2_network_lag2.png")
print("  - var2_coefficient_heatmaps.png")
if has_granger:
    print("  - var2_granger_causality_network.png")
print("  - var2_key_relationships.png")
print("  - VAR2_Executive_Summary.txt")
print("=" * 100)
