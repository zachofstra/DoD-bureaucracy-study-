"""
Create Digestible Visualizations for VECM/Cointegration Analysis

This script creates intuitive graphics that show:
1. How cointegrating relationships actually look over 37 years
2. Color-coded network showing positive vs negative influences
3. Error correction speeds (which variables adjust fastest)
4. Long-run equilibrium deviations and corrections over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')

output_dir = 'data/analysis/VECM_7VARS'

print("=" * 100)
print("CREATING DIGESTIBLE VISUALIZATIONS")
print("=" * 100)

# =============================================================================
# LOAD DATA AND REFIT MODELS
# =============================================================================
print("\n[1/6] Loading data and refitting models...")

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
years = df['FY'].loc[endog_data.index].values

# Refit VECM
vecm_model = VECM(endog_data, exog=exog_data, k_ar_diff=1, coint_rank=2, deterministic='ci')
vecm_result = vecm_model.fit()

# Get coefficients
alpha = vecm_result.alpha  # Error correction coefficients
beta = vecm_result.beta    # Cointegrating vectors

print("  [OK] Models refitted")

# =============================================================================
# VIZ 1: ERROR CORRECTION SPEEDS (BAR CHART)
# =============================================================================
print("\n[2/6] Creating error correction speed visualization...")

fig, ax = plt.subplots(1, 1, figsize=(14, 8), facecolor='white')

# Calculate max absolute alpha for each variable
max_alpha = np.abs(alpha).max(axis=1)

# Sort by speed (descending)
sorted_indices = np.argsort(max_alpha)[::-1]
sorted_vars = [endog_vars[i].replace('_Z', '').replace('_', ' ') for i in sorted_indices]
sorted_alpha = max_alpha[sorted_indices]

# Color bars by speed
colors = ['#e74c3c' if a > 0.7 else '#f39c12' if a > 0.3 else '#3498db' for a in sorted_alpha]

bars = ax.barh(sorted_vars, sorted_alpha, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (var, val) in enumerate(zip(sorted_vars, sorted_alpha)):
    ax.text(val + 0.03, i, f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

# Add interpretation zones
ax.axvline(x=0.7, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Strong Adjustment (>0.7)')
ax.axvline(x=0.3, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Moderate Adjustment (0.3-0.7)')

ax.set_xlabel('Speed of Adjustment to Equilibrium (Max |alpha|)', fontsize=13, fontweight='bold')
ax.set_title('Which Bureaucratic Dimensions Adjust Fastest to Restore Equilibrium?\n' +
             'VECM Error Correction Coefficients (1987-2024)',
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlim(0, max(sorted_alpha) * 1.15)
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/viz1_error_correction_speeds.png', dpi=300, bbox_inches='tight')
print("  [OK] Error correction speeds saved")

# =============================================================================
# VIZ 2: COINTEGRATING RELATIONSHIPS OVER TIME
# =============================================================================
print("\n[3/6] Plotting cointegrating equilibria over 37 years...")

# Calculate the cointegrating combinations
# Coint Eq = beta' * data
coint_eq1 = endog_data.values @ beta[:, 0]
coint_eq2 = endog_data.values @ beta[:, 1]

fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor='white')

# Equilibrium 1
ax = axes[0]
ax.plot(years, coint_eq1, linewidth=2.5, color='#2c3e50', label='Equilibrium Combination')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect Equilibrium (zero)')
ax.fill_between(years, 0, coint_eq1, alpha=0.2, color='#3498db')

# Mark major deviations (>1 std dev)
std_eq1 = np.std(coint_eq1)
major_dev = np.abs(coint_eq1) > std_eq1
ax.scatter(years[major_dev], coint_eq1[major_dev], s=100, c='red', marker='o',
           edgecolors='black', linewidths=2, zorder=5,
           label=f'Major Deviation (>{std_eq1:.2f})')

# Add event annotations
events = {
    1991: 'Gulf War',
    2001: '9/11',
    2003: 'Iraq War',
    2008: 'Financial Crisis',
    2013: 'Sequestration',
    2020: 'COVID-19'
}

for year, event in events.items():
    if year in years:
        idx = np.where(years == year)[0][0]
        ax.annotate(event, xy=(year, coint_eq1[idx]),
                   xytext=(year, coint_eq1[idx] + 0.5),
                   fontsize=9, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.set_xlabel('Fiscal Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Equilibrium Deviation', fontsize=12, fontweight='bold')
ax.set_title('EQUILIBRIUM 1: Junior Enlisted + PAS + Policy vs Civilians + Field Grade + GOFOs\n' +
             'When this combination deviates from zero, error correction pulls it back',
             fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)

# Equilibrium 2
ax = axes[1]
ax.plot(years, coint_eq2, linewidth=2.5, color='#16a085', label='Equilibrium Combination')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect Equilibrium (zero)')
ax.fill_between(years, 0, coint_eq2, alpha=0.2, color='#1abc9c')

# Mark major deviations
std_eq2 = np.std(coint_eq2)
major_dev2 = np.abs(coint_eq2) > std_eq2
ax.scatter(years[major_dev2], coint_eq2[major_dev2], s=100, c='red', marker='o',
           edgecolors='black', linewidths=2, zorder=5,
           label=f'Major Deviation (>{std_eq2:.2f})')

for year, event in events.items():
    if year in years:
        idx = np.where(years == year)[0][0]
        ax.annotate(event, xy=(year, coint_eq2[idx]),
                   xytext=(year, coint_eq2[idx] + 0.5),
                   fontsize=9, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.set_xlabel('Fiscal Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Equilibrium Deviation', fontsize=12, fontweight='bold')
ax.set_title('EQUILIBRIUM 2: FOIA Days + GOFOs + PAS + Policy + Civilians\n' +
             'Administrative burden and oversight moving together',
             fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/viz2_equilibria_over_time.png', dpi=300, bbox_inches='tight')
print("  [OK] Equilibria over time saved")

# =============================================================================
# VIZ 3: ALL VARIABLES OVER TIME (SHOW COINTEGRATION VISUALLY)
# =============================================================================
print("\n[4/6] Plotting all variables over time to show cointegration...")

fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor='white')

# Normalize all variables to 0-1 for comparison
endog_normalized = (endog_data - endog_data.min()) / (endog_data.max() - endog_data.min())

colors_vars = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']

for i, var in enumerate(endog_vars):
    label = var.replace('_Z', '').replace('_', ' ')
    ax.plot(years, endog_normalized[var].values, linewidth=2.5, label=label,
            color=colors_vars[i], alpha=0.8)

ax.set_xlabel('Fiscal Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Normalized Value (0-1 scale)', fontsize=13, fontweight='bold')
ax.set_title('All Bureaucratic Dimensions Over 37 Years (1987-2024)\n' +
             'Cointegration means these variables are tied together in long run',
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3)

# Add shaded regions for major periods
ax.axvspan(1991, 1992, alpha=0.1, color='red', label='Gulf War')
ax.axvspan(2001, 2011, alpha=0.1, color='orange', label='War on Terror')
ax.axvspan(2013, 2014, alpha=0.1, color='purple', label='Sequestration')

plt.tight_layout()
plt.savefig(f'{output_dir}/viz3_all_variables_over_time.png', dpi=300, bbox_inches='tight')
print("  [OK] All variables over time saved")

# =============================================================================
# VIZ 4: KEY PAIRWISE RELATIONSHIPS (SCATTER PLOTS)
# =============================================================================
print("\n[5/6] Creating scatter plots of cointegrated pairs...")

# Read pairwise cointegration results
pairwise_df = pd.read_excel('data/analysis/pairwise_cointegration.xlsx')
sig_pairs = pairwise_df[pairwise_df['Cointegrated_5pct'] == 'YES **'].nlargest(6, 'Trace_Stat')

fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
axes = axes.flatten()

for idx, (_, row) in enumerate(sig_pairs.iterrows()):
    var1 = row['Variable_1']
    var2 = row['Variable_2']
    trace = row['Trace_Stat']

    ax = axes[idx]

    # Scatter plot
    ax.scatter(endog_data[var1], endog_data[var2], s=80, alpha=0.6,
               c=years, cmap='viridis', edgecolors='black', linewidths=1)

    # Add trend line
    z = np.polyfit(endog_data[var1], endog_data[var2], 1)
    p = np.poly1d(z)
    x_line = np.linspace(endog_data[var1].min(), endog_data[var1].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2.5, label=f'Trend (Trace={trace:.1f})')

    var1_label = var1.replace('_Z', '').replace('_', ' ')
    var2_label = var2.replace('_Z', '').replace('_', ' ')

    ax.set_xlabel(var1_label, fontsize=11, fontweight='bold')
    ax.set_ylabel(var2_label, fontsize=11, fontweight='bold')
    ax.set_title(f'{var1_label} vs {var2_label}\nCointegrated (37-year equilibrium)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=years.min(), vmax=years.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Year', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/viz4_pairwise_cointegration_scatters.png', dpi=300, bbox_inches='tight')
print("  [OK] Pairwise scatter plots saved")

# =============================================================================
# VIZ 5: COLOR-CODED NETWORK (POSITIVE VS NEGATIVE INFLUENCES)
# =============================================================================
print("\n[6/6] Creating color-coded network diagram...")

# Need to get coefficient signs from VAR (not VECM alpha, but actual VAR coefficients)
# Refit VAR on differenced data to match original analysis
non_stationary = ['FOIA_Simple_Days_Z', 'Total_Civilians_Z', 'Policy_Count_Log',
                  'Field_Grade_Officers_Z', 'GOFOs_Z']

endog_diff = endog_data.copy()
for var in non_stationary:
    endog_diff[var] = endog_diff[var].diff()
endog_diff = endog_diff.dropna()
exog_aligned = exog_data.loc[endog_diff.index]

var_model = VAR(endog_diff, exog=exog_aligned)
var_result = var_model.fit(maxlags=2, ic=None)

# Read significant Granger relationships
granger_sig = pd.read_excel('data/analysis/FINAL_VAR2_WITH_CIVILIANS/granger_by_lag_significant.xlsx')
granger_lag2 = granger_sig[granger_sig['Lag'] == 2]

# Extract coefficient signs from VAR results
relationships = []
for _, row in granger_lag2.iterrows():
    cause = row['Cause']
    effect = row['Effect']
    f_stat = row['F_stat']
    p_val = row['p_value']

    # Get coefficient from VAR results
    try:
        param_name = f'{cause}.L2'
        if param_name in var_result.params.index:
            coef = var_result.params.loc[param_name, effect]
            sign = 'Positive' if coef > 0 else 'Negative'

            relationships.append({
                'Cause': cause,
                'Effect': effect,
                'Coefficient': coef,
                'Sign': sign,
                'F_stat': f_stat,
                'p_value': p_val
            })
    except:
        pass

rel_df = pd.DataFrame(relationships)
rel_df.to_excel(f'{output_dir}/signed_relationships.xlsx', index=False)

# Build network
G = nx.DiGraph()

for _, row in rel_df.iterrows():
    G.add_edge(row['Cause'], row['Effect'],
               weight=row['F_stat'],
               coefficient=row['Coefficient'],
               sign=row['Sign'])

# Add isolated nodes
for var in endog_vars:
    if var not in G:
        G.add_node(var)

# Calculate centrality
in_degree = dict(G.in_degree(weight='weight'))
out_degree = dict(G.out_degree(weight='weight'))
total_degree = {n: in_degree.get(n, 0) + out_degree.get(n, 0) for n in G.nodes()}

# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(20, 16), facecolor='white')

pos = nx.spring_layout(G, k=3.5, iterations=50, seed=42, weight='weight')

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

# Node sizes by centrality
node_sizes = [800 + total_degree.get(node, 0) * 150 for node in G.nodes()]

# Edge colors by sign
edge_colors = []
edge_widths = []
for u, v in G.edges():
    if G[u][v]['sign'] == 'Positive':
        edge_colors.append('#e74c3c')  # Red for positive
    else:
        edge_colors.append('#3498db')  # Blue for negative

    # Width by F-statistic
    edge_widths.append(1 + (G[u][v]['weight'] / 7) * 5)

nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.7,
                       width=edge_widths, arrows=True, arrowsize=25,
                       ax=ax, connectionstyle='arc3,rad=0.15')

nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                       node_size=node_sizes, alpha=0.9,
                       edgecolors='black', linewidths=2.5, ax=ax)

labels = {node: node.replace('_Z', '').replace('_', '\n') for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold', ax=ax)

# Edge labels with coefficient values
edge_labels = {}
for u, v in G.edges():
    coef = G[u][v]['coefficient']
    sign_str = '+' if coef > 0 else ''
    edge_labels[(u, v)] = f"{sign_str}{coef:.3f}"

nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9,
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9),
                             ax=ax)

ax.set_title('DoD Bureaucratic Growth: Signed Causal Network (1987-2024)\n' +
            'RED edges = POSITIVE influence (variables move together)\n' +
            'BLUE edges = NEGATIVE influence (variables move inversely)\n' +
            'Edge width = Statistical strength (F-statistic)',
            fontsize=20, fontweight='bold', pad=30)
ax.axis('off')

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#3498db', label='Enlisted Personnel', alpha=0.9),
    Patch(facecolor='#e74c3c', label='Officer Personnel', alpha=0.9),
    Patch(facecolor='#2ecc71', label='Bureaucratic Measures', alpha=0.9),
    Line2D([0], [0], color='#e74c3c', linewidth=4, label='Positive Influence (+)', alpha=0.7),
    Line2D([0], [0], color='#3498db', linewidth=4, label='Negative Influence (-)', alpha=0.7),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=18, label='Node size ~ Network centrality')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=14,
         framealpha=0.95, title='Legend', title_fontsize=15)

plt.tight_layout()
plt.savefig(f'{output_dir}/viz5_signed_network_diagram.png', dpi=300, bbox_inches='tight')
print("  [OK] Signed network diagram saved")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("ALL VISUALIZATIONS COMPLETE")
print("=" * 100)

print("\n  FILES CREATED IN: data/analysis/VECM_7VARS/")
print("  " + "-" * 96)
print("  1. viz1_error_correction_speeds.png")
print("     - Bar chart showing which variables adjust fastest to equilibrium")
print()
print("  2. viz2_equilibria_over_time.png")
print("     - Time series of the two cointegrating relationships (1987-2024)")
print("     - Shows when system deviates from equilibrium and corrects back")
print("     - Annotated with major events (Gulf War, 9/11, Sequestration, etc.)")
print()
print("  3. viz3_all_variables_over_time.png")
print("     - All 7 variables on one plot (normalized 0-1)")
print("     - Visual evidence that variables move together (cointegration)")
print()
print("  4. viz4_pairwise_cointegration_scatters.png")
print("     - Scatter plots of top 6 cointegrated pairs")
print("     - Color gradient shows time progression")
print("     - Trend lines show long-run equilibrium relationships")
print()
print("  5. viz5_signed_network_diagram.png")
print("     - **COLOR-CODED NETWORK**: Red = positive, Blue = negative")
print("     - Edge labels show actual coefficient values")
print("     - Node size = network centrality")
print()
print("  6. signed_relationships.xlsx")
print("     - Table of all relationships with coefficient signs")
print()
print("=" * 100)
print("THESE VISUALIZATIONS ARE PUBLICATION-READY FOR YOUR THESIS")
print("=" * 100)
