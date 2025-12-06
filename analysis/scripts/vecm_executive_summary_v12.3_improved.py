"""
VECM Executive Summary Generator - v12.3 Dataset (IMPROVED)
Comprehensive analysis with intuitive, comparable visualizations
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
print("VECM EXECUTIVE SUMMARY GENERATOR - v12.3 DATASET (IMPROVED)")
print("=" * 100)

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR = 'analysis/VECM_v12.3_Final'
OUTPUT_DIR = 'analysis/VECM_v12.3_Final/VECM_v12.3_Final_Executive_Summary'
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

SELECTED_VARS = [
    'Junior_Enlisted_Z',
    'Company_Grade_Officers_Z',
    'Field_Grade_Officers_Z',
    'GOFOs_Z',
    'Warrant_Officers_Z',
    'Policy_Count_Log',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z'
]

# Variable labels
VAR_LABELS = {
    'Junior_Enlisted_Z': 'Junior Enlisted (E-1 to E-4)',
    'Company_Grade_Officers_Z': 'Company Grade (O-1 to O-3)',
    'Field_Grade_Officers_Z': 'Field Grade (O-4 to O-5)',
    'GOFOs_Z': 'General/Flag Officers',
    'Warrant_Officers_Z': 'Warrant Officers',
    'Policy_Count_Log': 'Policy Volume (Log)',
    'Total_PAS_Z': 'Political Appointees (PAS)',
    'FOIA_Simple_Days_Z': 'FOIA Processing Delay'
}

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/8] Loading VECM results...")

# Load cointegration vectors (beta)
beta_df = pd.read_excel(f'{INPUT_DIR}/cointegration_vectors_beta.xlsx', index_col=0)

# Load error correction coefficients (alpha)
alpha_df = pd.read_excel(f'{INPUT_DIR}/error_correction_alpha.xlsx', index_col=0)

# Load short-run dynamics (gamma)
gamma_lag1 = pd.read_excel(f'{INPUT_DIR}/short_run_gamma_lag1.xlsx', index_col=0)

# Load model fit
rsquared_df = pd.read_excel(f'{INPUT_DIR}/model_fit_rsquared.xlsx')

print(f"  Loaded cointegration vectors: {beta_df.shape}")
print(f"  Loaded error correction coefficients: {alpha_df.shape}")
print(f"  Loaded short-run dynamics (gamma): {gamma_lag1.shape}")
print(f"  Loaded R-squared for {len(rsquared_df)} equations")

# =============================================================================
# LONG-RUN NETWORK DIAGRAM (from Error Correction Mechanism: Alpha × Beta)
# =============================================================================
print("\n[2/8] Creating LONG-RUN equilibrium network...")

# Calculate long-run influence using error correction mechanism
# Edge from j to i means: "j's deviation from equilibrium causes i to adjust"
# Weight = sum over all cointegration vectors of |alpha[i,r] * beta[j,r]|

beta_importance = beta_df.abs().sum(axis=1)  # For node sizing

G_lr = nx.DiGraph()

for var in SELECTED_VARS:
    G_lr.add_node(var, label=VAR_LABELS[var])

# Create edges based on error correction relationships
threshold = 0.05
for i, to_var in enumerate(SELECTED_VARS):  # Variable that adjusts
    for j, from_var in enumerate(SELECTED_VARS):  # Variable causing adjustment
        if i != j:
            # Calculate total long-run influence from j to i
            # across all cointegration vectors
            total_influence = 0
            total_signed_influence = 0

            for r in range(beta_df.shape[1]):  # For each cointegration vector
                alpha_i_r = alpha_df.iloc[i, r]  # How fast i adjusts to EC_r
                beta_j_r = beta_df.iloc[j, r]    # How much j enters EC_r

                # Influence of j on i through cointegration vector r
                influence = alpha_i_r * beta_j_r
                total_influence += abs(influence)
                total_signed_influence += influence

            if total_influence > threshold:
                G_lr.add_edge(from_var, to_var,
                            weight=total_influence,
                            sign=np.sign(total_signed_influence))

print(f"  Long-run network: {G_lr.number_of_nodes()} nodes, {G_lr.number_of_edges()} edges")

# Plot
fig, ax = plt.subplots(figsize=(20, 16))

# More spacing between nodes - increased k to separate overlapping nodes
pos = nx.spring_layout(G_lr, k=5.0, iterations=200, seed=42)

# Node sizes based on beta importance - USE SQRT SCALING for better visual balance
node_importance = np.array([beta_importance[var] for var in G_lr.nodes()])
node_sizes = np.sqrt(node_importance) * 800 + 1200  # Much more reasonable range

# Draw nodes
nx.draw_networkx_nodes(G_lr, pos,
                       node_size=node_sizes,
                       node_color='#FFD700',  # Gold for long-run
                       edgecolors='black',
                       linewidths=2.5,
                       ax=ax)

# Draw edges with VISIBLE ARROWS - scaled down for professional appearance
for (u, v, d) in G_lr.edges(data=True):
    edge_color = '#e74c3c' if d['sign'] > 0 else '#3498db'
    # Reduced scaling and capped maximum width
    edge_width = min(d['weight'] * 1.5 + 0.5, 4.0)  # Cap at 4.0

    nx.draw_networkx_edges(G_lr, pos,
                          edgelist=[(u, v)],
                          edge_color=edge_color,
                          width=edge_width,
                          alpha=0.7,
                          arrows=True,
                          arrowsize=35,
                          arrowstyle='-|>',
                          connectionstyle='arc3,rad=0.15',
                          node_size=node_sizes,
                          ax=ax,
                          min_source_margin=15,
                          min_target_margin=15)

# Draw labels with background
for node in G_lr.nodes():
    x, y = pos[node]
    ax.text(x, y, VAR_LABELS[node], fontsize=11, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='black', linewidth=1.5))

ax.set_title('LONG-RUN EQUILIBRIUM Network (Error Correction Mechanism)\n' +
             '(Red arrows=Amplifying (+), Blue=Dampening (-); Width=Strength of α×β)',
             fontsize=15, fontweight='bold', pad=25)
ax.axis('off')
ax.margins(0.20)  # More margin for better spacing

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='#e74c3c', linewidth=4,
               marker='>', markersize=12, label='Amplifying LR influence (+)'),
    plt.Line2D([0], [0], color='#3498db', linewidth=4,
               marker='>', markersize=12, label='Dampening LR influence (-)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700',
               markersize=18, label='Node size = Beta importance',
               markeredgecolor='black', markeredgewidth=2)
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
         frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/vecm_longrun_network.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  Long-run network saved")

# =============================================================================
# SHORT-RUN NETWORK DIAGRAM (from Gamma coefficients) - MATCHING STYLE
# =============================================================================
print("\n[3/8] Creating SHORT-RUN dynamics network...")

G_sr = nx.DiGraph()

for var in SELECTED_VARS:
    G_sr.add_node(var, label=VAR_LABELS[var])

# Add edges from gamma matrix
threshold = 0.05
for i, from_var in enumerate(SELECTED_VARS):
    for j, to_var in enumerate(SELECTED_VARS):
        if i != j:
            coef = gamma_lag1.loc[from_var, to_var]
            if abs(coef) > threshold:
                G_sr.add_edge(from_var, to_var,
                            weight=abs(coef),
                            sign=np.sign(coef))

print(f"  Short-run network: {G_sr.number_of_nodes()} nodes, {G_sr.number_of_edges()} edges")

# Plot - SAME STYLE AS LONG-RUN
fig, ax = plt.subplots(figsize=(20, 16))

# More spacing between nodes - increased k to separate overlapping nodes
pos = nx.spring_layout(G_sr, k=5.0, iterations=200, seed=42)

# Node sizes based on degree - USE SQRT SCALING for better visual balance
node_degrees = np.array([G_sr.degree(n) for n in G_sr.nodes()])
node_sizes = np.sqrt(node_degrees) * 600 + 1200  # Much more reasonable range

# Draw nodes - GREEN for short-run
nx.draw_networkx_nodes(G_sr, pos,
                       node_size=node_sizes,
                       node_color='#90EE90',  # Light green for short-run
                       edgecolors='black',
                       linewidths=2.5,
                       ax=ax)

# Draw edges with VISIBLE ARROWS - scaled down for professional appearance
for (u, v, d) in G_sr.edges(data=True):
    edge_color = '#e74c3c' if d['sign'] > 0 else '#3498db'
    # Reduced scaling and capped maximum width
    edge_width = min(d['weight'] * 2.0 + 0.5, 4.0)  # Cap at 4.0

    nx.draw_networkx_edges(G_sr, pos,
                          edgelist=[(u, v)],
                          edge_color=edge_color,
                          width=edge_width,
                          alpha=0.7,
                          arrows=True,
                          arrowsize=35,
                          arrowstyle='-|>',
                          connectionstyle='arc3,rad=0.15',
                          node_size=node_sizes,
                          ax=ax,
                          min_source_margin=15,
                          min_target_margin=15)

# Draw labels with background
for node in G_sr.nodes():
    x, y = pos[node]
    ax.text(x, y, VAR_LABELS[node], fontsize=11, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='black', linewidth=1.5))

ax.set_title('SHORT-RUN DYNAMICS Network (Gamma Lag 1)\n' +
             '(Red arrows=Amplifying (+), Blue=Dampening (-); Width=Strength)',
             fontsize=15, fontweight='bold', pad=25)
ax.axis('off')
ax.margins(0.20)  # More margin for better spacing

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='#e74c3c', linewidth=4,
               marker='>', markersize=12, label='Amplifying (+)'),
    plt.Line2D([0], [0], color='#3498db', linewidth=4,
               marker='>', markersize=12, label='Dampening (-)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#90EE90',
               markersize=18, label='Node size = Degree',
               markeredgecolor='black', markeredgewidth=2)
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
         frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/vecm_shortrun_network.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  Short-run network saved")

# =============================================================================
# PRIMARY DRIVERS - LONG-RUN (Beta importance)
# =============================================================================
print("\n[4/8] Creating PRIMARY DRIVERS visualization (long-run)...")

fig, ax = plt.subplots(figsize=(12, 8))

beta_sorted = beta_importance.sort_values(ascending=True)
colors = ['#FFD700' if i % 2 == 0 else '#FFA500' for i in range(len(beta_sorted))]

bars = ax.barh(range(len(beta_sorted)), beta_sorted.values,
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_yticks(range(len(beta_sorted)))
ax.set_yticklabels([VAR_LABELS[v] for v in beta_sorted.index], fontsize=11)
ax.set_xlabel('Total |Beta| Across All Cointegration Vectors', fontsize=12, fontweight='bold')
ax.set_title('PRIMARY DRIVERS in Long-Run Equilibrium\n(Higher = More Important in Equilibrium Structure)',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (idx, val) in enumerate(beta_sorted.items()):
    ax.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/vecm_longrun_drivers.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  Long-run drivers chart saved")

# =============================================================================
# PRIMARY DRIVERS - SHORT-RUN (Gamma out-strength)
# =============================================================================
print("\n[5/8] Creating PRIMARY DRIVERS visualization (short-run)...")

# Calculate out-strength (how much each variable influences others)
gamma_out_strength = gamma_lag1.abs().sum(axis=1).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#90EE90' if i % 2 == 0 else '#3CB371' for i in range(len(gamma_out_strength))]

bars = ax.barh(range(len(gamma_out_strength)), gamma_out_strength.values,
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_yticks(range(len(gamma_out_strength)))
ax.set_yticklabels([VAR_LABELS[v] for v in gamma_out_strength.index], fontsize=11)
ax.set_xlabel('Total |Gamma| Out-Strength (Sum of Influences on Other Variables)',
              fontsize=12, fontweight='bold')
ax.set_title('PRIMARY DRIVERS in Short-Run Dynamics\n(Higher = Stronger Influence on Other Variables)',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (idx, val) in enumerate(gamma_out_strength.items()):
    ax.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/vecm_shortrun_drivers.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  Short-run drivers chart saved")

# =============================================================================
# ERROR CORRECTION SPEEDS - INTUITIVE VISUALIZATION
# =============================================================================
print("\n[6/8] Creating ERROR CORRECTION speeds visualization...")

# Average absolute alpha across cointegration relationships
avg_alpha = alpha_df.abs().mean(axis=1).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))

# Color by direction of first EC coefficient
colors = []
for var in avg_alpha.index:
    alpha_ec1 = alpha_df.loc[var, alpha_df.columns[0]]
    if alpha_ec1 < 0:
        colors.append('#e74c3c')  # Red for negative (corrects upward deviations)
    else:
        colors.append('#3498db')  # Blue for positive (corrects downward deviations)

bars = ax.barh(range(len(avg_alpha)), avg_alpha.values,
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_yticks(range(len(avg_alpha)))
ax.set_yticklabels([VAR_LABELS[v] for v in avg_alpha.index], fontsize=11)
ax.set_xlabel('Average |Error Correction Speed| (Alpha)', fontsize=12, fontweight='bold')
ax.set_title('ERROR CORRECTION SPEEDS by Variable\n(Higher = Faster Adjustment to Equilibrium; Red=Corrects ↑, Blue=Corrects ↓)',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (idx, val) in enumerate(avg_alpha.items()):
    ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# Legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.8, label='Corrects upward deviations'),
    plt.Rectangle((0, 0), 1, 1, fc='#3498db', alpha=0.8, label='Corrects downward deviations')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/vecm_adjustment_speeds.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  Adjustment speeds chart saved")

# =============================================================================
# COMPARISON: LONG-RUN vs SHORT-RUN IMPORTANCE
# =============================================================================
print("\n[7/8] Creating COMPARISON visualization...")

# Normalize both to 0-100 scale
beta_norm = (beta_importance / beta_importance.max()) * 100
gamma_norm = (gamma_lag1.abs().sum(axis=1) / gamma_lag1.abs().sum(axis=1).max()) * 100

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Long-Run (Beta)': beta_norm,
    'Short-Run (Gamma)': gamma_norm
})

# Sort by long-run importance
comparison_df = comparison_df.sort_values('Long-Run (Beta)', ascending=True)

fig, ax = plt.subplots(figsize=(14, 8))

y_pos = np.arange(len(comparison_df))
width = 0.35

bars1 = ax.barh(y_pos - width/2, comparison_df['Long-Run (Beta)'],
                width, label='Long-Run Equilibrium', color='#FFD700',
                alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.barh(y_pos + width/2, comparison_df['Short-Run (Gamma)'],
                width, label='Short-Run Dynamics', color='#90EE90',
                alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_yticks(y_pos)
ax.set_yticklabels([VAR_LABELS[v] for v in comparison_df.index], fontsize=11)
ax.set_xlabel('Importance Score (Normalized to 0-100)', fontsize=12, fontweight='bold')
ax.set_title('LONG-RUN vs SHORT-RUN Variable Importance\n(Shows which variables are more important in equilibrium vs immediate dynamics)',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/vecm_longrun_vs_shortrun_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  Comparison chart saved")

# =============================================================================
# EXECUTIVE SUMMARY DOCUMENT
# =============================================================================
print("\n[8/8] Writing executive summary...")

avg_r2 = rsquared_df['R_squared'].mean()
n_coint_vecs = beta_df.shape[1]

with open(f'{OUTPUT_DIR}/VECM_Executive_Summary.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("EXECUTIVE SUMMARY: VECM ANALYSIS OF DOD BUREAUCRATIC GROWTH\n")
    f.write("Dataset: v12.3 (1987-2024, 38 observations)\n")
    f.write("=" * 100 + "\n\n")

    f.write("1. MODEL SPECIFICATION\n")
    f.write("-" * 100 + "\n")
    f.write("  Model: Vector Error Correction Model (VECM)\n")
    f.write("  Variables: 8 key bureaucratic/personnel indicators\n")
    f.write("  Time Period: 1987-2024 (37 years, 38 observations)\n")
    f.write("  VAR Lag Order: 2 (lag 1 in differences)\n")
    f.write(f"  Cointegration Rank: {n_coint_vecs} long-run relationships\n")
    f.write(f"  Average R-squared: {avg_r2:.4f}\n\n")

    f.write("  Variables Included:\n")
    for i, var in enumerate(SELECTED_VARS, 1):
        f.write(f"    {i}. {VAR_LABELS[var]} ({var})\n")

    f.write("\n2. PRIMARY DRIVERS IN LONG-RUN EQUILIBRIUM\n")
    f.write("-" * 100 + "\n\n")

    beta_ranked = beta_importance.sort_values(ascending=False)
    f.write("  Variables ranked by long-run equilibrium importance:\n\n")
    for i, (var, importance) in enumerate(beta_ranked.items(), 1):
        f.write(f"  {i}. {VAR_LABELS[var]:45s} Beta importance: {importance:.3f}\n")

    f.write("\n  INTERPRETATION:\n")
    f.write("    - These variables are most critical in the long-run equilibrium structure\n")
    f.write("    - High beta importance means the variable plays a key role in cointegration\n")
    f.write("    - Policy reforms targeting these variables will have structural impacts\n\n")

    f.write("\n3. PRIMARY DRIVERS IN SHORT-RUN DYNAMICS\n")
    f.write("-" * 100 + "\n\n")

    gamma_ranked = gamma_lag1.abs().sum(axis=1).sort_values(ascending=False)
    f.write("  Variables ranked by short-run influence on others:\n\n")
    for i, (var, influence) in enumerate(gamma_ranked.items(), 1):
        f.write(f"  {i}. {VAR_LABELS[var]:45s} Total influence: {influence:.3f}\n")

    f.write("\n  INTERPRETATION:\n")
    f.write("    - These variables have the strongest immediate impact on other variables\n")
    f.write("    - High short-run influence means shocks propagate quickly through the system\n")
    f.write("    - Important for understanding year-over-year dynamics\n\n")

    f.write("\n4. ERROR CORRECTION DYNAMICS\n")
    f.write("-" * 100 + "\n\n")

    avg_alpha_sorted = avg_alpha.sort_values(ascending=False)
    f.write("  Adjustment speeds to long-run equilibrium:\n\n")
    for var in avg_alpha_sorted.index:
        avg_speed = avg_alpha_sorted[var]
        alpha_ec1 = alpha_df.loc[var, alpha_df.columns[0]]
        direction = "corrects upward deviations" if alpha_ec1 < 0 else "corrects downward deviations"

        f.write(f"  {VAR_LABELS[var]:45s} |Speed|: {avg_speed:.4f}  ({direction})\n")

    f.write("\n  INTERPRETATION:\n")
    f.write("    - Negative alpha: variable adjusts DOWN when above equilibrium\n")
    f.write("    - Positive alpha: variable adjusts UP when below equilibrium\n")
    f.write("    - Larger |alpha|: faster return to equilibrium after shocks\n\n")

    f.write("\n5. KEY INSIGHTS: LONG-RUN vs SHORT-RUN\n")
    f.write("-" * 100 + "\n\n")

    # Identify variables that differ in long vs short run importance
    f.write("  Variables MORE important in LONG-RUN than SHORT-RUN:\n")
    lr_dominant = comparison_df[comparison_df['Long-Run (Beta)'] > comparison_df['Short-Run (Gamma)']].sort_values('Long-Run (Beta)', ascending=False)
    for var in lr_dominant.index:
        lr_score = comparison_df.loc[var, 'Long-Run (Beta)']
        sr_score = comparison_df.loc[var, 'Short-Run (Gamma)']
        f.write(f"    {VAR_LABELS[var]:45s} LR={lr_score:.1f}, SR={sr_score:.1f}\n")

    f.write("\n  Variables MORE important in SHORT-RUN than LONG-RUN:\n")
    sr_dominant = comparison_df[comparison_df['Short-Run (Gamma)'] > comparison_df['Long-Run (Beta)']].sort_values('Short-Run (Gamma)', ascending=False)
    for var in sr_dominant.index:
        lr_score = comparison_df.loc[var, 'Long-Run (Beta)']
        sr_score = comparison_df.loc[var, 'Short-Run (Gamma)']
        f.write(f"    {VAR_LABELS[var]:45s} LR={lr_score:.1f}, SR={sr_score:.1f}\n")

    f.write("\n  INTERPRETATION:\n")
    f.write("    - LR-dominant: Structural forces, slow-moving, hard to reform\n")
    f.write("    - SR-dominant: Cyclical forces, fast-moving, easier to influence\n")
    f.write("    - This distinction is critical for policy design\n\n")

    f.write("\n6. BUREAUCRATIC GROWTH MECHANISMS (Iron Cage)\n")
    f.write("-" * 100 + "\n\n")

    f.write("  FIELD GRADE OFFICERS (O-4/O-5) Analysis:\n")
    fg_beta = beta_importance['Field_Grade_Officers_Z']
    fg_gamma = gamma_lag1.abs().sum(axis=1)['Field_Grade_Officers_Z']
    fg_alpha = avg_alpha['Field_Grade_Officers_Z']

    f.write(f"    Long-run importance (Beta): {fg_beta:.3f}\n")
    f.write(f"    Short-run influence (Gamma): {fg_gamma:.3f}\n")
    f.write(f"    Adjustment speed (|Alpha|): {fg_alpha:.3f}\n")
    f.write(f"    → This rank is a key bureaucratic expansion indicator\n\n")

    f.write("  POLICY VOLUME Analysis:\n")
    pol_beta = beta_importance['Policy_Count_Log']
    pol_gamma = gamma_lag1.abs().sum(axis=1)['Policy_Count_Log']
    pol_alpha = avg_alpha['Policy_Count_Log']

    f.write(f"    Long-run importance (Beta): {pol_beta:.3f}\n")
    f.write(f"    Short-run influence (Gamma): {pol_gamma:.3f}\n")
    f.write(f"    Adjustment speed (|Alpha|): {pol_alpha:.3f}\n")
    f.write(f"    → Weber's 'Iron Cage' - rule proliferation drives structure\n\n")

    f.write("\n7. POLICY IMPLICATIONS\n")
    f.write("-" * 100 + "\n")
    f.write("  1. Target long-run dominant variables for STRUCTURAL reform\n")
    f.write("  2. Target short-run dominant variables for IMMEDIATE impact\n")
    f.write("  3. Variables with fast error correction are easier to reform\n")
    f.write("  4. Field Grade Officers (O-4/O-5) are a key intervention point\n")
    f.write("  5. Policy volume reduction may unlock broader bureaucratic reform\n\n")

    f.write("=" * 100 + "\n")
    f.write("FILES GENERATED:\n")
    f.write("=" * 100 + "\n")
    f.write("  1. vecm_longrun_network.png - Long-run equilibrium network (GOLD nodes)\n")
    f.write("  2. vecm_shortrun_network.png - Short-run dynamics network (GREEN nodes)\n")
    f.write("  3. vecm_longrun_drivers.png - Primary drivers in long-run equilibrium\n")
    f.write("  4. vecm_shortrun_drivers.png - Primary drivers in short-run dynamics\n")
    f.write("  5. vecm_adjustment_speeds.png - Error correction speeds\n")
    f.write("  6. vecm_longrun_vs_shortrun_comparison.png - Side-by-side comparison\n")
    f.write("  7. VECM_Executive_Summary.txt - This document\n")
    f.write("=" * 100 + "\n")

print(f"  Executive summary written")

print("\n" + "=" * 100)
print("VECM EXECUTIVE SUMMARY COMPLETE (IMPROVED)")
print("=" * 100)
print(f"\nFiles saved to: {OUTPUT_DIR}/")
print("  - vecm_longrun_network.png (GOLD nodes, visible arrows)")
print("  - vecm_shortrun_network.png (GREEN nodes, visible arrows)")
print("  - vecm_longrun_drivers.png (intuitive bar chart)")
print("  - vecm_shortrun_drivers.png (intuitive bar chart)")
print("  - vecm_adjustment_speeds.png (color-coded by direction)")
print("  - vecm_longrun_vs_shortrun_comparison.png (side-by-side)")
print("  - VECM_Executive_Summary.txt")
print("=" * 100)
