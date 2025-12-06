"""
Create ALL visualizations for sign-corrected Rank=2 VECM
Matches the structure from VECM_Rank2_Final_Executive_Summary
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.lines import Line2D
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
INPUT_DIR = BASE_DIR / "VECM_Rank2_CORRECTED"
OUTPUT_DIR = INPUT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

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

DISPLAY_NAMES_SHORT = {
    'Junior_Enlisted_Z': 'Junior\nEnlisted',
    'Company_Grade_Officers_Z': 'Company\nGrade',
    'Field_Grade_Officers_Z': 'Field\nGrade',
    'GOFOs_Z': 'GOFOs',
    'Warrant_Officers_Z': 'Warrant\nOfficers',
    'Policy_Count_Log': 'Policy\nCount',
    'Total_PAS_Z': 'Total\nPAS',
    'FOIA_Simple_Days_Z': 'FOIA\nDays'
}

DISPLAY_NAMES_LONG = {
    'Junior_Enlisted_Z': 'Junior Enlisted\n(E-1 to E-4)',
    'Company_Grade_Officers_Z': 'Company Grade\n(O-1 to O-3)',
    'Field_Grade_Officers_Z': 'Field Grade\n(O-4 to O-5)',
    'GOFOs_Z': 'General/Flag\nOfficers',
    'Warrant_Officers_Z': 'Warrant\nOfficers',
    'Policy_Count_Log': 'Policy Volume\n(Log)',
    'Total_PAS_Z': 'Political\nAppointees (PAS)',
    'FOIA_Simple_Days_Z': 'FOIA Processing\nDelay'
}

print("=" * 80)
print("CREATING ALL CORRECTED VISUALIZATIONS (RANK=2)")
print("=" * 80)

# Load corrected matrices
print("\n[1] Loading corrected matrices...")
alpha_df = pd.read_excel(INPUT_DIR / "alpha_matrix_rank2_CORRECTED.xlsx", index_col=0)
beta_df = pd.read_excel(INPUT_DIR / "beta_matrix_rank2_CORRECTED.xlsx", index_col=0)
gamma_df = pd.read_excel(INPUT_DIR / "gamma_matrix_rank2_CORRECTED.xlsx", index_col=0)
longrun_df = pd.read_excel(INPUT_DIR / "longrun_influence_rank2_CORRECTED.xlsx", index_col=0)

print(f"    Alpha: {alpha_df.shape}")
print(f"    Beta: {beta_df.shape}")
print(f"    Gamma: {gamma_df.shape}")
print(f"    Long-run: {longrun_df.shape}")

# Calculate signed direction for long-run
print("\n[2] Calculating signed direction...")
signed_direction = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))
for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        signed_sum = 0
        for r in range(2):
            alpha_i = alpha_df.iloc[i, r]
            beta_j = beta_df.iloc[j, r]
            signed_sum += alpha_i * beta_j
        signed_direction[i, j] = np.sign(signed_sum)

signed_magnitude = longrun_df.values * signed_direction

print("    Signed direction calculated")

# ============================================================================
# VISUALIZATION 1: INFLUENCE COMPARISON HEATMAP
# ============================================================================
print("\n[3] Creating influence comparison heatmap...")

display_names = [DISPLAY_NAMES_SHORT[var] for var in SELECTED_VARS]
gamma_values = gamma_df.values

fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Short-run (LEFT)
sns.heatmap(gamma_values, annot=np.abs(gamma_values), fmt='.2f',
            cmap='RdBu_r', center=0, cbar_kws={'label': 'Coefficient'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=axes[0])
axes[0].set_title('SHORT-RUN DYNAMICS (Gamma)\nYear-to-year VAR effects',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('From Variable (t-1)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('To Variable', fontsize=11, fontweight='bold')

# Long-run (RIGHT) - CORRECTED SIGNS
sns.heatmap(signed_magnitude, annot=longrun_df.values, fmt='.2f',
            cmap='RdBu_r', center=0, cbar_kws={'label': 'Direction (RED=Amplifying, BLUE=Dampening)'},
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.5, linecolor='gray', ax=axes[1])
axes[1].set_title('LONG-RUN INFLUENCE (Error Correction)\nMagnitude with directional coloring (sum across 2 vectors)',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('From Variable (equilibrium deviation)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('To Variable (adjustment)', fontsize=11, fontweight='bold')

fig.suptitle('VECM INFLUENCE COMPARISON: Short-Run vs Long-Run (RANK=2 - CORRECTED SIGNS)\n(Magnitude values with directional coloring)',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_influence_comparison_rank2_CORRECTED.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Influence comparison saved")

# ============================================================================
# VISUALIZATION 2: LONG-RUN VS SHORT-RUN VARIABLE IMPORTANCE
# ============================================================================
print("\n[4] Creating long-run vs short-run comparison...")

# Beta importance (long-run)
beta_importance = np.abs(beta_df).sum(axis=1)
beta_normalized = (beta_importance / beta_importance.max()) * 100

# Gamma importance (short-run)
gamma_importance = np.abs(gamma_df).sum(axis=0) + np.abs(gamma_df).sum(axis=1)
gamma_normalized = (gamma_importance / gamma_importance.max()) * 100

importance_df = pd.DataFrame({
    'Variable': [DISPLAY_NAMES_LONG[var].replace('\n', ' ') for var in SELECTED_VARS],
    'Long_Run': beta_normalized.values,
    'Short_Run': gamma_normalized.values
})

importance_df = importance_df.sort_values('Long_Run', ascending=True)

fig, ax = plt.subplots(figsize=(14, 10))
y_pos = np.arange(len(importance_df))
bar_height = 0.35

bars1 = ax.barh(y_pos, importance_df['Long_Run'], bar_height,
                label='Long-Run Equilibrium', color='#FFD54F',
                edgecolor='black', linewidth=1.5, alpha=0.9)
bars2 = ax.barh(y_pos + bar_height, importance_df['Short_Run'], bar_height,
                label='Short-Run Dynamics', color='#81C784',
                edgecolor='black', linewidth=1.5, alpha=0.9)

ax.set_yticks(y_pos + bar_height / 2)
ax.set_yticklabels(importance_df['Variable'], fontsize=11)
ax.set_xlabel('Importance Score (Normalized to 0-100)', fontsize=12, fontweight='bold')
ax.set_title('LONG-RUN vs SHORT-RUN Variable Importance (CORRECTED SIGNS)\n(Shows which variables are more important in equilibrium vs immediate dynamics)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, 110)

# Add value labels
for i, (lr, sr) in enumerate(zip(importance_df['Long_Run'], importance_df['Short_Run'])):
    if lr > 5:
        ax.text(lr + 1, i, f'{lr:.0f}', va='center', fontsize=9, fontweight='bold')
    if sr > 5:
        ax.text(sr + 1, i + bar_height, f'{sr:.0f}', va='center', fontsize=9, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_longrun_vs_shortrun_comparison_CORRECTED.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Long-run vs short-run comparison saved")

# ============================================================================
# VISUALIZATION 3: NETWORK DIAGRAMS
# ============================================================================
print("\n[5] Creating network diagrams...")

# Long-run network
print("    [5a] Long-run network...")
G_longrun = nx.DiGraph()

beta_importance_vals = np.abs(beta_df).sum(axis=1).values
beta_importance_normalized = (beta_importance_vals / beta_importance_vals.max()) * 3000 + 500

for i, var in enumerate(SELECTED_VARS):
    G_longrun.add_node(var, importance=beta_importance_vals[i], size=beta_importance_normalized[i])

threshold = 0.15
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            magnitude = longrun_df.iloc[i, j]
            if magnitude > threshold:
                direction = np.sign(signed_direction[i, j])
                G_longrun.add_edge(var_from, var_to, weight=magnitude, direction=direction)

pos = nx.circular_layout(G_longrun)
node_sizes = [G_longrun.nodes[node]['size'] for node in G_longrun.nodes()]

# Individual long-run diagram
fig, ax = plt.subplots(figsize=(16, 14))

nx.draw_networkx_nodes(G_longrun, pos, node_size=node_sizes,
                       node_color='yellow', edgecolors='black',
                       linewidths=2.5, alpha=0.9, ax=ax)

edges_amplifying = [(u, v) for u, v, d in G_longrun.edges(data=True) if d['direction'] > 0]
edges_dampening = [(u, v) for u, v, d in G_longrun.edges(data=True) if d['direction'] < 0]

if edges_amplifying or edges_dampening:
    max_weight = max([d['weight'] for u, v, d in G_longrun.edges(data=True)])

    if edges_amplifying:
        weights_amp = [G_longrun[u][v]['weight'] for u, v in edges_amplifying]
        widths_amp = [3 + (w / max_weight) * 5 for w in weights_amp]
        nx.draw_networkx_edges(G_longrun, pos, edgelist=edges_amplifying,
                               width=widths_amp, edge_color='red', alpha=0.7,
                               arrows=True, arrowsize=25, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15',
                               min_source_margin=25, min_target_margin=25, ax=ax)

    if edges_dampening:
        weights_damp = [G_longrun[u][v]['weight'] for u, v in edges_dampening]
        widths_damp = [3 + (w / max_weight) * 5 for w in weights_damp]
        nx.draw_networkx_edges(G_longrun, pos, edgelist=edges_dampening,
                               width=widths_damp, edge_color='blue', alpha=0.7,
                               arrows=True, arrowsize=25, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15',
                               min_source_margin=25, min_target_margin=25, ax=ax)

labels = {node: DISPLAY_NAMES_LONG[node] for node in G_longrun.nodes()}
nx.draw_networkx_labels(G_longrun, pos, labels=labels,
                        font_size=10, font_weight='bold', ax=ax)

ax.set_title('LONG-RUN EQUILIBRIUM Network (Error Correction) - Rank=2 CORRECTED SIGNS\n(Red=Amplifying, Blue=Dampening; Width=Strength; Node size=Beta importance)',
             fontsize=14, fontweight='bold', pad=20)

legend_elements = [
    Line2D([0], [0], color='red', linewidth=3, label='Amplifying (+)'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dampening (-)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
           markersize=10, markeredgecolor='black', markeredgewidth=2,
           label='Node size = Beta importance', linestyle='None')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_longrun_network_rank2_CORRECTED.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Long-run network saved")

# Short-run network
print("    [5b] Short-run network...")
G_shortrun = nx.DiGraph()

for var in SELECTED_VARS:
    G_shortrun.add_node(var)

gamma_threshold = 0.15
for i, var_to in enumerate(SELECTED_VARS):
    for j, var_from in enumerate(SELECTED_VARS):
        if i != j:
            coef = gamma_df.iloc[i, j]
            if abs(coef) > gamma_threshold:
                G_shortrun.add_edge(var_from, var_to, weight=abs(coef), direction=np.sign(coef))

fig, ax = plt.subplots(figsize=(16, 14))

pos_sr = nx.circular_layout(G_shortrun)

nx.draw_networkx_nodes(G_shortrun, pos_sr, node_size=2000,
                       node_color='lightblue', edgecolors='black',
                       linewidths=2.5, alpha=0.9, ax=ax)

edges_pos = [(u, v) for u, v, d in G_shortrun.edges(data=True) if d['direction'] > 0]
edges_neg = [(u, v) for u, v, d in G_shortrun.edges(data=True) if d['direction'] < 0]

if edges_pos or edges_neg:
    max_gamma = max([d['weight'] for u, v, d in G_shortrun.edges(data=True)])

    if edges_pos:
        weights_pos = [G_shortrun[u][v]['weight'] for u, v in edges_pos]
        widths_pos = [3 + (w / max_gamma) * 5 for w in weights_pos]
        nx.draw_networkx_edges(G_shortrun, pos_sr, edgelist=edges_pos,
                               width=widths_pos, edge_color='red', alpha=0.7,
                               arrows=True, arrowsize=25, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15',
                               min_source_margin=20, min_target_margin=20, ax=ax)

    if edges_neg:
        weights_neg = [G_shortrun[u][v]['weight'] for u, v in edges_neg]
        widths_neg = [3 + (w / max_gamma) * 5 for w in weights_neg]
        nx.draw_networkx_edges(G_shortrun, pos_sr, edgelist=edges_neg,
                               width=widths_neg, edge_color='blue', alpha=0.7,
                               arrows=True, arrowsize=25, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.15',
                               min_source_margin=20, min_target_margin=20, ax=ax)

labels_sr = {node: DISPLAY_NAMES_LONG[node] for node in G_shortrun.nodes()}
nx.draw_networkx_labels(G_shortrun, pos_sr, labels=labels_sr,
                        font_size=10, font_weight='bold', ax=ax)

ax.set_title('SHORT-RUN DYNAMICS Network (Year-to-Year VAR) - Rank=2\n(Red=Amplifying, Blue=Dampening; Width=Coefficient strength)',
             fontsize=14, fontweight='bold', pad=20)

legend_sr = [
    Line2D([0], [0], color='red', linewidth=3, label='Amplifying (+)'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dampening (-)')
]
ax.legend(handles=legend_sr, loc='upper left', fontsize=11)
ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_shortrun_network_rank2_CORRECTED.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Short-run network saved")

# Side-by-side network comparison
print("    [5c] Side-by-side network comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))

# LEFT: Long-run
nx.draw_networkx_nodes(G_longrun, pos, node_size=node_sizes,
                       node_color='yellow', edgecolors='black',
                       linewidths=2.5, alpha=0.9, ax=ax1)

if edges_amplifying:
    nx.draw_networkx_edges(G_longrun, pos, edgelist=edges_amplifying,
                           width=widths_amp, edge_color='red', alpha=0.7,
                           arrows=True, arrowsize=25, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=25, min_target_margin=25, ax=ax1)

if edges_dampening:
    nx.draw_networkx_edges(G_longrun, pos, edgelist=edges_dampening,
                           width=widths_damp, edge_color='blue', alpha=0.7,
                           arrows=True, arrowsize=25, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=25, min_target_margin=25, ax=ax1)

nx.draw_networkx_labels(G_longrun, pos, labels=labels,
                        font_size=9, font_weight='bold', ax=ax1)

ax1.set_title('LONG-RUN EQUILIBRIUM\n(Error Correction)\n\nRed=Amplifying, Blue=Dampening',
             fontsize=13, fontweight='bold', pad=15)
ax1.axis('off')

# RIGHT: Short-run
nx.draw_networkx_nodes(G_shortrun, pos_sr, node_size=2000,
                       node_color='lightblue', edgecolors='black',
                       linewidths=2.5, alpha=0.9, ax=ax2)

if edges_pos:
    nx.draw_networkx_edges(G_shortrun, pos_sr, edgelist=edges_pos,
                           width=widths_pos, edge_color='red', alpha=0.7,
                           arrows=True, arrowsize=25, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=20, min_target_margin=20, ax=ax2)

if edges_neg:
    nx.draw_networkx_edges(G_shortrun, pos_sr, edgelist=edges_neg,
                           width=widths_neg, edge_color='blue', alpha=0.7,
                           arrows=True, arrowsize=25, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=20, min_target_margin=20, ax=ax2)

nx.draw_networkx_labels(G_shortrun, pos_sr, labels=labels_sr,
                        font_size=9, font_weight='bold', ax=ax2)

ax2.set_title('SHORT-RUN DYNAMICS\n(Year-to-Year VAR)\n\nRed=Amplifying, Blue=Dampening',
             fontsize=13, fontweight='bold', pad=15)
ax2.axis('off')

fig.suptitle('VECM Network Comparison: Long-Run Equilibrium vs Short-Run Dynamics (Rank=2 - CORRECTED SIGNS)\nArrow width = Relationship strength; Node size (left) = Beta importance',
             fontsize=15, fontweight='bold', y=0.98)

legend_elements = [
    Line2D([0], [0], color='red', linewidth=4, label='Amplifying (+)', marker='>', markersize=10),
    Line2D([0], [0], color='blue', linewidth=4, label='Dampening (-)', marker='>', markersize=10)
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12,
           bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(OUTPUT_DIR / "vecm_network_comparison_rank2_CORRECTED.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Side-by-side comparison saved")

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS COMPLETE!")
print("=" * 80)
print(f"\nSaved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. vecm_influence_comparison_rank2_CORRECTED.png")
print("  2. vecm_longrun_vs_shortrun_comparison_CORRECTED.png")
print("  3. vecm_longrun_network_rank2_CORRECTED.png")
print("  4. vecm_shortrun_network_rank2_CORRECTED.png")
print("  5. vecm_network_comparison_rank2_CORRECTED.png")
print("\nAll visualizations use SIGN-CORRECTED beta coefficients that match")
print("empirical directional relationships in the data.")
print("=" * 80)
