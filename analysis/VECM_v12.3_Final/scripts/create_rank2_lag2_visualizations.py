"""
Create all visualizations for VECM rank=2, lag=2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path

# Setup
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
INPUT_DIR = BASE_DIR / "VECM_Rank2_Lag2_FINAL"
OUTPUT_DIR = INPUT_DIR

print("Creating visualizations for VECM rank=2, lag=2...")

# Load matrices
alpha_df = pd.read_excel(INPUT_DIR / "alpha_matrix_rank2_lag2.xlsx", index_col=0)
beta_df = pd.read_excel(INPUT_DIR / "beta_matrix_rank2_lag2.xlsx", index_col=0)
longrun_df = pd.read_excel(INPUT_DIR / "longrun_influence_matrix.xlsx", index_col=0)
importance_df = pd.read_excel(INPUT_DIR / "variable_importance.xlsx")

SELECTED_VARS = alpha_df.index.tolist()

# Clean variable names for display
def clean_name(name):
    return name.replace('_Z', '').replace('_', ' ').replace('Policy Count Log', 'Policy Volume (Log)')

display_names = [clean_name(v) for v in SELECTED_VARS]

# 1. LONG-RUN VS SHORT-RUN IMPORTANCE
print("\n1. Creating long-run importance chart...")

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(display_names))
width = 0.6

# Only have long-run importance for this visualization
beta_importance = importance_df['Long_Run_Importance'].values

colors = plt.cm.YlOrRd(beta_importance / 100)

bars = ax.barh(x, beta_importance, width, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, beta_importance)):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.0f}',
            va='center', ha='left', fontweight='bold', fontsize=10)

ax.set_yticks(x)
ax.set_yticklabels(display_names, fontsize=11)
ax.set_xlabel('Importance Score (Normalized to 0-100)', fontsize=12, fontweight='bold')
ax.set_title('VECM Variable Importance: Long-Run Equilibrium (RANK=2, LAG=2)\n' +
             'Arrow width = Relationship strength, Node size = Beta importance',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 110)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_longrun_importance_rank2_lag2.png", dpi=300, bbox_inches='tight')
print(f"   Saved: vecm_longrun_importance_rank2_lag2.png")
plt.close()

# 2. LONG-RUN INFLUENCE HEATMAP
print("\n2. Creating long-run influence heatmap...")

# Determine sign for each relationship (for color direction)
signed_direction = np.sign(longrun_df.values)

# Calculate signed magnitude for color intensity
signed_magnitude = longrun_df.values.copy()

fig, ax = plt.subplots(figsize=(14, 12))

# Create color map: blue = dampening (-), red = amplifying (+)
vmax = np.abs(longrun_df.values).max()
im = ax.imshow(signed_magnitude, cmap='RdBu_r', aspect='auto',
               vmin=-vmax, vmax=vmax)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Influence Strength\n(Red = Amplifying, Blue = Dampening)',
               rotation=270, labelpad=25, fontsize=11, fontweight='bold')

# Add grid
ax.set_xticks(np.arange(len(display_names)))
ax.set_yticks(np.arange(len(display_names)))
ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(display_names, fontsize=10)

# Add text annotations
for i in range(len(SELECTED_VARS)):
    for j in range(len(SELECTED_VARS)):
        value = longrun_df.values[i, j]
        text_color = 'white' if abs(value) > vmax * 0.5 else 'black'
        ax.text(j, i, f'{value:.2f}',
                ha='center', va='center', color=text_color,
                fontsize=9, fontweight='bold')

ax.set_title('VECM Long-Run Influence Matrix (RANK=2, LAG=2)\n' +
             'Rows = Target variables, Columns = Source variables\n' +
             'WARNING: GOFOs=>Junior Enlisted shows AMPLIFYING (+) despite empirical r=-0.775',
             fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('From Variable (Source)', fontsize=12, fontweight='bold')
ax.set_ylabel('To Variable (Target)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_longrun_heatmap_rank2_lag2.png", dpi=300, bbox_inches='tight')
print(f"   Saved: vecm_longrun_heatmap_rank2_lag2.png")
plt.close()

# 3. NETWORK DIAGRAM
print("\n3. Creating network diagram...")

fig, ax = plt.subplots(figsize=(16, 12))

G = nx.DiGraph()

# Add nodes
for i, var in enumerate(display_names):
    importance = beta_importance[i]
    G.add_node(var, importance=importance)

# Add edges (only significant relationships, |influence| > 0.1)
edges_amplifying = []
edges_dampening = []
widths_amp = []
widths_damp = []

for i, target in enumerate(SELECTED_VARS):
    for j, source in enumerate(SELECTED_VARS):
        if i == j:
            continue

        influence = longrun_df.loc[target, source]

        if abs(influence) > 0.1:  # Only show significant relationships
            target_display = display_names[i]
            source_display = display_names[j]

            if influence > 0:
                edges_amplifying.append((source_display, target_display))
                widths_amp.append(min(abs(influence) * 3, 8))
            else:
                edges_dampening.append((source_display, target_display))
                widths_damp.append(min(abs(influence) * 3, 8))

# Circular layout
pos = nx.circular_layout(G)

# Draw nodes
node_sizes = [importance_df.loc[importance_df['Variable'] == SELECTED_VARS[i], 'Long_Run_Importance'].values[0] * 50 + 1000
              for i, name in enumerate(display_names)]
node_colors = plt.cm.YlOrRd(beta_importance / 100)

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                       edgecolors='black', linewidths=2, ax=ax)

# Draw edges
if edges_amplifying:
    nx.draw_networkx_edges(G, pos, edgelist=edges_amplifying,
                           width=widths_amp, edge_color='red', alpha=0.7,
                           arrows=True, arrowsize=25, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=25, min_target_margin=25, ax=ax)

if edges_dampening:
    nx.draw_networkx_edges(G, pos, edgelist=edges_dampening,
                           width=widths_damp, edge_color='blue', alpha=0.7,
                           arrows=True, arrowsize=25, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.15',
                           min_source_margin=25, min_target_margin=25, ax=ax)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

ax.set_title('VECM Long-Run Equilibrium Network (RANK=2, LAG=2)\n' +
             'Red = Amplifying (+), Blue = Dampening (-)\n' +
             'Node size = Beta importance',
             fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=3, label='Amplifying (+)'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dampening (-)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vecm_network_rank2_lag2.png", dpi=300, bbox_inches='tight')
print(f"   Saved: vecm_network_rank2_lag2.png")
plt.close()

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
print("\nFiles created:")
print("  - vecm_longrun_importance_rank2_lag2.png")
print("  - vecm_longrun_heatmap_rank2_lag2.png")
print("  - vecm_network_rank2_lag2.png")
