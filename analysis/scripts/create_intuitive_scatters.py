"""
Create INTUITIVE Pairwise Cointegration Scatter Plots
More straightforward visualizations with clear direction indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

output_dir = 'data/analysis/VECM_7VARS'

print("Creating intuitive scatter plots...")

# Load data
df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

endog_vars = ['Junior_Enlisted_Z', 'FOIA_Simple_Days_Z', 'Total_PAS_Z',
              'Total_Civilians_Z', 'Policy_Count_Log', 'Field_Grade_Officers_Z', 'GOFOs_Z']

data = df[endog_vars].copy().dropna()
years = df['FY'].loc[data.index].values

# Read top pairwise relationships
pairwise_df = pd.read_excel('data/analysis/pairwise_cointegration.xlsx')
sig_pairs = pairwise_df[pairwise_df['Cointegrated_5pct'] == 'YES **'].nlargest(6, 'Trace_Stat')

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor='white')
axes = axes.flatten()

for idx, (_, row) in enumerate(sig_pairs.iterrows()):
    var1 = row['Variable_1']
    var2 = row['Variable_2']
    trace = row['Trace_Stat']

    ax = axes[idx]

    x_data = data[var1].values
    y_data = data[var2].values

    # Calculate correlation
    correlation = np.corrcoef(x_data, y_data)[0, 1]

    # Determine direction over time
    x_start = x_data[0]
    x_end = x_data[-1]
    y_start = y_data[0]
    y_end = y_data[-1]

    x_change = "↑ INCREASED" if x_end > x_start else "↓ DECREASED"
    y_change = "↑ INCREASED" if y_end > y_start else "↓ DECREASED"
    relationship = "SAME direction" if (x_end > x_start) == (y_end > y_start) else "OPPOSITE directions"

    # Plot with gradient
    scatter = ax.scatter(x_data, y_data, s=120, alpha=0.7, c=years, cmap='viridis',
                        edgecolors='black', linewidths=1.5, zorder=3)

    # Trend line
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=3, label=f'Trend (r={correlation:.2f})', alpha=0.8)

    # Mark start and end points with years
    ax.scatter([x_start], [y_start], s=400, c='purple', marker='s',
              edgecolors='black', linewidths=3, zorder=5, label=f'Start ({int(years[0])})')
    ax.scatter([x_end], [y_end], s=400, c='yellow', marker='*',
              edgecolors='black', linewidths=3, zorder=5, label=f'End ({int(years[-1])})')

    # Add arrow showing time progression
    mid_idx = len(x_data) // 2
    ax.annotate('', xy=(x_data[mid_idx+5], y_data[mid_idx+5]),
               xytext=(x_data[mid_idx], y_data[mid_idx]),
               arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.7))

    # Clean labels
    var1_label = var1.replace('_Z', '').replace('_', ' ')
    var2_label = var2.replace('_Z', '').replace('_', ' ')

    ax.set_xlabel(f'{var1_label}\n(z-scored)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{var2_label}\n(z-scored)', fontsize=13, fontweight='bold')

    # Title with clear interpretation
    title = f'{var1_label} vs {var2_label}\n'
    if correlation < -0.5:
        title += 'INVERSE Relationship (move opposite directions)'
    elif correlation > 0.5:
        title += 'POSITIVE Relationship (move together)'
    else:
        title += 'WEAK Relationship'

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # Add text box with interpretation
    textstr = f'Over 37 years:\n'
    textstr += f'{var1_label}: {x_change}\n'
    textstr += f'{var2_label}: {y_change}\n'
    textstr += f'→ Move in {relationship}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontweight='bold')

    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Year', fontsize=10, fontweight='bold')

plt.suptitle('Pairwise Cointegration: 37-Year Relationships (1987-2024)\n' +
            'Purple Square = Start | Yellow Star = End | Red Arrow = Time Direction',
            fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(f'{output_dir}/INTUITIVE_pairwise_scatters.png', dpi=300, bbox_inches='tight')

print(f"Saved: {output_dir}/INTUITIVE_pairwise_scatters.png")

# ============================================================================
# CREATE SIMPLIFIED VERSION - JUST TOP 3 WITH MORE SPACE
# ============================================================================
print("\nCreating simplified version with top 3 relationships...")

fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')

top3_pairs = sig_pairs.head(3)

for idx, (_, row) in enumerate(top3_pairs.iterrows()):
    var1 = row['Variable_1']
    var2 = row['Variable_2']
    trace = row['Trace_Stat']

    ax = axes[idx]

    x_data = data[var1].values
    y_data = data[var2].values

    correlation = np.corrcoef(x_data, y_data)[0, 1]

    # Changes over time
    x_pct = ((x_data[-1] - x_data[0]) / abs(x_data[0])) * 100 if x_data[0] != 0 else 0
    y_pct = ((y_data[-1] - y_data[0]) / abs(y_data[0])) * 100 if y_data[0] != 0 else 0

    # Plot
    scatter = ax.scatter(x_data, y_data, s=150, alpha=0.7, c=years, cmap='RdYlGn_r',
                        edgecolors='black', linewidths=2, zorder=3)

    # Trend line
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    ax.plot(x_line, p(x_line), "navy", linewidth=4, alpha=0.6, linestyle='--')

    # Highlight start and end
    ax.scatter([x_data[0]], [y_data[0]], s=600, c='darkblue', marker='o',
              edgecolors='white', linewidths=4, zorder=10, alpha=0.9)
    ax.scatter([x_data[-1]], [y_data[-1]], s=600, c='red', marker='o',
              edgecolors='white', linewidths=4, zorder=10, alpha=0.9)

    # Annotations
    ax.annotate(f'{int(years[0])}', xy=(x_data[0], y_data[0]),
               xytext=(x_data[0]-0.3, y_data[0]-0.3),
               fontsize=14, fontweight='bold', color='darkblue')
    ax.annotate(f'{int(years[-1])}', xy=(x_data[-1], y_data[-1]),
               xytext=(x_data[-1]+0.2, y_data[-1]+0.2),
               fontsize=14, fontweight='bold', color='red')

    # Labels
    var1_clean = var1.replace('_Z', '').replace('_', ' ')
    var2_clean = var2.replace('_Z', '').replace('_', ' ')

    ax.set_xlabel(var1_clean, fontsize=16, fontweight='bold')
    ax.set_ylabel(var2_clean, fontsize=16, fontweight='bold')

    # Interpretation
    if 'Junior' in var1 and 'Field' in var2:
        interp = '"TEETH TO TAIL" SHIFT\nAs combat troops decline,\nbureaucratic officers increase'
    elif 'Junior' in var1 and 'Civilian' in var2:
        interp = 'CIVILIANIZATION\nMilitary downsizes,\ncivilians backfill functions'
    elif 'Junior' in var1 and 'GOFO' in var2:
        interp = 'TOP-HEAVY LEADERSHIP\nFewer troops,\nmore generals/admirals'
    elif 'Civilian' in var1 and 'Policy' in var2:
        interp = 'BUREAUCRATIC EMPIRE\nMore civilians,\nmore regulations'
    else:
        interp = f'Correlation: {correlation:.2f}'

    ax.text(0.5, 0.02, interp, transform=ax.transAxes,
            fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=3))

    ax.set_title(f'{var1_clean} vs {var2_clean}\n' +
                f'Correlation: {correlation:.2f} | Trace: {trace:.1f}',
                fontsize=15, fontweight='bold', pad=15)

    ax.grid(alpha=0.4, linestyle='--', linewidth=1.5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Year', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

plt.suptitle('Top 3 Long-Run Relationships: DoD Bureaucratic Growth (1987-2024)\n' +
            'Blue Circle = 1987 Start | Red Circle = 2024 End',
            fontsize=20, fontweight='bold', y=1.00)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{output_dir}/TOP3_relationships_simple.png', dpi=300, bbox_inches='tight')

print(f"Saved: {output_dir}/TOP3_relationships_simple.png")

print("\n" + "="*80)
print("INTUITIVE SCATTER PLOTS CREATED")
print("="*80)
print("1. INTUITIVE_pairwise_scatters.png - All 6 with clear annotations")
print("2. TOP3_relationships_simple.png - Your 3 key findings, large and clear")
print("="*80)
