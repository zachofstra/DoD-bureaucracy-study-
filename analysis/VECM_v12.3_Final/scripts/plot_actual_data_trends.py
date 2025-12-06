"""
Plot actual data trends to verify GOFOs vs Junior Enlisted relationship
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis")
DATA_FILE = BASE_DIR / "complete_normalized_dataset_v12.3.xlsx"
OUTPUT_DIR = BASE_DIR / "VECM_v12.3_Final" / "VECM_Rank2_Final_Executive_Summary"

print("=" * 80)
print("PLOTTING ACTUAL DATA TRENDS: GOFOs vs Junior Enlisted")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
df = pd.read_excel(DATA_FILE)
print(f"    Data shape: {df.shape}")
print(f"    Columns: {df.columns.tolist()}")

# Drop rows with NaN values
df_clean = df.dropna(subset=['GOFOs_Z', 'Junior_Enlisted_Z'])
print(f"    After dropping NaN: {df_clean.shape}")

# Check if Year column exists
if 'FY' in df_clean.columns:
    years = df_clean['FY'].values
elif 'Year' in df_clean.columns:
    years = df_clean['Year'].values
elif 'year' in df_clean.columns:
    years = df_clean['year'].values
else:
    years = np.arange(len(df_clean))
    print("    Using index as years")

print(f"    Year range: {years.min()} to {years.max()}")

# Get the variables
gofo_col = 'GOFOs_Z'
je_col = 'Junior_Enlisted_Z'

if gofo_col not in df_clean.columns or je_col not in df_clean.columns:
    print(f"\nERROR: Columns not found!")
    print(f"Available columns: {df_clean.columns.tolist()}")
    exit()

gofos = df_clean[gofo_col].values
junior_enlisted = df_clean[je_col].values

print(f"\n[2] Data statistics:")
print(f"    GOFOs_Z: min={gofos.min():.3f}, max={gofos.max():.3f}, mean={gofos.mean():.3f}")
print(f"    Junior_Enlisted_Z: min={junior_enlisted.min():.3f}, max={junior_enlisted.max():.3f}, mean={junior_enlisted.mean():.3f}")

# Calculate trends
from scipy import stats
gofo_trend = stats.linregress(range(len(gofos)), gofos)
je_trend = stats.linregress(range(len(junior_enlisted)), junior_enlisted)

print(f"\n[3] Linear trends:")
print(f"    GOFOs_Z slope: {gofo_trend.slope:.4f} (per year)")
print(f"    Junior_Enlisted_Z slope: {je_trend.slope:.4f} (per year)")
print(f"\n    GOFOs direction: {'INCREASING' if gofo_trend.slope > 0 else 'DECREASING'}")
print(f"    Junior Enlisted direction: {'INCREASING' if je_trend.slope > 0 else 'DECREASING'}")

if np.sign(gofo_trend.slope) != np.sign(je_trend.slope):
    print(f"\n    >>> OPPOSITE TRENDS CONFIRMED! <<<")
else:
    print(f"\n    >>> Same direction trends")

# Create visualization
print("\n[4] Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Time series overlay
ax1 = axes[0, 0]
ax1.plot(years, gofos, 'o-', color='darkblue', linewidth=2, markersize=6,
         label=f'GOFOs (slope={gofo_trend.slope:.4f})', alpha=0.7)
ax1.plot(years, junior_enlisted, 's-', color='darkred', linewidth=2, markersize=6,
         label=f'Junior Enlisted (slope={je_trend.slope:.4f})', alpha=0.7)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Z-Score (normalized)', fontsize=12, fontweight='bold')
ax1.set_title('GOFOs vs Junior Enlisted Over Time\n(Both Z-score normalized)',
              fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(alpha=0.3)

# Plot 2: Trend lines
ax2 = axes[0, 1]
x_vals = np.array(range(len(gofos)))
gofo_fitted = gofo_trend.intercept + gofo_trend.slope * x_vals
je_fitted = je_trend.intercept + je_trend.slope * x_vals

ax2.scatter(years, gofos, color='darkblue', s=50, alpha=0.5, label='GOFOs (actual)')
ax2.plot(years, gofo_fitted, color='darkblue', linewidth=3,
         label=f'GOFOs trend (slope={gofo_trend.slope:.4f})')
ax2.scatter(years, junior_enlisted, color='darkred', s=50, alpha=0.5,
            label='Junior Enlisted (actual)')
ax2.plot(years, je_fitted, color='darkred', linewidth=3,
         label=f'JE trend (slope={je_trend.slope:.4f})')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Z-Score (normalized)', fontsize=12, fontweight='bold')
ax2.set_title('Linear Trends with Fitted Lines', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(alpha=0.3)

# Plot 3: Scatter plot to show relationship
ax3 = axes[1, 0]
scatter_trend = stats.linregress(gofos, junior_enlisted)
ax3.scatter(gofos, junior_enlisted, s=100, alpha=0.6, c=range(len(gofos)),
            cmap='viridis', edgecolors='black', linewidth=1)
ax3.plot(gofos, scatter_trend.intercept + scatter_trend.slope * gofos,
         'r--', linewidth=2, label=f'Correlation slope={scatter_trend.slope:.4f}')
ax3.set_xlabel('GOFOs_Z', fontsize=12, fontweight='bold')
ax3.set_ylabel('Junior_Enlisted_Z', fontsize=12, fontweight='bold')
ax3.set_title(f'Scatter: GOFOs vs Junior Enlisted\n(Correlation r={stats.pearsonr(gofos, junior_enlisted)[0]:.3f})',
              fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=11)
ax3.grid(alpha=0.3)

# Add colorbar for time
sm = plt.cm.ScalarMappable(cmap='viridis',
                           norm=plt.Normalize(vmin=years.min(), vmax=years.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax3)
cbar.set_label('Year', fontsize=10, fontweight='bold')

# Plot 4: Summary statistics
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
EMPIRICAL VERIFICATION
{'=' * 50}

DATA TRENDS (1987-2024):
  • GOFOs_Z slope: {gofo_trend.slope:.4f} per year
  • Junior_Enlisted_Z slope: {je_trend.slope:.4f} per year

DIRECTION:
  • GOFOs: {'INCREASING ↑' if gofo_trend.slope > 0 else 'DECREASING ↓'}
  • Junior Enlisted: {'INCREASING ↑' if je_trend.slope > 0 else 'DECREASING ↓'}

RELATIONSHIP:
  • Correlation: r = {stats.pearsonr(gofos, junior_enlisted)[0]:.3f}
  • They move: {'OPPOSITE' if np.sign(gofo_trend.slope) != np.sign(je_trend.slope) or stats.pearsonr(gofos, junior_enlisted)[0] < 0 else 'TOGETHER'}

VECM MODEL ASSUMPTION:
  • Beta coefficients: SAME SIGN (both positive)
  • Implies: Variables move TOGETHER in equilibrium

CONCLUSION:
  • {'MISMATCH: Model assumes same-direction' if stats.pearsonr(gofos, junior_enlisted)[0] < 0 else 'Match: Model correct'}
  • {'  movement, but data shows opposite!' if stats.pearsonr(gofos, junior_enlisted)[0] < 0 else '  Both move in same direction'}
"""

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "empirical_verification_GOFOs_vs_JuniorEnlisted.png",
            dpi=300, bbox_inches='tight')
plt.close()

print("    Visualization saved!")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print(f"\nSaved to: {OUTPUT_DIR / 'empirical_verification_GOFOs_vs_JuniorEnlisted.png'}")
print("\nKey finding:")
if np.sign(gofo_trend.slope) != np.sign(je_trend.slope):
    print("  >>> GOFOs and Junior Enlisted moved in OPPOSITE directions")
    print("  >>> VECM beta coefficients (same sign) don't match this reality")
else:
    print("  >>> GOFOs and Junior Enlisted moved in SAME direction")
    print("  >>> VECM beta coefficients match this relationship")
print("=" * 80)
