"""
NetLogo BehaviorSpace Analysis - First Mover Detection
=======================================================
Analyzes which variable moves ±2 standard deviations FIRST across all runs.

For each run:
1. Calculate standard deviation for each variable
2. Find first tick where |value| > 2*SD
3. Identify which variable crossed threshold first

Then aggregate across all runs to find most common first mover.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
NETLOGO_FILE = BASE_DIR / "netlogo" / "DoD_Bureaucracy_VECM_Rank2 DoW_bureaucracy-spreadsheet.csv"
OUTPUT_DIR = BASE_DIR / "netlogo_first_mover_analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

THRESHOLD_SD = 2.0  # Number of standard deviations for threshold

print("=" * 80)
print("NETLOGO FIRST MOVER ANALYSIS - DoD BUREAUCRACY VECM")
print("=" * 80)
print(f"\nThreshold: ±{THRESHOLD_SD} standard deviations")

# ============================================================================
# STEP 1: Load and parse BehaviorSpace format
# ============================================================================
print("\n[1] Loading BehaviorSpace output...")

# BehaviorSpace format has metadata rows at top
# Find where actual data starts
with open(NETLOGO_FILE, 'r') as f:
    lines = f.readlines()

# Find the row that contains column headers
header_row_idx = None
for i, line in enumerate(lines):
    if '[step]' in line.lower():
        header_row_idx = i
        print(f"    Found header at row {i}")
        break

if header_row_idx is None:
    print("ERROR: Could not find header row with [step]")
    exit(1)

# Load data starting from header row
print(f"    Reading from row {header_row_idx}...")
df = pd.read_csv(NETLOGO_FILE, skiprows=header_row_idx)

print(f"    Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"    Columns: {list(df.columns[:15])}...")

# ============================================================================
# STEP 2: Identify variable columns and clean data
# ============================================================================
print("\n[2] Identifying variables...")

# Common metadata column names in BehaviorSpace
metadata_cols = [
    '[run number]', '[step]', 'year', 'tick-count',
    'error-correction-strength', 'var-strength', 'noise-level', 'max-year'
]

# Find all columns that are NOT metadata
var_cols = [col for col in df.columns if col not in metadata_cols]
print(f"    Found {len(var_cols)} variables:")
for var in var_cols:
    print(f"      - {var}")

# Check if we have run numbers
if '[run number]' in df.columns:
    runs = sorted(df['[run number]'].unique())
    print(f"\n    Detected {len(runs)} runs")
else:
    print("\n    ERROR: No [run number] column found!")
    exit(1)

# ============================================================================
# STEP 3: Analyze each run to find first mover
# ============================================================================
print("\n[3] Analyzing first mover for each run...")

first_mover_results = []

for run_num in runs:
    run_data = df[df['[run number]'] == run_num].copy()

    # Skip if run has too few observations
    if len(run_data) < 10:
        continue

    # Calculate standard deviation for each variable (using all ticks in this run)
    std_devs = {}
    for var in var_cols:
        std_devs[var] = run_data[var].std()

    # Find first tick where each variable exceeds ±2 SD
    first_crossing = {}

    for var in var_cols:
        if std_devs[var] == 0 or np.isnan(std_devs[var]):
            first_crossing[var] = np.inf
            continue

        threshold = THRESHOLD_SD * std_devs[var]

        # Find first tick where |value| > threshold
        mask = np.abs(run_data[var]) > threshold

        if mask.any():
            first_tick = run_data.loc[mask, '[step]'].iloc[0]
            first_crossing[var] = first_tick
        else:
            first_crossing[var] = np.inf

    # Identify which variable moved first (minimum tick)
    valid_crossings = {var: tick for var, tick in first_crossing.items() if tick != np.inf}

    if valid_crossings:
        first_mover = min(valid_crossings, key=valid_crossings.get)
        first_tick = valid_crossings[first_mover]

        first_mover_results.append({
            'run': run_num,
            'first_mover': first_mover,
            'first_tick': first_tick,
            'all_crossings': first_crossing
        })

print(f"    Analyzed {len(first_mover_results)} runs successfully")

# ============================================================================
# STEP 4: Aggregate results across all runs
# ============================================================================
print("\n[4] Aggregating results across all runs...")

# Count how many times each variable was first
first_mover_counts = Counter([r['first_mover'] for r in first_mover_results])

# Sort by frequency
sorted_first_movers = first_mover_counts.most_common()

print(f"\n{'='*80}")
print("RESULTS: FIRST MOVER FREQUENCY")
print(f"{'='*80}")
print(f"{'Rank':<6} {'Variable':<35} {'Count':<10} {'Percentage':<15}")
print(f"{'-'*80}")

total_runs = len(first_mover_results)
for rank, (var, count) in enumerate(sorted_first_movers, 1):
    pct = (count / total_runs) * 100
    print(f"{rank:<6} {var:<35} {count:<10} {pct:<15.2f}%")

# Identify THE most common first mover
most_common_first_mover = sorted_first_movers[0][0]
most_common_count = sorted_first_movers[0][1]
most_common_pct = (most_common_count / total_runs) * 100

print(f"\n{'='*80}")
print(f"PRIMARY INITIAL CAUSE: {most_common_first_mover}")
print(f"Moves first in {most_common_count}/{total_runs} runs ({most_common_pct:.1f}%)")
print(f"{'='*80}")

# ============================================================================
# STEP 5: Calculate average first movement times
# ============================================================================
print("\n[5] Calculating average first movement times...")

# For each variable, calculate average tick when it first crosses threshold
avg_first_times = {}

for var in var_cols:
    times = [r['all_crossings'][var] for r in first_mover_results
             if r['all_crossings'][var] != np.inf]

    if times:
        avg_first_times[var] = {
            'mean': np.mean(times),
            'median': np.median(times),
            'min': np.min(times),
            'max': np.max(times),
            'count': len(times)
        }
    else:
        avg_first_times[var] = {
            'mean': np.inf,
            'median': np.inf,
            'min': np.inf,
            'max': np.inf,
            'count': 0
        }

# Sort by mean first crossing time
sorted_by_time = sorted(avg_first_times.items(), key=lambda x: x[1]['mean'])

print(f"\n{'Variable':<35} {'Mean Tick':<12} {'Median':<12} {'Min':<8} {'Max':<8} {'% Runs':<10}")
print(f"{'-'*100}")

for var, times in sorted_by_time:
    if times['count'] > 0:
        pct_runs = (times['count'] / total_runs) * 100
        print(f"{var:<35} {times['mean']:<12.1f} {times['median']:<12.1f} "
              f"{times['min']:<8.0f} {times['max']:<8.0f} {pct_runs:<10.1f}%")
    else:
        print(f"{var:<35} {'Never':<12} {'Never':<12} {'-':<8} {'-':<8} {'0.0%':<10}")

# ============================================================================
# VISUALIZATION 1: First Mover Frequency
# ============================================================================
print("\n[6] Creating visualizations...")

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Plot 1: Bar chart of first mover frequency
ax1 = axes[0]

vars_sorted = [var for var, count in sorted_first_movers]
counts_sorted = [count for var, count in sorted_first_movers]
colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(vars_sorted))]

bars = ax1.barh(vars_sorted, counts_sorted, color=colors, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Number of Runs Where Variable Moved First', fontsize=12, fontweight='bold')
ax1.set_ylabel('Variable', fontsize=12, fontweight='bold')
ax1.set_title(f'Which Variable Moves First? (Crosses ±{THRESHOLD_SD} SD Threshold)\n'
              f'Across {total_runs} NetLogo Simulation Runs',
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3)

# Add counts on bars
for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
    width = bar.get_width()
    pct = (count / total_runs) * 100
    ax1.text(width, bar.get_y() + bar.get_height()/2,
            f'  {count} ({pct:.1f}%)',
            va='center', fontsize=10, fontweight='bold')

# Highlight the winner
if len(vars_sorted) > 0:
    ax1.text(counts_sorted[0] * 1.15, 0,
            'PRIMARY\nINITIAL CAUSE',
            va='center', fontsize=11, fontweight='bold',
            color='#e74c3c',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Plot 2: Average first crossing time
ax2 = axes[1]

vars_by_time = [var for var, times in sorted_by_time if times['count'] > 0]
mean_times = [times['mean'] for var, times in sorted_by_time if times['count'] > 0]

if len(vars_by_time) > 0:
    bars2 = ax2.barh(vars_by_time, mean_times,
                     color=sns.color_palette('viridis', len(vars_by_time)),
                     edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Average Tick Number (First Crossing)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Variable', fontsize=12, fontweight='bold')
    ax2.set_title(f'Average Time to First Movement (±{THRESHOLD_SD} SD)',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(axis='x', alpha=0.3)

    # Add values on bars
    for bar, time in zip(bars2, mean_times):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'  {time:.1f}',
                va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'first_mover_analysis.png', dpi=300, bbox_inches='tight')
print(f"    [SAVED] first_mover_analysis.png")

# ============================================================================
# VISUALIZATION 2: Distribution of first crossing times
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Create box plot showing distribution of first crossing times for top variables
top_vars = [var for var, count in sorted_first_movers[:8]]  # Top 8

box_data = []
box_labels = []

for var in top_vars:
    times = [r['all_crossings'][var] for r in first_mover_results
             if r['all_crossings'][var] != np.inf]
    if times:
        box_data.append(times)
        box_labels.append(var)

if box_data:
    bp = ax.boxplot(box_data, labels=box_labels, vert=False, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color boxes
    for i, patch in enumerate(bp['boxes']):
        if i == 0:  # Most common first mover
            patch.set_facecolor('#e74c3c')
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.5)

    ax.set_xlabel('Tick Number (First Crossing)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution of First Movement Times (±{THRESHOLD_SD} SD)\n'
                 f'Red = Most Common First Mover',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'first_movement_distribution.png', dpi=300, bbox_inches='tight')
print(f"    [SAVED] first_movement_distribution.png")

# ============================================================================
# SAVE DETAILED RESULTS
# ============================================================================
print("\n[7] Saving detailed results...")

with open(OUTPUT_DIR / 'FIRST_MOVER_ANALYSIS.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NETLOGO FIRST MOVER ANALYSIS - DoD BUREAUCRACY VECM RANK 2\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Analysis Parameters:\n")
    f.write(f"  - Threshold: ±{THRESHOLD_SD} standard deviations\n")
    f.write(f"  - Total runs analyzed: {total_runs}\n")
    f.write(f"  - Variables tracked: {len(var_cols)}\n\n")

    f.write("=" * 80 + "\n")
    f.write("PRIMARY FINDING: INITIAL CAUSE\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"The variable that moves FIRST (crosses ±{THRESHOLD_SD} SD) most often:\n\n")
    f.write(f"  >>> {most_common_first_mover} <<<\n\n")
    f.write(f"Frequency: {most_common_count}/{total_runs} runs ({most_common_pct:.1f}%)\n\n")

    if most_common_first_mover in avg_first_times:
        times = avg_first_times[most_common_first_mover]
        f.write(f"Timing Statistics:\n")
        f.write(f"  - Average first movement: tick {times['mean']:.1f}\n")
        f.write(f"  - Median: tick {times['median']:.1f}\n")
        f.write(f"  - Range: tick {times['min']:.0f} to {times['max']:.0f}\n\n")

    f.write("INTERPRETATION:\n")
    f.write(f"{most_common_first_mover} acts as the PRIMARY DRIVER in the VECM model.\n")
    f.write(f"Changes in this variable trigger cascading effects throughout the\n")
    f.write(f"DoD bureaucratic system. This suggests {most_common_first_mover} is either:\n")
    f.write(f"  1. An exogenous shock to the system\n")
    f.write(f"  2. The most volatile/reactive variable\n")
    f.write(f"  3. A fundamental forcing mechanism in bureaucratic growth\n\n")

    f.write("=" * 80 + "\n")
    f.write("FIRST MOVER FREQUENCY RANKING\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"{'Rank':<6} {'Variable':<35} {'Count':<10} {'Percentage':<15}\n")
    f.write("-" * 80 + "\n")

    for rank, (var, count) in enumerate(sorted_first_movers, 1):
        pct = (count / total_runs) * 100
        f.write(f"{rank:<6} {var:<35} {count:<10} {pct:<15.2f}%\n")

    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("AVERAGE FIRST MOVEMENT TIMES\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"{'Variable':<35} {'Mean':<12} {'Median':<12} {'Min':<8} {'Max':<8} {'% Runs':<10}\n")
    f.write("-" * 100 + "\n")

    for var, times in sorted_by_time:
        if times['count'] > 0:
            pct_runs = (times['count'] / total_runs) * 100
            f.write(f"{var:<35} {times['mean']:<12.1f} {times['median']:<12.1f} "
                   f"{times['min']:<8.0f} {times['max']:<8.0f} {pct_runs:<10.1f}%\n")
        else:
            f.write(f"{var:<35} {'Never':<12} {'Never':<12} {'-':<8} {'-':<8} {'0.0%':<10}\n")

    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("IMPLICATIONS FOR DOD BUREAUCRACY THESIS\n")
    f.write("=" * 80 + "\n\n")

    f.write("This analysis identifies the INITIAL CAUSE of bureaucratic dynamics in\n")
    f.write("the VECM model. The variable that moves first represents the triggering\n")
    f.write("mechanism for the Iron Cage of bureaucracy.\n\n")

    f.write("For Weber's Iron Cage theory:\n")
    f.write("  - The first mover is the exogenous shock that sets rationalization in motion\n")
    f.write("  - Subsequent movements show the cascading effects of bureaucratic expansion\n")
    f.write("  - Variables that never cross threshold are stable/constrained by equilibrium\n\n")

    f.write("For policy recommendations:\n")
    f.write(f"  - Controlling {most_common_first_mover} could prevent bureaucratic growth spirals\n")
    f.write(f"  - Interventions targeting {most_common_first_mover} will have maximum leverage\n")
    f.write(f"  - Monitoring {most_common_first_mover} provides early warning of expansion\n")

print(f"    [SAVED] FIRST_MOVER_ANALYSIS.txt")

# Save raw data
results_df = pd.DataFrame(first_mover_results)
results_df.to_excel(OUTPUT_DIR / 'first_mover_by_run.xlsx', index=False)
print(f"    [SAVED] first_mover_by_run.xlsx")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nKEY FINDING:")
print(f"  PRIMARY INITIAL CAUSE: {most_common_first_mover}")
print(f"  Frequency: {most_common_count}/{total_runs} runs ({most_common_pct:.1f}%)")
print(f"\nFiles generated:")
print(f"  1. first_mover_analysis.png")
print(f"  2. first_movement_distribution.png")
print(f"  3. FIRST_MOVER_ANALYSIS.txt")
print(f"  4. first_mover_by_run.xlsx")
