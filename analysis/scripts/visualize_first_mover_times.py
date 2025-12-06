"""
Visualize First Mover Times - When each variable crosses ±2 SD
===============================================================
Creates line graphs showing the distribution of first crossing times
"""

import csv
import numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
NETLOGO_FILE = BASE_DIR / "netlogo" / "DoD_Bureaucracy_VECM_Rank2 DoW_bureaucracy-spreadsheet_3"
OUTPUT_DIR = BASE_DIR / "netlogo_first_mover_analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

THRESHOLD_SD = 2.0

print("=" * 80)
print("VISUALIZING FIRST MOVER TIMES")
print("=" * 80)

# Step 1: Parse header
print("\n[1] Parsing header...")
with open(NETLOGO_FILE, 'r', encoding='utf-8') as f:
    header_lines = [next(f).strip() for _ in range(15)]

run_numbers_raw = header_lines[6].replace('"', '').split(',')
run_numbers = [int(x) if x else None for x in run_numbers_raw[1:]]

reporter_row = header_lines[11].replace('"', '').split(',')
variables = []
i = 1
while len(variables) < 8:
    if i >= len(reporter_row):
        break
    col_name = reporter_row[i]
    if col_name and col_name != '[step]' and col_name not in variables:
        variables.append(col_name)
    i += 1

print(f"    Variables: {variables}")
cols_per_run = len(variables) + 1
num_runs = max([rn for rn in run_numbers if rn is not None])
print(f"    Total runs: {num_runs}")

# Step 2: Read and analyze data
print("\n[2] Reading and analyzing data...")
run_data = defaultdict(lambda: {var: [] for var in variables})
run_steps = defaultdict(list)

with open(NETLOGO_FILE, 'r', encoding='utf-8') as f:
    for _ in range(19):
        next(f)

    reader = csv.reader(f)
    row_count = 0

    for row in reader:
        if row_count % 100 == 0:
            print(f"    Processing row {row_count}...", flush=True)

        if row and row[0] and row[0].startswith('['):
            continue

        for run_idx in range(num_runs):
            col_offset = 1 + (run_idx * cols_per_run)

            if col_offset >= len(row):
                break

            try:
                step = int(float(row[col_offset]))
            except (ValueError, IndexError):
                continue

            var_values = {}
            for var_idx, var_name in enumerate(variables):
                try:
                    val_str = row[col_offset + 1 + var_idx]
                    if val_str:
                        var_values[var_name] = float(val_str)
                    else:
                        var_values[var_name] = np.nan
                except (ValueError, IndexError):
                    var_values[var_name] = np.nan

            run_num = run_idx + 1
            run_steps[run_num].append(step)
            for var_name, val in var_values.items():
                run_data[run_num][var_name].append(val)

        row_count += 1
        if row_count >= 1000:
            break

print(f"    Loaded {len(run_data)} runs")

# Step 3: Calculate first crossing times for each variable across all runs
print("\n[3] Calculating first crossing times...")
first_crossing_times = {var: [] for var in variables}

for run_num in run_data.keys():
    data = run_data[run_num]
    steps = np.array(run_steps[run_num])

    if len(steps) < 10:
        continue

    # Calculate std dev for each variable
    std_devs = {var: np.nanstd(np.array(data[var])) for var in variables}

    # Find first crossing for each variable
    for var in variables:
        if std_devs[var] == 0 or np.isnan(std_devs[var]):
            continue

        values = np.array(data[var])
        threshold = THRESHOLD_SD * std_devs[var]
        mask = np.abs(values) > threshold

        if mask.any():
            first_idx = np.argmax(mask)
            first_crossing_times[var].append(steps[first_idx])

print(f"    Calculated crossing times for all variables")

# Step 4: Create visualizations
print("\n[4] Creating visualizations...")

# Figure 1: Box plot of first crossing times
fig, ax = plt.subplots(figsize=(14, 8))

data_for_boxplot = []
labels_for_boxplot = []
for var in variables:
    if first_crossing_times[var]:
        data_for_boxplot.append(first_crossing_times[var])
        labels_for_boxplot.append(var.replace('_', ' ').replace('-', ' ').title())

bp = ax.boxplot(data_for_boxplot, labels=labels_for_boxplot,
                patch_artist=True, vert=False, widths=0.6)

# Color the boxes
colors = plt.cm.viridis(np.linspace(0, 1, len(data_for_boxplot)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Simulation Step (First Crossing of ±2 SD)', fontsize=12, fontweight='bold')
ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
ax.set_title(f'Distribution of First Crossing Times (±{THRESHOLD_SD} SD Threshold)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'first_crossing_boxplot.png', dpi=300, bbox_inches='tight')
print(f"    [SAVED] first_crossing_boxplot.png")

# Figure 2: Violin plot for more detail (ordered by first mover frequency)
fig, ax = plt.subplots(figsize=(14, 8))

# Calculate first mover frequency for ordering
first_mover_freq = {}
for var in variables:
    if first_crossing_times[var]:
        # Count how many times this var moved first
        count_first = 0
        for run_num in run_data.keys():
            data = run_data[run_num]
            steps_run = np.array(run_steps[run_num])
            if len(steps_run) < 10:
                continue

            std_devs = {v: np.nanstd(np.array(data[v])) for v in variables}
            first_times = {}
            for v in variables:
                if std_devs[v] == 0 or np.isnan(std_devs[v]):
                    first_times[v] = np.inf
                else:
                    values = np.array(data[v])
                    threshold = THRESHOLD_SD * std_devs[v]
                    mask = np.abs(values) > threshold
                    if mask.any():
                        first_times[v] = steps_run[np.argmax(mask)]
                    else:
                        first_times[v] = np.inf

            valid_times = {v: t for v, t in first_times.items() if t != np.inf}
            if valid_times and min(valid_times.values()) == first_times[var]:
                count_first += 1

        first_mover_freq[var] = count_first

# Sort variables by first mover frequency (ascending for bottom-to-top, so reverse for top-to-bottom display)
vars_sorted = sorted([v for v in variables if first_crossing_times[v]],
                     key=lambda v: first_mover_freq.get(v, 0), reverse=False)

positions = list(range(1, len(vars_sorted) + 1))
parts = ax.violinplot([first_crossing_times[var] for var in vars_sorted],
                       positions=positions, vert=False, widths=0.7,
                       showmeans=True, showmedians=True)

# Color the violins with gradient (most common = red, least = blue)
colors_violin = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(vars_sorted)))
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_violin[i])
    pc.set_alpha(0.7)

ax.set_yticks(positions)
ax.set_yticklabels([f"{var.replace('_', ' ').replace('-', ' ').title()} ({first_mover_freq.get(var, 0)} runs)"
                    for var in vars_sorted])
ax.set_xlabel('Simulation Step (First Crossing of ±2 SD)', fontsize=12, fontweight='bold')
ax.set_ylabel('Variable (First Mover Frequency)', fontsize=12, fontweight='bold')
ax.set_title(f'Distribution of First Crossing Times - Ordered by First Mover Frequency (±{THRESHOLD_SD} SD)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'first_crossing_violin.png', dpi=300, bbox_inches='tight')
print(f"    [SAVED] first_crossing_violin.png")

# Figure 3: Cumulative distribution - Line graph
fig, ax = plt.subplots(figsize=(14, 8))

max_step = max([max(times) if times else 0 for times in first_crossing_times.values()])
steps_range = np.arange(0, max_step + 1, 10)

for var in variables:
    if not first_crossing_times[var]:
        continue

    times = np.array(first_crossing_times[var])
    total_crossings = len(times)

    # Calculate cumulative percentage
    cumulative_pct = []
    for step in steps_range:
        count = np.sum(times <= step)
        cumulative_pct.append((count / total_crossings) * 100)

    ax.plot(steps_range, cumulative_pct, linewidth=2.5,
            label=var.replace('_', ' ').replace('-', ' ').title(), alpha=0.8)

ax.set_xlabel('Simulation Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative % of Runs Crossed', fontsize=12, fontweight='bold')
ax.set_title(f'Cumulative First Crossing Times (±{THRESHOLD_SD} SD Threshold)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_xlim(0, max_step)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'first_crossing_cumulative.png', dpi=300, bbox_inches='tight')
print(f"    [SAVED] first_crossing_cumulative.png")

# Figure 4: Mean first crossing time comparison
fig, ax = plt.subplots(figsize=(12, 8))

mean_times = []
var_labels = []
for var in variables:
    if first_crossing_times[var]:
        mean_times.append(np.mean(first_crossing_times[var]))
        var_labels.append(var.replace('_', ' ').replace('-', ' ').title())

# Sort by mean time
sorted_indices = np.argsort(mean_times)
mean_times_sorted = [mean_times[i] for i in sorted_indices]
var_labels_sorted = [var_labels[i] for i in sorted_indices]

colors_sorted = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(mean_times_sorted)))

bars = ax.barh(var_labels_sorted, mean_times_sorted, color=colors_sorted,
               edgecolor='black', linewidth=1.5)

ax.set_xlabel('Average First Crossing Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
ax.set_title(f'Average Time to First Cross ±{THRESHOLD_SD} SD Threshold',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, mean_times_sorted):
    ax.text(val, bar.get_y() + bar.get_height()/2,
           f'  {val:.1f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'average_first_crossing.png', dpi=300, bbox_inches='tight')
print(f"    [SAVED] average_first_crossing.png")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print(f"\nGenerated 4 visualizations:")
print(f"  1. first_crossing_boxplot.png - Box plot showing distribution")
print(f"  2. first_crossing_violin.png - Violin plot showing density")
print(f"  3. first_crossing_cumulative.png - Cumulative line graph")
print(f"  4. average_first_crossing.png - Bar chart of averages")
