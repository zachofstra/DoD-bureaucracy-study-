"""
NetLogo BehaviorSpace First Mover Analysis - Time Series Parser
================================================================
Efficiently parses 302MB BehaviorSpace file to find which variable
crosses ±2 SD threshold first across all runs.
"""

import csv
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import sys

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
NETLOGO_FILE = BASE_DIR / "netlogo" / "DoD_Bureaucracy_VECM_Rank2 DoW_bureaucracy-spreadsheet_3"
OUTPUT_DIR = BASE_DIR / "netlogo_first_mover_analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

THRESHOLD_SD = 0.25

print("=" * 80)
print("NETLOGO FIRST MOVER ANALYSIS - TIME SERIES PARSER (302MB FILE)")
print("=" * 80)

# Step 1: Parse header to understand structure
print("\n[1] Parsing header...")
with open(NETLOGO_FILE, 'r', encoding='utf-8') as f:
    header_lines = [next(f).strip() for _ in range(15)]

# Parse run numbers (row 7, index 6)
run_numbers_raw = header_lines[6].replace('"', '').split(',')
run_numbers = [int(x) if x else None for x in run_numbers_raw[1:]]  # Skip [run number] label

# Parse parameters
var_strength_raw = header_lines[7].replace('"', '').split(',')
noise_level_raw = header_lines[8].replace('"', '').split(',')
error_corr_raw = header_lines[10].replace('"', '').split(',')

var_strength = [float(x) if x else None for x in var_strength_raw[1:]]
noise_level = [float(x) if x else None for x in noise_level_raw[1:]]
error_corr = [float(x) if x else None for x in error_corr_raw[1:]]

# Parse variable names (row 12, index 11)
reporter_row = header_lines[11].replace('"', '').split(',')

# Extract variables (skip [step] columns)
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

# Determine column structure
cols_per_run = len(variables) + 1  # +1 for [step]

# Count runs
num_runs = 0
for i, rn in enumerate(run_numbers):
    if rn is not None and rn > num_runs:
        num_runs = rn

print(f"    Total runs: {num_runs}")
print(f"    Columns per run: {cols_per_run}")

# Step 2: Read time series data and build run histories
print("\n[2] Reading time series data (this will take a few minutes)...")

# Store data for each run: {run_num: {var_name: [values]}}
run_data = defaultdict(lambda: {var: [] for var in variables})
run_steps = defaultdict(list)

with open(NETLOGO_FILE, 'r', encoding='utf-8') as f:
    # Skip header (rows 1-19: metadata, parameter rows, summary stats, column headers)
    for _ in range(19):
        next(f)

    reader = csv.reader(f)
    row_count = 0

    for row in reader:
        if row_count % 100 == 0:
            print(f"    Processing time step row {row_count}...", flush=True)

        # Skip any remaining summary rows
        if row and row[0] and row[0].startswith('['):
            continue

        # Parse each run's data from this time step
        for run_idx in range(num_runs):
            col_offset = 1 + (run_idx * cols_per_run)  # +1 to skip [reporter] column

            if col_offset >= len(row):
                break

            # Get step number
            try:
                step_str = row[col_offset]
                if not step_str:
                    continue
                step = int(float(step_str))
            except (ValueError, IndexError):
                continue

            # Get variable values
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

            # Store
            run_num = run_idx + 1
            run_steps[run_num].append(step)
            for var_name, val in var_values.items():
                run_data[run_num][var_name].append(val)

        row_count += 1

        # Limit to reasonable number of rows for testing
        if row_count >= 1000:
            break

print(f"    Loaded {len(run_data)} runs with up to {row_count} time steps each")

# Step 3: Calculate first mover for each run
print("\n[3] Calculating first mover for each run...")

# Diagnostic: Check runs with different parameters
print(f"\n    DIAGNOSTIC - Sample run data:")
for check_run in [1, 4, 28]:  # Run 1 (all 0s), Run 4 (noise 0.25), Run 28 (var 2.5, noise 0.25)
    if check_run in run_data:
        all_vals = np.array(run_data[check_run]['junior-enlisted'])
        print(f"\n      Run {check_run} (params: var={var_strength[check_run-1]}, noise={noise_level[check_run-1]}, err_cor={error_corr[check_run-1]}):")
        print(f"        Total steps: {len(all_vals)}")
        print(f"        Steps 0-9: {all_vals[:10]}")
        print(f"        Steps 990-999: {all_vals[990:1000] if len(all_vals) > 990 else 'N/A'}")
        print(f"        Stats: mean={np.nanmean(all_vals):.6f}, std={np.nanstd(all_vals):.6f}, min={np.nanmin(all_vals):.6f}, max={np.nanmax(all_vals):.6f}")

first_mover_results = []

for run_num in sorted(run_data.keys()):
    if run_num % 10 == 0:
        print(f"    Analyzing run {run_num}/{num_runs}...", flush=True)

    data = run_data[run_num]
    steps = np.array(run_steps[run_num])

    if len(steps) < 10:
        continue

    # Calculate standard deviation for each variable
    std_devs = {}
    for var in variables:
        values = np.array(data[var])
        std_devs[var] = np.nanstd(values)

    # Find first crossing for each variable
    first_crossing = {}
    for var in variables:
        if std_devs[var] == 0 or np.isnan(std_devs[var]):
            first_crossing[var] = np.inf
            continue

        values = np.array(data[var])
        threshold = THRESHOLD_SD * std_devs[var]
        mask = np.abs(values) > threshold

        if mask.any():
            first_idx = np.argmax(mask)
            first_crossing[var] = steps[first_idx]
        else:
            first_crossing[var] = np.inf

    # Find first mover
    valid_crossings = {v: t for v, t in first_crossing.items() if t != np.inf}

    if valid_crossings:
        first_mover = min(valid_crossings, key=valid_crossings.get)

        # Get parameters for this run
        param_idx = run_num - 1
        if param_idx < len(var_strength):
            params = {
                'var_strength': var_strength[param_idx],
                'noise_level': noise_level[param_idx],
                'error_correction': error_corr[param_idx]
            }
        else:
            params = {}

        first_mover_results.append({
            'run': run_num,
            'first_mover': first_mover,
            'first_step': valid_crossings[first_mover],
            'all_crossings': first_crossing,
            'params': params
        })

print(f"    Successfully analyzed {len(first_mover_results)} runs")

# Step 4: Aggregate and report results
print("\n[4] Aggregating results...")
first_mover_counts = Counter([r['first_mover'] for r in first_mover_results])
sorted_first_movers = first_mover_counts.most_common()

print(f"\n{'='*80}")
print("RESULTS: FIRST MOVER FREQUENCY")
print(f"{'='*80}")

total_runs = len(first_mover_results)
for rank, (var, count) in enumerate(sorted_first_movers, 1):
    pct = (count / total_runs) * 100
    print(f"{rank}. {var:40s} {count:4d} runs ({pct:5.1f}%)")

if sorted_first_movers:
    most_common = sorted_first_movers[0]
    print(f"\n{'='*80}")
    print(f"PRIMARY INITIAL CAUSE: {most_common[0]}")
    print(f"Frequency: {most_common[1]}/{total_runs} runs ({most_common[1]/total_runs*100:.1f}%)")
    print(f"{'='*80}")

# Step 5: Calculate average first movement times
print("\n[5] Average first movement times...")
avg_times = {}
for var in variables:
    times = [r['all_crossings'][var] for r in first_mover_results
             if r['all_crossings'][var] != np.inf]
    if times:
        avg_times[var] = {'mean': np.mean(times), 'median': np.median(times), 'count': len(times)}

sorted_by_time = sorted(avg_times.items(), key=lambda x: x[1]['mean'])
for var, stats in sorted_by_time:
    pct = (stats['count'] / total_runs) * 100
    print(f"  {var:40s} Avg: {stats['mean']:6.1f}  Median: {stats['median']:6.1f}  ({pct:5.1f}% of runs)")

# Step 6: Analyze by parameter configuration
print("\n[6] Analyzing by parameter configuration...")
by_params = defaultdict(list)
for r in first_mover_results:
    if r['params']:
        key = (r['params'].get('var_strength'),
               r['params'].get('noise_level'),
               r['params'].get('error_correction'))
        by_params[key].append(r['first_mover'])

print(f"\n  First mover by parameter configuration:")
print(f"  {'Var':<6} {'Noise':<6} {'ErrCor':<6} {'Most Common First Mover':<40} {'Count':<10}")
print(f"  {'-'*75}")
for (vs, nl, ec), first_movers in sorted(by_params.items(), key=lambda x: (x[0][0] or 0, x[0][1] or 0, x[0][2] or 0)):
    if vs is not None:
        counts = Counter(first_movers)
        top_fm = counts.most_common(1)[0]
        print(f"  {vs:<6.1f} {nl:<6.2f} {ec:<6.1f} {top_fm[0]:<40s} {top_fm[1]:<10d}")

# Step 7: Save summary
print("\n[7] Saving summary...")
with open(OUTPUT_DIR / 'FIRST_MOVER_SUMMARY.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NETLOGO FIRST MOVER ANALYSIS - TIME SERIES DATA\n")
    f.write("=" * 80 + "\n\n")

    if sorted_first_movers:
        f.write(f"PRIMARY INITIAL CAUSE: {most_common[0]}\n")
        f.write(f"Moves first in {most_common[1]}/{total_runs} runs ({most_common[1]/total_runs*100:.1f}%)\n\n")

    f.write("FREQUENCY RANKING:\n")
    f.write("-" * 80 + "\n")
    for rank, (var, count) in enumerate(sorted_first_movers, 1):
        pct = (count / total_runs) * 100
        f.write(f"{rank:3d}. {var:40s} {count:4d} ({pct:5.1f}%)\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("AVERAGE FIRST MOVEMENT TIMES:\n")
    f.write("-" * 80 + "\n")
    for var, stats in sorted_by_time:
        pct = (stats['count'] / total_runs) * 100
        f.write(f"{var:40s} Avg: {stats['mean']:7.1f}  Median: {stats['median']:7.1f}  ({stats['count']:3d} runs, {pct:5.1f}%)\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("BY PARAMETER CONFIGURATION:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Var Str':<8} {'Noise':<8} {'Err Cor':<8} {'Most Common First Mover':<40} {'Count':<10}\n")
    f.write("-" * 80 + "\n")
    for (vs, nl, ec), first_movers in sorted(by_params.items(), key=lambda x: (x[0][0] or 0, x[0][1] or 0, x[0][2] or 0)):
        if vs is not None:
            counts = Counter(first_movers)
            top_fm = counts.most_common(1)[0]
            f.write(f"{vs:<8.1f} {nl:<8.2f} {ec:<8.1f} {top_fm[0]:<40s} {top_fm[1]:<10d}\n")

print(f"    [SAVED] FIRST_MOVER_SUMMARY.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
if sorted_first_movers:
    print(f"\nPRIMARY FINDING: {most_common[0]}")
    print(f"This variable moves ±{THRESHOLD_SD} SD first in {most_common[1]/total_runs*100:.1f}% of runs")
