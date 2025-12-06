"""
NetLogo BehaviorSpace First Mover Analysis - Custom Parser
===========================================================
Efficiently parses BehaviorSpace "spreadsheet" format to find
which variable crosses ±2 SD threshold first across runs.
"""

import csv
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
NETLOGO_FILE = BASE_DIR / "netlogo" / "DoD_Bureaucracy_VECM_Rank2 DoW_bureaucracy-spreadsheet.csv"
OUTPUT_DIR = BASE_DIR / "netlogo_first_mover_analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

THRESHOLD_SD = 2.0
MAX_RUNS_TO_ANALYZE = 1000  # Analyze first 1000 runs for speed

print("=" * 80)
print("NETLOGO FIRST MOVER ANALYSIS - CUSTOM BEHAVIORSPACE PARSER")
print("=" * 80)

# Step 1: Read header to understand structure
print("\n[1] Parsing BehaviorSpace header...")
with open(NETLOGO_FILE, 'r', encoding='utf-8') as f:
    lines = []
    for i in range(20):
        lines.append(next(f).strip())

    # Find run numbers row (row 7, index 6)
    run_numbers_row = lines[6].replace('"', '').split(',')

    # Find header row (row 13, index 12)
    header_row = lines[12].replace('"', '').split(',')

    # Find variables (skip [final value], [step] columns)
    # Pattern: [final value], [step], var1, var2, ..., var8, [step], var1, var2, ..., var8
    variables = []
    i = 1  # Skip [final value]
    while i < len(header_row) and len(variables) < 8:
        col_name = header_row[i]
        if col_name == '[step]':
            i += 1
            continue
        if col_name and col_name not in variables:
            variables.append(col_name)
        i += 1

    print(f"    Variables found: {variables}")
    print(f"    Variables per run: {len(variables)}")

    # Determine how many columns per run
    # Pattern: [step] + 8 variables = 9 columns per run
    cols_per_run = len(variables) + 1  # +1 for [step]

    # Count runs in file
    total_cols = len(header_row)
    estimated_runs = (total_cols - 1) // cols_per_run  # -1 for [final value]

    print(f"    Estimated total runs: {estimated_runs}")
    print(f"    Analyzing first {min(MAX_RUNS_TO_ANALYZE, estimated_runs)} runs")

# Step 2: Parse data for each run
print("\n[2] Parsing run data (this may take a moment)...")

runs_to_analyze = min(MAX_RUNS_TO_ANALYZE, estimated_runs)

# Read data starting from row 14 (index 13)
run_data = defaultdict(list)  # {run_num: {var_name: [values]}}

with open(NETLOGO_FILE, 'r', encoding='utf-8') as f:
    # Skip to data rows
    for _ in range(13):
        next(f)

    reader = csv.reader(f)
    row_count = 0

    for row in reader:
        if row_count % 100 == 0 and row_count > 0:
            print(f"    Processing row {row_count}...")

        # Parse each run's data from this row
        for run_idx in range(runs_to_analyze):
            col_offset = 1 + (run_idx * cols_per_run)  # +1 to skip [final value]

            if col_offset >= len(row):
                break

            # Get step number
            try:
                step = int(float(row[col_offset]))
            except (ValueError, IndexError):
                continue

            # Get variable values
            var_values = {}
            for var_idx, var_name in enumerate(variables):
                try:
                    val = float(row[col_offset + 1 + var_idx])
                    var_values[var_name] = val
                except (ValueError, IndexError):
                    var_values[var_name] = np.nan

            # Store
            if run_idx not in run_data:
                run_data[run_idx] = {var: [] for var in variables}
                run_data[run_idx]['steps'] = []

            run_data[run_idx]['steps'].append(step)
            for var_name, val in var_values.items():
                run_data[run_idx][var_name].append(val)

        row_count += 1

        # Stop after reasonable number of rows
        if row_count >= 1000:
            break

print(f"    Loaded {len(run_data)} runs with {row_count} time steps each")

# Step 3: Calculate first mover for each run
print("\n[3] Calculating first mover for each run...")

first_mover_results = []

for run_idx in sorted(run_data.keys()):
    data = run_data[run_idx]
    steps = np.array(data['steps'])

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
        first_mover_results.append({
            'run': run_idx,
            'first_mover': first_mover,
            'first_step': valid_crossings[first_mover],
            'all_crossings': first_crossing
        })

    if run_idx % 100 == 0:
        print(f"    Processed run {run_idx}...")

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
        avg_times[var] = {'mean': np.mean(times), 'count': len(times)}

sorted_by_time = sorted(avg_times.items(), key=lambda x: x[1]['mean'])
for var, stats in sorted_by_time:
    pct = (stats['count'] / total_runs) * 100
    print(f"  {var:40s} Avg step: {stats['mean']:6.1f} ({pct:5.1f}% of runs)")

# Step 6: Save summary
print("\n[6] Saving summary...")
with open(OUTPUT_DIR / 'FIRST_MOVER_SUMMARY.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NETLOGO FIRST MOVER ANALYSIS\n")
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
        f.write(f"{var:40s} Step {stats['mean']:7.1f} ({stats['count']:3d} runs, {pct:5.1f}%)\n")

print(f"    [SAVED] FIRST_MOVER_SUMMARY.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
if sorted_first_movers:
    print(f"\nPRIMARY FINDING: {most_common[0]}")
    print(f"This variable moves ±{THRESHOLD_SD} SD first in {most_common[1]/total_runs*100:.1f}% of runs")
