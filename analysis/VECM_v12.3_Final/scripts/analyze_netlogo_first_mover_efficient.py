"""
NetLogo BehaviorSpace Analysis - First Mover Detection (Memory Efficient)
==========================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
NETLOGO_FILE = BASE_DIR / "netlogo" / "DoD_Bureaucracy_VECM_Rank2 DoW_bureaucracy-spreadsheet.csv"
OUTPUT_DIR = BASE_DIR / "netlogo_first_mover_analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

THRESHOLD_SD = 2.0

print("=" * 80)
print("NETLOGO FIRST MOVER ANALYSIS - MEMORY EFFICIENT VERSION")
print("=" * 80)

# Find header row
print("\n[1] Finding header row...")
with open(NETLOGO_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        if '[step]' in line.lower():
            header_row = i
            print(f"    Header at row {i}")
            break

# Read just the header to get column names
print("\n[2] Reading column names...")
df_header = pd.read_csv(NETLOGO_FILE, skiprows=header_row, nrows=0)
cols = list(df_header.columns)
print(f"    Found {len(cols)} columns")

# Identify variable columns
metadata_cols = ['[run number]', '[step]', 'year', 'tick-count',
                'error-correction-strength', 'var-strength', 'noise-level', 'max-year']
var_cols = [col for col in cols if col not in metadata_cols]
print(f"    Variables: {len(var_cols)}")
for v in var_cols[:5]:
    print(f"      - {v}")
if len(var_cols) > 5:
    print(f"      ... and {len(var_cols)-5} more")

# Read data in chunks and process by run
print("\n[3] Processing data by run (reading in chunks)...")

chunk_size = 10000
run_data = defaultdict(list)

for chunk in pd.read_csv(NETLOGO_FILE, skiprows=header_row, chunksize=chunk_size,
                          low_memory=False):
    print(f"    Processing chunk of {len(chunk)} rows...")

    for run_num in chunk['[run number]'].unique():
        run_rows = chunk[chunk['[run number]'] == run_num]
        run_data[run_num].append(run_rows)

print(f"    Found {len(run_data)} runs total")

# Process each run
print("\n[4] Analyzing first mover for each run...")
first_mover_results = []

for run_num, chunks in run_data.items():
    # Combine chunks for this run
    run_df = pd.concat(chunks, ignore_index=True)

    if len(run_df) < 10:
        continue

    # Calculate std dev for each variable
    std_devs = {var: run_df[var].std() for var in var_cols}

    # Find first crossing for each variable
    first_crossing = {}
    for var in var_cols:
        if std_devs[var] == 0 or np.isnan(std_devs[var]):
            first_crossing[var] = np.inf
            continue

        threshold = THRESHOLD_SD * std_devs[var]
        mask = np.abs(run_df[var]) > threshold

        if mask.any():
            first_crossing[var] = run_df.loc[mask, '[step]'].iloc[0]
        else:
            first_crossing[var] = np.inf

    # Find first mover
    valid_crossings = {v: t for v, t in first_crossing.items() if t != np.inf}

    if valid_crossings:
        first_mover = min(valid_crossings, key=valid_crossings.get)
        first_mover_results.append({
            'run': run_num,
            'first_mover': first_mover,
            'first_tick': valid_crossings[first_mover],
            'all_crossings': first_crossing
        })

    if run_num % 50 == 0:
        print(f"      Processed {run_num} runs...")

print(f"    Successfully analyzed {len(first_mover_results)} runs")

# Aggregate results
print("\n[5] Aggregating results...")
first_mover_counts = Counter([r['first_mover'] for r in first_mover_results])
sorted_first_movers = first_mover_counts.most_common()

# Display results
print(f"\n{'='*80}")
print("RESULTS: FIRST MOVER FREQUENCY")
print(f"{'='*80}")

total_runs = len(first_mover_results)
for rank, (var, count) in enumerate(sorted_first_movers, 1):
    pct = (count / total_runs) * 100
    print(f"{rank}. {var:40s} {count:4d} runs ({pct:5.1f}%)")

most_common = sorted_first_movers[0]
print(f"\n{'='*80}")
print(f"PRIMARY INITIAL CAUSE: {most_common[0]}")
print(f"Frequency: {most_common[1]}/{total_runs} runs ({most_common[1]/total_runs*100:.1f}%)")
print(f"{'='*80}")

# Calculate average times
print("\n[6] Calculating average first movement times...")
avg_times = {}
for var in var_cols:
    times = [r['all_crossings'][var] for r in first_mover_results
             if r['all_crossings'][var] != np.inf]
    if times:
        avg_times[var] = {'mean': np.mean(times), 'count': len(times)}

sorted_by_time = sorted(avg_times.items(), key=lambda x: x[1]['mean'])
for var, stats in sorted_by_time[:10]:
    pct = (stats['count'] / total_runs) * 100
    print(f"  {var:40s} Avg tick: {stats['mean']:6.1f} ({pct:5.1f}% of runs)")

# Save visualizations
print("\n[7] Creating visualizations...")
fig, ax = plt.subplots(figsize=(14, 10))

vars_sorted = [var for var, count in sorted_first_movers[:15]]  # Top 15
counts_sorted = [count for var, count in sorted_first_movers[:15]]
colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(vars_sorted))]

bars = ax.barh(vars_sorted, counts_sorted, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Number of Runs Where Variable Moved First', fontsize=12, fontweight='bold')
ax.set_title(f'Which Variable Moves First? (±{THRESHOLD_SD} SD Threshold)\n'
             f'Across {total_runs} NetLogo Runs',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

for bar, count in zip(bars, counts_sorted):
    pct = (count / total_runs) * 100
    ax.text(count, bar.get_y() + bar.get_height()/2,
           f'  {count} ({pct:.1f}%)',
           va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'first_mover_frequency.png', dpi=300, bbox_inches='tight')
print(f"    [SAVED] first_mover_frequency.png")

# Save summary
print("\n[8] Saving summary...")
with open(OUTPUT_DIR / 'FIRST_MOVER_SUMMARY.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NETLOGO FIRST MOVER ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

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
        f.write(f"{var:40s} Tick {stats['mean']:7.1f} ({stats['count']:3d} runs, {pct:5.1f}%)\n")

print(f"    [SAVED] FIRST_MOVER_SUMMARY.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nPRIMARY FINDING: {most_common[0]}")
print(f"This variable moves ±{THRESHOLD_SD} SD first in {most_common[1]/total_runs*100:.1f}% of runs")
