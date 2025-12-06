"""
NetLogo Output Analysis - DoD Bureaucracy VECM Rank 2
======================================================
Analyzes the NetLogo simulation output to determine:
1. Which variable moves FIRST (initial cause)
2. Variable influences and trends
3. Key dynamics over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
NETLOGO_FILE = BASE_DIR / "netlogo" / "DoD_Bureaucracy_VECM_Rank2 DoW_bureaucracy-spreadsheet.csv"
OUTPUT_DIR = BASE_DIR / "netlogo_analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("NETLOGO SIMULATION ANALYSIS - DoD BUREAUCRACY VECM RANK 2")
print("=" * 80)

# Load data
print("\n[1] Loading NetLogo output...")
# NetLogo BehaviorSpace files have metadata header rows - skip to actual data
# Find the row where data starts (after all headers)
with open(NETLOGO_FILE, 'r') as f:
    for i, line in enumerate(f):
        if line.startswith('"[step]"') or line.startswith('[step]'):
            skiprows = i
            break
    else:
        # If no [step] found, try alternative patterns
        skiprows = 6  # Default for BehaviorSpace format

print(f"    Skipping {skiprows} header rows...")
df = pd.read_csv(NETLOGO_FILE, skiprows=skiprows)

print(f"    Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}")

# Identify variable columns (exclude metadata like run number, step)
metadata_cols = ['[run number]', '[step]', 'year', 'tick-count']
var_cols = [col for col in df.columns if col not in metadata_cols]

print(f"\n    Variables tracked: {len(var_cols)}")
for var in var_cols:
    print(f"      - {var}")

# ============================================================================
# ANALYSIS 1: WHICH VARIABLE MOVES FIRST?
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 1: WHICH VARIABLE MOVES FIRST (INITIAL CAUSE)?")
print("=" * 80)

# Define significant movement threshold
THRESHOLD = 1.0  # 1 standard deviation

# For each run, track when each variable first crosses threshold
if '[run number]' in df.columns:
    runs = df['[run number]'].unique()
    print(f"\n    Detected {len(runs)} simulation runs")

    first_movement = {var: [] for var in var_cols}

    for run in runs:
        run_data = df[df['[run number]'] == run].sort_values('[step]')

        for var in var_cols:
            # Find first time variable exceeds threshold
            mask = np.abs(run_data[var]) > THRESHOLD
            if mask.any():
                first_step = run_data.loc[mask, '[step]'].iloc[0]
                first_movement[var].append(first_step)
            else:
                first_movement[var].append(np.inf)  # Never crossed

    # Calculate average first movement time
    avg_first_movement = {var: np.mean([x for x in times if x != np.inf])
                          for var, times in first_movement.items()}

    # Sort by earliest movement
    sorted_vars = sorted(avg_first_movement.items(), key=lambda x: x[1])

    print(f"\n    Variables ranked by FIRST SIGNIFICANT MOVEMENT (>{THRESHOLD} SD):")
    print(f"    {'Rank':<6} {'Variable':<30} {'Avg Step':<15} {'% Runs Moved':<20}")
    print(f"    {'-'*75}")

    for rank, (var, avg_step) in enumerate(sorted_vars, 1):
        pct_moved = (1 - (first_movement[var].count(np.inf) / len(runs))) * 100
        if avg_step == np.inf:
            print(f"    {rank:<6} {var:<30} {'Never':<15} {pct_moved:<20.1f}%")
        else:
            print(f"    {rank:<6} {var:<30} {avg_step:<15.2f} {pct_moved:<20.1f}%")

    # Identify THE initial cause
    initial_cause = sorted_vars[0][0]
    initial_step = sorted_vars[0][1]

    print(f"\n    {'='*75}")
    print(f"    INITIAL CAUSE: {initial_cause}")
    print(f"    First significant movement at step: {initial_step:.2f}")
    print(f"    {'='*75}")

else:
    # Single run analysis
    print("\n    Single run detected - analyzing time to first movement...")

    first_movement_step = {}
    for var in var_cols:
        mask = np.abs(df[var]) > THRESHOLD
        if mask.any():
            first_movement_step[var] = df.loc[mask, '[step]'].iloc[0]
        else:
            first_movement_step[var] = np.inf

    sorted_vars = sorted(first_movement_step.items(), key=lambda x: x[1])

    print(f"\n    Variables ranked by FIRST SIGNIFICANT MOVEMENT (>{THRESHOLD} SD):")
    print(f"    {'Rank':<6} {'Variable':<30} {'Step':<15}")
    print(f"    {'-'*55}")

    for rank, (var, step) in enumerate(sorted_vars, 1):
        if step == np.inf:
            print(f"    {rank:<6} {var:<30} {'Never':<15}")
        else:
            print(f"    {rank:<6} {var:<30} {step:<15}")

    initial_cause = sorted_vars[0][0]
    initial_step = sorted_vars[0][1]

    print(f"\n    {'='*55}")
    print(f"    INITIAL CAUSE: {initial_cause}")
    print(f"    First movement at step: {initial_step}")
    print(f"    {'='*55}")

# ============================================================================
# ANALYSIS 2: VARIABLE DYNAMICS OVER TIME
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 2: VARIABLE DYNAMICS OVER TIME")
print("=" * 80)

# Use first run or aggregate if multiple runs
if '[run number]' in df.columns:
    # Average across all runs
    df_plot = df.groupby('[step]')[var_cols].mean().reset_index()
    print(f"\n    Averaging across {len(runs)} runs...")
else:
    df_plot = df.copy()

# Calculate when each variable reaches different thresholds
thresholds = [0.5, 1.0, 1.5, 2.0]
threshold_steps = {var: {t: None for t in thresholds} for var in var_cols}

for var in var_cols:
    for threshold in thresholds:
        mask = np.abs(df_plot[var]) > threshold
        if mask.any():
            threshold_steps[var][threshold] = df_plot.loc[mask, '[step]'].iloc[0]

print(f"\n    Time to reach thresholds (steps):")
print(f"    {'Variable':<30} {'0.5 SD':<12} {'1.0 SD':<12} {'1.5 SD':<12} {'2.0 SD':<12}")
print(f"    {'-'*80}")

for var in var_cols:
    steps = threshold_steps[var]
    print(f"    {var:<30} {str(steps[0.5]):<12} {str(steps[1.0]):<12} {str(steps[1.5]):<12} {str(steps[2.0]):<12}")

# ============================================================================
# VISUALIZATION 1: Time Series of All Variables
# ============================================================================
print(f"\n[2] Creating visualizations...")

fig, axes = plt.subplots(len(var_cols), 1, figsize=(14, 3*len(var_cols)), sharex=True)

if len(var_cols) == 1:
    axes = [axes]

for i, var in enumerate(var_cols):
    ax = axes[i]
    ax.plot(df_plot['[step]'], df_plot[var], linewidth=2, color='#2c3e50')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(THRESHOLD, color='red', linestyle='--', alpha=0.7, label=f'±{THRESHOLD} SD threshold')
    ax.axhline(-THRESHOLD, color='red', linestyle='--', alpha=0.7)

    # Mark first significant movement
    if var == initial_cause and initial_step != np.inf:
        ax.axvline(initial_step, color='green', linestyle=':', linewidth=2,
                   label='First significant movement', alpha=0.8)

    ax.set_ylabel(var.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('Simulation Step', fontsize=12, fontweight='bold')
fig.suptitle('NetLogo Simulation: Variable Trajectories Over Time',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'variable_trajectories.png', dpi=300, bbox_inches='tight')
print(f"    [SAVED] variable_trajectories.png")

# ============================================================================
# VISUALIZATION 2: First Movement Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data for first movement comparison
movement_data = []
for rank, (var, step) in enumerate(sorted_vars[:10], 1):  # Top 10
    if step != np.inf:
        movement_data.append({'Variable': var.replace('_', ' ').title(), 'Step': step, 'Rank': rank})

if movement_data:
    df_movement = pd.DataFrame(movement_data)

    bars = ax.barh(df_movement['Variable'], df_movement['Step'],
                   color=sns.color_palette('viridis', len(df_movement)))

    # Highlight the initial cause
    if len(df_movement) > 0:
        bars[0].set_color('#e74c3c')
        bars[0].set_linewidth(3)
        bars[0].set_edgecolor('black')

    ax.set_xlabel('Simulation Step (First Significant Movement)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
    ax.set_title('Which Variable Moves First? (Crosses ±1 SD Threshold)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    # Add values on bars
    for i, (idx, row) in enumerate(df_movement.iterrows()):
        ax.text(row['Step'], i, f"  {row['Step']:.1f}",
                va='center', fontsize=10, fontweight='bold')

    # Add annotation for initial cause
    if len(df_movement) > 0:
        ax.text(df_movement.iloc[0]['Step'] * 1.1, 0,
                'INITIAL CAUSE',
                va='center', fontsize=11, fontweight='bold',
                color='#e74c3c',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'first_movement_ranking.png', dpi=300, bbox_inches='tight')
print(f"    [SAVED] first_movement_ranking.png")

# ============================================================================
# VISUALIZATION 3: Variable Correlation Matrix
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 10))

# Calculate correlation matrix
corr_matrix = df_plot[var_cols].corr()

# Create heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            xticklabels=[v.replace('_', ' ').title() for v in var_cols],
            yticklabels=[v.replace('_', ' ').title() for v in var_cols],
            ax=ax)

ax.set_title('Variable Correlation Matrix (NetLogo Simulation)',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
print(f"    [SAVED] correlation_matrix.png")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print(f"\n[3] Generating summary report...")

with open(OUTPUT_DIR / 'NETLOGO_ANALYSIS_SUMMARY.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NETLOGO SIMULATION ANALYSIS - DoD BUREAUCRACY VECM RANK 2\n")
    f.write("=" * 80 + "\n\n")

    f.write("KEY FINDING: INITIAL CAUSE\n")
    f.write("-" * 80 + "\n")
    f.write(f"The variable that moves FIRST (crosses ±{THRESHOLD} SD threshold):\n\n")
    f.write(f"  >>> {initial_cause} <<<\n\n")

    if initial_step != np.inf:
        f.write(f"First significant movement occurs at step: {initial_step:.2f}\n\n")
    else:
        f.write(f"No variable crossed the {THRESHOLD} SD threshold.\n\n")

    f.write("INTERPRETATION:\n")
    f.write(f"In the VECM model dynamics, {initial_cause} is the primary driver.\n")
    f.write(f"Changes in this variable trigger cascading effects on other variables.\n")
    f.write(f"This suggests {initial_cause} acts as an exogenous shock or fundamental\n")
    f.write(f"forcing mechanism in the DoD bureaucratic system.\n\n")

    f.write("VARIABLE MOVEMENT RANKING:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Rank':<6} {'Variable':<35} {'First Movement (Step)':<25}\n")
    f.write("-" * 80 + "\n")

    for rank, (var, step) in enumerate(sorted_vars, 1):
        if step == np.inf:
            f.write(f"{rank:<6} {var:<35} {'Never crossed threshold':<25}\n")
        else:
            f.write(f"{rank:<6} {var:<35} {step:<25.2f}\n")

    f.write("\n")
    f.write("THRESHOLD CROSSING TIMES:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Variable':<35} {'0.5 SD':<12} {'1.0 SD':<12} {'1.5 SD':<12} {'2.0 SD':<12}\n")
    f.write("-" * 80 + "\n")

    for var in var_cols:
        steps = threshold_steps[var]
        f.write(f"{var:<35} {str(steps[0.5]):<12} {str(steps[1.0]):<12} {str(steps[1.5]):<12} {str(steps[2.0]):<12}\n")

    f.write("\n")
    f.write("CORRELATION INSIGHTS:\n")
    f.write("-" * 80 + "\n")
    f.write("Top 5 strongest positive correlations:\n")

    # Get top correlations
    corr_flat = []
    for i in range(len(var_cols)):
        for j in range(i+1, len(var_cols)):
            corr_flat.append((var_cols[i], var_cols[j], corr_matrix.iloc[i, j]))

    corr_sorted = sorted(corr_flat, key=lambda x: abs(x[2]), reverse=True)

    for i, (var1, var2, corr) in enumerate(corr_sorted[:5], 1):
        f.write(f"  {i}. {var1} <-> {var2}: {corr:+.3f}\n")

    f.write("\nTop 5 strongest negative correlations:\n")
    corr_sorted_neg = sorted(corr_flat, key=lambda x: x[2])

    for i, (var1, var2, corr) in enumerate(corr_sorted_neg[:5], 1):
        f.write(f"  {i}. {var1} <-> {var2}: {corr:+.3f}\n")

print(f"    [SAVED] NETLOGO_ANALYSIS_SUMMARY.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nKey Finding:")
print(f"  INITIAL CAUSE: {initial_cause}")
if initial_step != np.inf:
    print(f"  First movement at step: {initial_step:.2f}")
print(f"\nFiles generated:")
print(f"  1. variable_trajectories.png")
print(f"  2. first_movement_ranking.png")
print(f"  3. correlation_matrix.png")
print(f"  4. NETLOGO_ANALYSIS_SUMMARY.txt")
