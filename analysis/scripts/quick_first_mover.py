import pandas as pd
import numpy as np
from collections import Counter

print("Quick First Mover Analysis (Sampled)")
print("=" * 60)

# Read with sampling
print("\nSampling every 100th row to speed up analysis...")
df = pd.read_csv(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final\netlogo\DoD_Bureaucracy_VECM_Rank2 DoW_bureaucracy-spreadsheet.csv",
                 skiprows=range(1, 7))  # Skip header metadata

print(f"Shape: {df.shape}")

# Take only first N rows to make it manageable
df = df.head(50000)
print(f"Using first 50k rows")

# Identify variables
var_cols = [c for c in df.columns if c not in [
    '[run number]', '[step]', 'year', 'tick-count',
    'error-correction-strength', 'var-strength', 'noise-level', 'max-year'
]]

print(f"\nVariables: {var_cols[:5]}... ({len(var_cols)} total)")

# Analyze first 50 runs
runs = sorted(df['[run number]'].unique())[:50]
print(f"\nAnalyzing {len(runs)} runs...")

first_movers = []
for run in runs:
    run_df = df[df['[run number]'] == run]
    if len(run_df) < 5:
        continue

    std_devs = {v: run_df[v].std() for v in var_cols}

    first_cross = {}
    for v in var_cols:
        if std_devs[v] > 0 and not np.isnan(std_devs[v]):
            mask = np.abs(run_df[v]) > 2 * std_devs[v]
            if mask.any():
                first_cross[v] = run_df.loc[mask, '[step]'].iloc[0]

    if first_cross:
        winner = min(first_cross, key=first_cross.get)
        first_movers.append(winner)
        if run % 10 == 0:
            print(f"  Run {run}: {winner} at tick {first_cross[winner]}")

print(f"\n{'='*60}")
print("RESULTS (Sample of {len(first_movers)} runs)")
print(f"{'='*60}")

counts = Counter(first_movers)
for var, count in counts.most_common():
    pct = count / len(first_movers) * 100
    print(f"{var:40s} {count:3d} runs ({pct:5.1f}%)")

print(f"\n{'='*60}")
print(f"PRIMARY INITIAL CAUSE (in sample): {counts.most_common()[0][0]}")
print(f"{'='*60}")
