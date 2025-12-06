"""
Johansen Cointegration Test Tutorial
Shows how to read and interpret Johansen test results for VECM v12.3 variables
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("JOHANSEN COINTEGRATION TEST TUTORIAL")
print("Using your 8 selected VECM variables")
print("=" * 100)

# Load data
df = pd.read_excel('analysis/complete_normalized_dataset_v12.3.xlsx')

# Your 8 selected variables for VECM v12.3
SELECTED_VARS = [
    'Junior_Enlisted_Z',
    'Company_Grade_Officers_Z',
    'Field_Grade_Officers_Z',
    'GOFOs_Z',
    'Warrant_Officers_Z',
    'Policy_Count_Log',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z'
]

data = df[SELECTED_VARS].dropna()

print(f"\nVariables: {len(SELECTED_VARS)}")
print(f"Observations: {len(data)}\n")

for i, var in enumerate(SELECTED_VARS, 1):
    print(f"  {i}. {var}")

# Run Johansen test
print("\n" + "=" * 100)
print("RUNNING JOHANSEN TEST")
print("=" * 100)
print("\nParameters:")
print("  det_order=0  (no deterministic trend in cointegration relation)")
print("  k_ar_diff=1  (lag order = 1 for VECM, equivalent to VAR(2) in levels)")

joh_result = coint_johansen(data, det_order=0, k_ar_diff=1)

# Extract results
trace_stats = joh_result.trace_stat
trace_crit_90 = joh_result.trace_stat_crit_vals[:, 0]  # 90% critical values
trace_crit_95 = joh_result.trace_stat_crit_vals[:, 1]  # 95% critical values
trace_crit_99 = joh_result.trace_stat_crit_vals[:, 2]  # 99% critical values

max_eig_stats = joh_result.max_eig_stat
max_eig_crit_90 = joh_result.max_eig_stat_crit_vals[:, 0]
max_eig_crit_95 = joh_result.max_eig_stat_crit_vals[:, 1]
max_eig_crit_99 = joh_result.max_eig_stat_crit_vals[:, 2]

eigenvalues = joh_result.eig

# Display results
print("\n" + "=" * 100)
print("1. TRACE STATISTIC TEST")
print("=" * 100)
print("\nTests H0: 'At most r cointegration relationships'\n")
print(f"{'Rank':<6} {'Eigenvalue':<12} {'Trace Stat':<12} {'90% Crit':<12} {'95% Crit':<12} {'99% Crit':<12} {'Decision (95%)':<20}")
print("-" * 100)

trace_rank = 0
for r in range(len(SELECTED_VARS)):
    decision = "REJECT H0 (r > " + str(r) + ")" if trace_stats[r] > trace_crit_95[r] else f"Accept H0 (r <= {r})"

    # Highlight where we stop rejecting
    if trace_stats[r] > trace_crit_95[r]:
        trace_rank = r + 1
        marker = " ***"
    else:
        marker = ""

    print(f"{r:<6} {eigenvalues[r]:<12.4f} {trace_stats[r]:<12.2f} {trace_crit_90[r]:<12.2f} {trace_crit_95[r]:<12.2f} {trace_crit_99[r]:<12.2f} {decision:<20}{marker}")

print(f"\nTRACE TEST CONCLUSION: Cointegration rank = {trace_rank}")
print(f"  -> There are {trace_rank} independent long-run equilibrium relationships")

print("\n" + "=" * 100)
print("2. MAX EIGENVALUE STATISTIC TEST")
print("=" * 100)
print("\nTests H0: 'Exactly r cointegration relationships'\n")
print(f"{'Rank':<6} {'Eigenvalue':<12} {'MaxEig Stat':<12} {'90% Crit':<12} {'95% Crit':<12} {'99% Crit':<12} {'Decision (95%)':<20}")
print("-" * 100)

max_eig_rank = 0
for r in range(len(SELECTED_VARS)):
    decision = "REJECT H0 (r != " + str(r) + ")" if max_eig_stats[r] > max_eig_crit_95[r] else f"Accept H0 (r = {r})"

    if max_eig_stats[r] > max_eig_crit_95[r]:
        max_eig_rank = r + 1
        marker = " ***"
    else:
        marker = ""

    print(f"{r:<6} {eigenvalues[r]:<12.4f} {max_eig_stats[r]:<12.2f} {max_eig_crit_90[r]:<12.2f} {max_eig_crit_95[r]:<12.2f} {max_eig_crit_99[r]:<12.2f} {decision:<20}{marker}")

print(f"\nMAX EIGENVALUE TEST CONCLUSION: Cointegration rank = {max_eig_rank}")
print(f"  -> There are {max_eig_rank} independent long-run equilibrium relationships")

# Interpretation
print("\n" + "=" * 100)
print("3. HOW TO READ THESE RESULTS")
print("=" * 100)

print("""
STEP-BY-STEP INTERPRETATION:

1. START AT r=0 (first row):
   - Look at Trace Statistic vs 95% Critical Value
   - If Trace Stat > 95% Crit -> REJECT H0 -> There IS cointegration (r > 0)
   - Move to next row

2. CONTINUE SEQUENTIALLY:
   - Keep going down rows while Trace Stat > 95% Crit
   - Each rejection means "at least one more cointegration relationship"
   - STOP when Trace Stat < 95% Crit (first time you DON'T reject)

3. COINTEGRATION RANK:
   - The rank = number of times you rejected H0
   - Example: If you reject r=0,1,2,3,4,5 but accept r=6 -> rank = 6

4. WHAT RANK MEANS:
   - Rank = 0: NO cointegration -> Use VAR in first differences (not VECM)
   - Rank = 1-2: Weak cointegration -> VECM possible but simple
   - Rank = k-2 to k-1: Strong cointegration -> IDEAL for VECM
   - Rank = k (full rank): TOO MUCH cointegration -> May indicate issues
     (All variables stationary, or overfitting with small sample)
""")

print("\n" + "=" * 100)
print("4. YOUR SPECIFIC RESULTS")
print("=" * 100)

print(f"""
Your Data (8 variables, {len(data)} observations):

TRACE TEST: Rank = {trace_rank}
MAX EIGENVALUE TEST: Rank = {max_eig_rank}

""")

if trace_rank == max_eig_rank:
    print(f"Both tests agree on rank = {trace_rank} [OK]")
else:
    print(f"Tests disagree: Trace={trace_rank}, MaxEig={max_eig_rank}")
    print(f"  -> Common to differ by 1-2")
    print(f"  -> Trace test is more commonly used")
    print(f"  -> RECOMMENDATION: Use rank = {trace_rank} (trace test)")

print(f"\nINTERPRETATION:")
if trace_rank == 0:
    print("  NO cointegration detected -> Use VAR in first differences")
elif trace_rank <= 2:
    print(f"  Weak cointegration ({trace_rank} relationships)")
    print("  -> VECM possible but simple structure")
elif trace_rank >= len(SELECTED_VARS) - 1:
    print(f"  Very high rank ({trace_rank} out of {len(SELECTED_VARS)} possible)")
    if trace_rank == len(SELECTED_VARS):
        print("  -> FULL RANK: May indicate all variables are stationary")
        print("  -> Check stationarity tests (ADF) carefully")
        print("  -> Consider reducing rank to k-1 or k-2 for stable estimation")
    else:
        print("  -> Strong long-run relationships")
        print("  -> Good for VECM but leaves little room for error correction")
        print("  -> Consider using rank = k-2 for more stable estimation")
else:
    print(f"  Moderate cointegration ({trace_rank} relationships)")
    print("  -> IDEAL for VECM analysis")
    print("  -> Strong long-run structure with room for error correction")

# Eigenvalue analysis
print(f"\n" + "=" * 100)
print("5. EIGENVALUE ANALYSIS")
print("=" * 100)
print("\nEigenvalues measure the strength of each cointegration relationship:")
print(f"\n{'Rank':<6} {'Eigenvalue':<12} {'Strength':<15} {'Interpretation'}")
print("-" * 80)

for r in range(trace_rank if trace_rank > 0 else len(SELECTED_VARS)):
    eig = eigenvalues[r]
    if eig > 0.5:
        strength = "VERY STRONG"
    elif eig > 0.3:
        strength = "STRONG"
    elif eig > 0.1:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    if r < trace_rank:
        interp = f"Significant cointegration relationship #{r+1}"
    else:
        interp = "Below cointegration rank (not significant)"

    print(f"{r:<6} {eig:<12.4f} {strength:<15} {interp}")

# Recommendations
print("\n" + "=" * 100)
print("6. CHOOSING THE RIGHT RANK FOR YOUR VECM")
print("=" * 100)

print(f"""
STATISTICAL TESTS SUGGEST: rank = {trace_rank}

PRACTICAL CONSIDERATIONS:

1. Sample Size (n={len(data)}):
   - Small sample -> Lower rank more stable
   - Rule of thumb: rank < n/5 = {len(data)//5}
   - Your rank {trace_rank}: {'OK [X]' if trace_rank < len(data)//5 else 'May be too high [!]'}

2. Degrees of Freedom:
   - Each cointegration relationship uses parameters
   - Higher rank -> fewer degrees of freedom
   - Your DOF for error correction: {len(data) - trace_rank * len(SELECTED_VARS)} obs - {trace_rank * len(SELECTED_VARS)} params

3. Interpretability:
   - Lower rank -> Simpler long-run structure
   - Higher rank -> More complex equilibrium relationships
   - Rank = {trace_rank}: {'Simple (1-2 relationships)' if trace_rank <= 2 else 'Moderate complexity' if trace_rank <= 4 else 'High complexity (many relationships)'}

4. Theoretical Justification:
   - Can you explain {trace_rank} separate long-run equilibria?
   - DoD bureaucracy context: What are these equilibrium relationships?

RECOMMENDATIONS:

Option A (Statistical): Use rank = {trace_rank} (trace test result)
  -> Maximizes statistical fit
  -> May be harder to interpret

Option B (Conservative): Use rank = {max(1, trace_rank - 1)}
  -> More stable with small sample
  -> Easier to interpret
  -> Still captures main long-run dynamics

Option C (Very Conservative): Use rank = {max(1, trace_rank - 2)}
  -> Most stable
  -> Simplest interpretation
  -> May miss some long-run relationships
""")

print("\n" + "=" * 100)
print("NEXT STEPS")
print("=" * 100)
print("""
1. Review the trace statistic table above
2. Understand which rows show Trace Stat > 95% Crit (these are significant)
3. Choose your cointegration rank based on:
   - Statistical tests (trace/max eigenvalue)
   - Sample size considerations
   - Interpretability of results
4. Run VECM with your chosen rank
5. Examine beta (cointegration) vectors to understand long-run equilibria
6. Examine alpha (error correction) speeds to see adjustment dynamics
""")

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)
