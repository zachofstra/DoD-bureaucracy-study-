"""
Johansen Cointegration Analysis - DoD Bureaucratic Growth
Testing for long-run equilibrium relationships between trending variables

This complements the VAR(2) short-run dynamics by examining which variables
are tied together over the full 37-year period (1987-2024).

KEY INSIGHT:
VAR with differencing captures year-to-year dynamics (short run)
Cointegration captures multi-decade trends (long run)

Your thesis story is about LONG-RUN trends:
- O4-O6 field grade officers growing over 37 years
- E1-E5 junior enlisted declining over 37 years
- Policy burden increasing over 37 years
- These trends moving together (or inversely) over time

Variables tested:
1. Junior_Enlisted_Z (trending DOWN)
2. FOIA_Simple_Days_Z (trending UP - bureaucratic delay)
3. Total_PAS_Z (trending UP - political appointees)
4. Total_Civilians_Z (trending UP - civilian bureaucracy)
5. Policy_Count_Log (trending UP - regulatory burden)
6. Field_Grade_Officers_Z (trending UP - O4-O6 bureaucratic layer)
7. GOFOs_Z (trending UP - flag officers)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("JOHANSEN COINTEGRATION ANALYSIS - LONG-RUN EQUILIBRIUM RELATIONSHIPS")
print("=" * 100)

# =============================================================================
# LOAD DATA - USE LEVELS (NOT DIFFERENCED)
# =============================================================================
print("\n[1/5] Loading data (LEVELS, not differenced)...")

df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

# Variables to test for cointegration (all trending)
coint_vars = [
    'Junior_Enlisted_Z',
    'FOIA_Simple_Days_Z',
    'Total_PAS_Z',
    'Total_Civilians_Z',
    'Policy_Count_Log',
    'Field_Grade_Officers_Z',
    'GOFOs_Z'
]

data = df[coint_vars].copy().dropna()

print(f"  Variables: {len(coint_vars)}")
print(f"  Observations: {len(data)}")
print(f"  Time period: {df['FY'].min():.0f}-{df['FY'].max():.0f}")

# =============================================================================
# VERIFY VARIABLES ARE I(1) - INTEGRATED OF ORDER 1
# =============================================================================
print("\n[2/5] Verifying variables are I(1) (non-stationary in levels, stationary in differences)...")
print("  " + "-" * 96)

integration_results = []

for var in coint_vars:
    # Test levels
    adf_levels = adfuller(data[var], maxlag=4, regression='ct')
    pval_levels = adf_levels[1]

    # Test first differences
    adf_diff = adfuller(data[var].diff().dropna(), maxlag=4, regression='ct')
    pval_diff = adf_diff[1]

    # Determine integration order
    if pval_levels < 0.05:
        order = "I(0) - STATIONARY IN LEVELS"
        warning = "** WARNING **"
    elif pval_diff < 0.05:
        order = "I(1) - OK FOR COINTEGRATION"
        warning = ""
    else:
        order = "I(2) or higher - PROBLEM"
        warning = "** WARNING **"

    integration_results.append({
        'Variable': var,
        'ADF_Levels_p': pval_levels,
        'ADF_Diff_p': pval_diff,
        'Integration_Order': order,
        'Warning': warning
    })

    print(f"  {warning:15s} {var:30s} Levels p={pval_levels:.4f}, Diff p={pval_diff:.4f} -> {order}")

integration_df = pd.DataFrame(integration_results)

# Count how many are I(1)
n_i1 = sum(1 for r in integration_results if 'I(1)' in r['Integration_Order'])
print(f"\n  Result: {n_i1}/{len(coint_vars)} variables are I(1) - suitable for cointegration")

if n_i1 < len(coint_vars):
    print("\n  NOTE: Some variables are I(0). Cointegration tests assume all I(1).")
    print("        Results should be interpreted with caution for mixed orders.")

# =============================================================================
# JOHANSEN COINTEGRATION TEST
# =============================================================================
print("\n[3/5] Running Johansen cointegration test...")
print("  " + "-" * 96)

# Johansen test with deterministic trend in data, constant in cointegrating equation
# det_order options:
# -1 = no deterministic terms
#  0 = constant in cointegrating equation only
#  1 = constant in both cointegrating and VAR equations
joh_result = coint_johansen(data, det_order=0, k_ar_diff=2)

print("\n  TRACE STATISTIC TEST:")
print("  " + "-" * 96)
print(f"  {'Rank':>6s} {'Trace Stat':>12s} {'10% CV':>10s} {'5% CV':>10s} {'1% CV':>10s} {'Reject?':>10s}")
print("  " + "-" * 96)

cointegration_ranks = []
for i in range(len(coint_vars)):
    trace_stat = joh_result.lr1[i]
    cv_10 = joh_result.cvt[i, 0]
    cv_5 = joh_result.cvt[i, 1]
    cv_1 = joh_result.cvt[i, 2]

    # Reject null if trace stat > critical value at 5%
    reject_5pct = "YES **" if trace_stat > cv_5 else "NO"

    cointegration_ranks.append({
        'Rank': i,
        'Trace_Stat': trace_stat,
        'CV_10pct': cv_10,
        'CV_5pct': cv_5,
        'CV_1pct': cv_1,
        'Reject_5pct': reject_5pct
    })

    print(f"  r<={i:2d}  {trace_stat:12.2f} {cv_10:10.2f} {cv_5:10.2f} {cv_1:10.2f} {reject_5pct:>10s}")

print("\n  INTERPRETATION:")
print("  The trace statistic tests H0: 'at most r cointegrating relationships'")
print("  We reject H0 if Trace Stat > Critical Value (5%)")

# Find number of cointegrating relationships
n_coint = 0
for result in cointegration_ranks:
    if result['Reject_5pct'] == "YES **":
        n_coint = result['Rank'] + 1

print(f"\n  RESULT: {n_coint} cointegrating relationship(s) detected at 5% level")

if n_coint == 0:
    print("\n  ** NO COINTEGRATION FOUND **")
    print("  Variables do NOT share long-run equilibrium relationships.")
    print("  They drift independently over the 37-year period.")
elif n_coint == 1:
    print("\n  ** ONE COINTEGRATING VECTOR **")
    print("  One linear combination of these variables is stationary (tied together long-run).")
elif n_coint > 1:
    print(f"\n  ** {n_coint} COINTEGRATING VECTORS **")
    print(f"  {n_coint} independent long-run equilibrium relationships exist.")

# =============================================================================
# EIGENVECTORS (COINTEGRATING VECTORS)
# =============================================================================
print("\n[4/5] Examining cointegrating vectors (if any)...")
print("  " + "-" * 96)

if n_coint > 0:
    print(f"\n  Displaying {min(n_coint, 3)} cointegrating vector(s):\n")

    eigenvectors = joh_result.evec

    coint_vectors = []
    for i in range(min(n_coint, 3)):
        print(f"  COINTEGRATING VECTOR #{i+1}:")
        print(f"  {'Variable':30s} {'Coefficient':>12s} {'Interpretation':>40s}")
        print("  " + "-" * 96)

        for j, var in enumerate(coint_vars):
            coef = eigenvectors[j, i]

            # Interpret coefficient
            if abs(coef) < 0.1:
                interp = "weak/negligible"
            elif coef > 0:
                interp = "moves together (positive)"
            else:
                interp = "moves inversely (negative)"

            coint_vectors.append({
                'Vector': i+1,
                'Variable': var,
                'Coefficient': coef,
                'Interpretation': interp
            })

            print(f"  {var:30s} {coef:12.4f} {interp:>40s}")

        print("")

    # Save eigenvectors
    coint_vectors_df = pd.DataFrame(coint_vectors)
    coint_vectors_df.to_excel('data/analysis/cointegration_vectors.xlsx', index=False)
    print("  [OK] Cointegrating vectors saved to cointegration_vectors.xlsx")

else:
    print("\n  No cointegrating vectors to display (n_coint = 0)")

# =============================================================================
# PAIRWISE COINTEGRATION TESTS
# =============================================================================
print("\n[5/5] Testing pairwise cointegration (all variable pairs)...")
print("  " + "-" * 96)

pairwise_results = []

for i in range(len(coint_vars)):
    for j in range(i+1, len(coint_vars)):
        var1 = coint_vars[i]
        var2 = coint_vars[j]

        pair_data = data[[var1, var2]].dropna()

        try:
            joh_pair = coint_johansen(pair_data, det_order=0, k_ar_diff=2)

            # Check if trace stat > 5% critical value for r=0
            trace_stat = joh_pair.lr1[0]
            cv_5 = joh_pair.cvt[0, 1]

            cointegrated = "YES **" if trace_stat > cv_5 else "NO"

            pairwise_results.append({
                'Variable_1': var1,
                'Variable_2': var2,
                'Trace_Stat': trace_stat,
                'CV_5pct': cv_5,
                'Cointegrated_5pct': cointegrated,
                'Eigenvector_1': joh_pair.evec[0, 0],
                'Eigenvector_2': joh_pair.evec[1, 0]
            })
        except:
            pairwise_results.append({
                'Variable_1': var1,
                'Variable_2': var2,
                'Trace_Stat': np.nan,
                'CV_5pct': np.nan,
                'Cointegrated_5pct': 'ERROR',
                'Eigenvector_1': np.nan,
                'Eigenvector_2': np.nan
            })

pairwise_df = pd.DataFrame(pairwise_results)
pairwise_df.to_excel('data/analysis/pairwise_cointegration.xlsx', index=False)

# Show significant pairwise relationships
sig_pairs = pairwise_df[pairwise_df['Cointegrated_5pct'] == 'YES **']
n_sig_pairs = len(sig_pairs)

print(f"\n  Total pairs tested: {len(pairwise_df)}")
print(f"  Cointegrated pairs (5% level): {n_sig_pairs}")

if n_sig_pairs > 0:
    print("\n  SIGNIFICANT PAIRWISE COINTEGRATION:")
    print("  " + "-" * 96)
    print(f"  {'Variable 1':30s} {'Variable 2':30s} {'Trace':>10s} {'CV 5%':>10s}")
    print("  " + "-" * 96)

    for idx, row in sig_pairs.iterrows():
        print(f"  {row['Variable_1']:30s} {row['Variable_2']:30s} {row['Trace_Stat']:10.2f} {row['CV_5pct']:10.2f}")
else:
    print("\n  ** NO SIGNIFICANT PAIRWISE COINTEGRATION FOUND **")

# =============================================================================
# VISUALIZE COINTEGRATION NETWORK
# =============================================================================
if n_sig_pairs > 0:
    print("\n  Creating cointegration network diagram...")

    import networkx as nx

    G = nx.Graph()

    for idx, row in sig_pairs.iterrows():
        G.add_edge(row['Variable_1'], row['Variable_2'],
                   weight=row['Trace_Stat'],
                   cv=row['CV_5pct'])

    # Add isolated nodes
    for var in coint_vars:
        if var not in G:
            G.add_node(var)

    # Calculate centrality
    degree_cent = dict(G.degree(weight='weight'))

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='white')

    pos = nx.spring_layout(G, k=3, iterations=50, seed=42, weight='weight')

    # Node colors by category
    node_colors = []
    for node in G.nodes():
        if 'Enlisted' in node:
            node_colors.append('#3498db')  # Blue
        elif 'Officers' in node or 'GOFO' in node:
            node_colors.append('#e74c3c')  # Red
        elif 'Civilian' in node or 'PAS' in node or 'FOIA' in node or 'Policy' in node:
            node_colors.append('#2ecc71')  # Green
        else:
            node_colors.append('#95a5a6')  # Gray

    # Node sizes by degree
    node_sizes = [800 + degree_cent.get(node, 0) * 100 for node in G.nodes()]

    # Edge widths by trace statistic
    edge_widths = [1 + (G[u][v]['weight'] / 20) * 4 for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, edge_color='#34495e', alpha=0.6,
                           width=edge_widths, ax=ax)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9,
                           edgecolors='black', linewidths=2.5, ax=ax)

    labels = {node: node.replace('_Z', '').replace('_', '\n') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=11, font_weight='bold', ax=ax)

    # Edge labels (trace statistics)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9,
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                                 ax=ax)

    ax.set_title('Long-Run Equilibrium Relationships (Cointegration)\n' +
                'DoD Bureaucratic Growth (1987-2024)\n' +
                'Edge = Significant cointegration (5% level), Width = Trace statistic strength',
                fontsize=18, fontweight='bold', pad=30)
    ax.axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Enlisted Personnel', alpha=0.9),
        Patch(facecolor='#e74c3c', label='Officer Personnel', alpha=0.9),
        Patch(facecolor='#2ecc71', label='Bureaucratic Measures', alpha=0.9)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
             framealpha=0.95, title='Variable Categories', title_fontsize=13)

    plt.tight_layout()
    plt.savefig('data/analysis/cointegration_network.png', dpi=300, bbox_inches='tight')
    print("  [OK] Cointegration network saved")

# =============================================================================
# SUMMARY OUTPUT
# =============================================================================
print("\n" + "=" * 100)
print("COINTEGRATION ANALYSIS COMPLETE")
print("=" * 100)

summary_text = f"""
================================================================================
JOHANSEN COINTEGRATION ANALYSIS - EXECUTIVE SUMMARY
DoD Bureaucratic Growth: Long-Run Equilibrium Relationships (1987-2024)
================================================================================

WHAT IS COINTEGRATION?
----------------------
Cointegration tests whether non-stationary variables (trending over time) are
tied together in the LONG RUN. Even though individual variables may drift up
or down over decades, cointegrated variables maintain a stable relationship.

WHY THIS COMPLEMENTS VAR(2)?
-----------------------------
VAR(2) with differencing captures SHORT-TERM dynamics (year-to-year changes).
Cointegration captures LONG-TERM trends (multi-decade equilibrium).

Your thesis story is about 37-year trends:
  - O4-O6 field grade officers GROWING
  - E1-E5 junior enlisted DECLINING
  - Policy burden INCREASING
  - Civilian workforce EXPANDING

The VAR removed these trends by differencing. Cointegration puts them back in.

================================================================================
RESULTS
================================================================================

Full System Test (All 7 Variables):
-----------------------------------
Number of cointegrating relationships: {n_coint}

Interpretation:
"""

if n_coint == 0:
    summary_text += """
** NO COINTEGRATION IN FULL SYSTEM **

The 7 variables do NOT share long-run equilibrium relationships when analyzed
together. They drift independently over the 37-year period.

This does NOT mean there are no long-run relationships - it means the full
system is too complex. Check pairwise results below for simpler relationships.
"""
elif n_coint == 1:
    summary_text += """
** ONE COINTEGRATING VECTOR FOUND **

One linear combination of these 7 variables is stationary (tied together in
the long run). See cointegration_vectors.xlsx for the specific relationship.

This suggests the bureaucratic system has ONE fundamental long-run equilibrium
that constrains how variables move together over decades.
"""
else:
    summary_text += f"""
** {n_coint} COINTEGRATING VECTORS FOUND **

{n_coint} independent long-run equilibrium relationships exist. See
cointegration_vectors.xlsx for details.

This suggests the bureaucratic system has MULTIPLE long-run equilibria
governing different aspects of growth.
"""

summary_text += f"""

Pairwise Cointegration Tests:
------------------------------
Total pairs tested: {len(pairwise_df)}
Cointegrated pairs (5% level): {n_sig_pairs}

"""

if n_sig_pairs > 0:
    summary_text += "SIGNIFICANT PAIRWISE RELATIONSHIPS:\n\n"
    for idx, row in sig_pairs.iterrows():
        v1 = row['Variable_1'].replace('_Z', '').replace('_', ' ')
        v2 = row['Variable_2'].replace('_Z', '').replace('_', ' ')
        summary_text += f"  {v1} <--> {v2}\n"
        summary_text += f"    Trace Statistic: {row['Trace_Stat']:.2f} (Critical Value: {row['CV_5pct']:.2f})\n"
        summary_text += "    These variables are tied together over the 37-year period.\n\n"
else:
    summary_text += """** NO SIGNIFICANT PAIRWISE COINTEGRATION **

None of the variable pairs show long-run equilibrium relationships. This is
SURPRISING given your descriptive findings (O4s growing, E1-E5 declining).

Possible explanations:
1. Structural breaks (e.g., 9/11, sequestration, COVID) disrupted long-run ties
2. Relationships are non-linear (cointegration assumes linear equilibrium)
3. Sample size (38 observations) may be insufficient for cointegration tests
4. Variables truly drift independently despite apparent trends
"""

summary_text += """

================================================================================
INTEGRATION WITH YOUR THESIS
================================================================================

"""

if n_sig_pairs > 0:
    summary_text += """
Your thesis can now make TWO types of claims:

1. SHORT-RUN DYNAMICS (from VAR):
   - How variables respond to each other year-to-year
   - Feedback loops and causal pathways
   - Example: "A 1-unit shock to civilians increases PAS by 0.36 the next year"

2. LONG-RUN EQUILIBRIUM (from Cointegration):
   - Which variables are fundamentally tied together over decades
   - Persistent relationships that constrain the system
   - Example: "Field grade officers and policy count maintain equilibrium"

NARRATIVE:
"DoD bureaucracy exhibits dual dynamics: short-term adaptations (VAR) within
long-term structural constraints (cointegration). Even as individual variables
fluctuate year-to-year, the cointegrating relationships reveal fundamental
equilibria that have persisted across 37 years."
"""
else:
    summary_text += """
** ABSENCE OF COINTEGRATION IS ALSO A FINDING **

The lack of long-run equilibrium relationships suggests:

1. STRUCTURAL INSTABILITY: Major shocks (9/11, sequestration, COVID) may have
   fundamentally altered bureaucratic relationships over time, preventing
   stable long-run ties.

2. POLICY REGIME CHANGES: Different administrations and strategic priorities
   may have shifted the "rules of the game" for bureaucratic growth.

3. FUNCTIONAL SUBSTITUTION: Different bureaucratic dimensions (military vs
   civilian, O4-O6 vs policy count) may serve as substitutes rather than
   complements, allowing independent drift.

THESIS IMPLICATION:
"Unlike traditional bureaucracies that Weber described as iron cages with
fixed equilibria, DoD bureaucracy shows ADAPTIVE DRIFT. Variables grow or
decline independently over decades, suggesting the system continuously
rebalances rather than settling into permanent structures. This supports
the 'demigarch' concept - leaders who adapt bureaucracy to mission needs
rather than preserving fixed hierarchies."
"""

summary_text += """

================================================================================
NEXT STEPS FOR THESIS
================================================================================

1. DESCRIBE both VAR (short-run) and cointegration (long-run) findings

2. If cointegration found:
   - Emphasize STRUCTURAL PERSISTENCE despite year-to-year fluctuations
   - Long-run relationships show which bureaucratic dimensions are locked together

3. If no cointegration:
   - Emphasize ADAPTIVE FLEXIBILITY and structural breaks
   - Absence of fixed equilibria supports your "demigarch" innovation concept

4. RECONCILE with descriptive trends:
   - Cointegration tests LINEAR relationships
   - Your O4 growth vs E1-E5 decline may be NON-LINEAR or threshold-based
   - Consider vector error correction model (VECM) if cointegration found

================================================================================
FILES GENERATED
================================================================================

1. cointegration_vectors.xlsx
   - Full eigenvectors for cointegrating relationships (if any)

2. pairwise_cointegration.xlsx
   - All pairwise tests with trace statistics and critical values

3. cointegration_network.png
   - Visual network of long-run equilibrium relationships (if any found)

================================================================================
"""

with open('data/analysis/COINTEGRATION_SUMMARY.txt', 'w') as f:
    f.write(summary_text)

print("\n[OK] Executive summary saved to COINTEGRATION_SUMMARY.txt")

print("\n" + "=" * 100)
print("KEY FINDING:")
print("=" * 100)
if n_coint > 0 or n_sig_pairs > 0:
    print(f"  {n_sig_pairs} pairwise long-run relationships found")
    print("  These variables are tied together over the 37-year period")
    print("  See COINTEGRATION_SUMMARY.txt for interpretation")
else:
    print("  NO long-run equilibrium relationships detected")
    print("  Variables drift independently over 37 years")
    print("  This is ALSO a meaningful finding - see summary for implications")
print("=" * 100)

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print("  1. cointegration_vectors.xlsx - Eigenvectors (if cointegration found)")
print("  2. pairwise_cointegration.xlsx - All pairwise tests")
print("  3. cointegration_network.png - Visual network (if relationships found)")
print("  4. COINTEGRATION_SUMMARY.txt - Full interpretation and thesis guidance")
print("=" * 100)
