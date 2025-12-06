"""
Create Network Adjacency Matrices from VAR(2) Coefficients
Generates network representations showing variable relationships
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("CREATING NETWORK ADJACENCY MATRICES FROM VAR(2) COEFFICIENTS")
print("=" * 80)

# Variables in order
VARS = [
    'Warrant_Officers_Z',
    'Policy_Count_Log',
    'Company_Grade_Officers_Z',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z',
    'Junior_Enlisted_Z',
    'Field_Grade_Officers_Z',
    'Total_Civilians_Z'
]

print("\n[1/3] Loading coefficient matrices...")

# Load lag 1 and lag 2 coefficients
coef_lag1 = pd.read_excel('coefficients_lag1.xlsx', index_col=0)
coef_lag2 = pd.read_excel('coefficients_lag2.xlsx', index_col=0)

print(f"  Lag 1 shape: {coef_lag1.shape}")
print(f"  Lag 2 shape: {coef_lag2.shape}")

# =============================================================================
# CREATE COMBINED NETWORK ADJACENCY MATRIX (sum of absolute values)
# =============================================================================
print("\n[2/3] Creating combined network adjacency matrix...")

# Sum absolute values across both lags
adjacency_abs = np.abs(coef_lag1.values) + np.abs(coef_lag2.values)
adjacency_abs_df = pd.DataFrame(adjacency_abs, index=VARS, columns=VARS)

# Save
adjacency_abs_df.to_excel('network_adjacency_matrix_combined.xlsx')
print("  Saved: network_adjacency_matrix_combined.xlsx")

# =============================================================================
# CREATE SIGNED NETWORK MATRIX (preserving direction)
# =============================================================================
print("\n[3/3] Creating signed network adjacency matrix...")

# Sum with signs preserved
adjacency_signed = coef_lag1.values + coef_lag2.values
adjacency_signed_df = pd.DataFrame(adjacency_signed, index=VARS, columns=VARS)

# Save
adjacency_signed_df.to_excel('network_adjacency_matrix_signed.xlsx')
print("  Saved: network_adjacency_matrix_signed.xlsx")

# =============================================================================
# CREATE NETWORK EDGE LIST
# =============================================================================
print("\n[4/4] Creating network edge list...")

edges = []
threshold = 0.05  # Minimum coefficient magnitude to include

for i, from_var in enumerate(VARS):
    for j, to_var in enumerate(VARS):
        if i == j:
            continue

        # Get coefficients from both lags
        coef_t1 = coef_lag1.iloc[i, j]
        coef_t2 = coef_lag2.iloc[i, j]

        # Combined magnitude
        magnitude = abs(coef_t1) + abs(coef_t2)

        if magnitude > threshold:
            # Determine overall direction
            signed_sum = coef_t1 + coef_t2
            direction = 'Amplifying' if signed_sum > 0 else 'Dampening'

            edges.append({
                'From': from_var,
                'To': to_var,
                'Lag1_Coef': coef_t1,
                'Lag2_Coef': coef_t2,
                'Combined_Magnitude': magnitude,
                'Signed_Sum': signed_sum,
                'Direction': direction
            })

edges_df = pd.DataFrame(edges)
edges_df = edges_df.sort_values('Combined_Magnitude', ascending=False)
edges_df.to_excel('network_edge_list.xlsx', index=False)

print(f"  Saved: network_edge_list.xlsx ({len(edges_df)} edges)")
print(f"  Threshold: |coef| > {threshold}")

# =============================================================================
# NETWORK STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("NETWORK STATISTICS")
print("=" * 80)

total_possible = len(VARS) * (len(VARS) - 1)
density = len(edges_df) / total_possible

print(f"\nNodes (variables): {len(VARS)}")
print(f"Edges (relationships > {threshold}): {len(edges_df)}")
print(f"Total possible directed edges: {total_possible}")
print(f"Network density: {density:.3f}")

# Top 5 strongest relationships
print(f"\nTOP 5 STRONGEST RELATIONSHIPS:")
print("-" * 80)
for idx, row in edges_df.head(5).iterrows():
    print(f"  {row['From']:30s} → {row['To']:30s}: {row['Combined_Magnitude']:6.3f} ({row['Direction']})")

# Degree centrality (in-degree and out-degree)
print(f"\nDEGREE CENTRALITY:")
print("-" * 80)

in_degree = edges_df['To'].value_counts()
out_degree = edges_df['From'].value_counts()

print("\nMost Influenced (In-Degree):")
for var, count in in_degree.head(5).items():
    print(f"  {var:40s}: {count} incoming edges")

print("\nMost Influential (Out-Degree):")
for var, count in out_degree.head(5).items():
    print(f"  {var:40s}: {count} outgoing edges")

print("\n" + "=" * 80)
print("FILES GENERATED:")
print("=" * 80)
print("  1. network_adjacency_matrix_combined.xlsx - Sum of |coefficients| from both lags")
print("  2. network_adjacency_matrix_signed.xlsx - Signed sum preserving direction")
print("  3. network_edge_list.xlsx - Ranked list of variable relationships")
print("=" * 80)
print("\nInterpretation:")
print("  - Adjacency matrices show strength of relationships between variables")
print("  - Edge list identifies specific causal pathways in the VAR(2) model")
print("  - Direction indicates whether relationship is amplifying (+) or dampening (-)")
print("  - Magnitude shows combined strength across both lags")
print("=" * 80)
