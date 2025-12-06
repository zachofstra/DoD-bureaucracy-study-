"""
Update complete_normalized_dataset_v10.6_FULL.xlsx
Replace all rank percentages with their z-scored versions
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("UPDATING DATASET WITH Z-SCORED RANK PERCENTAGES")
print("=" * 100)

# Load the current dataset
print("\n[1/3] Loading dataset...")
df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')
print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

# Rank percentages to z-score normalize
rank_pcts = [
    'E1_Pct', 'E2_Pct', 'E3_Pct', 'E4_Pct', 'E5_Pct',
    'E6_Pct', 'E7_Pct', 'E8_Pct', 'E9_Pct',
    'O1_Pct', 'O2_Pct', 'O3_Pct', 'O4_MajorLTCDR_Pct',
    'O5_LtColCDR_Pct', 'O6_ColCAPT_Pct', 'O7_Pct', 'O8_Pct',
    'O9_Pct', 'O10_Pct'
]

print("\n[2/3] Z-score normalizing rank percentages...")
print("  " + "-" * 96)

for rank in rank_pcts:
    if rank in df.columns:
        mean_val = df[rank].mean()
        std_val = df[rank].std()

        # Replace the percentage column with z-scored version
        df[rank] = (df[rank] - mean_val) / std_val

        print(f"    {rank:25s} -> z-scored (mean={mean_val:6.2f}%, std={std_val:5.2f}%)")

# Save the updated dataset
print("\n[3/3] Saving updated dataset...")
output_path = 'data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx'
df.to_excel(output_path, index=False)

print(f"\n[OK] Saved to: {output_path}")
print(f"  {len(df)} rows, {len(df.columns)} columns")

print("\n" + "=" * 100)
print("COMPLETE - All rank percentages replaced with z-scores")
print("=" * 100)
