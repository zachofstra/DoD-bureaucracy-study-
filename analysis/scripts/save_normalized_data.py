"""
Save all variables in normalized/transformed forms to Excel
"""

import pandas as pd
import numpy as np

print("Creating normalized dataset...")

# Load data
df = pd.read_excel('data/analysis/complete_relative_dataset.xlsx')

# Calculate Total_Civilians
df['Total_Civilians'] = df['Civ_Army'] + df['Civ_Navy'] + df['Civ_AirForce']

# Create log-transformed Policy_Count
df['Policy_Count_Log'] = np.log(df['Policy_Count'] + 1)

# Z-score normalize Total_Civilians
civ_mean = df['Total_Civilians'].mean()
civ_std = df['Total_Civilians'].std()
df['Total_Civilians_Z'] = (df['Total_Civilians'] - civ_mean) / civ_std

# Z-score normalize Total_PAS
pas_mean = df['Total_PAS'].mean()
pas_std = df['Total_PAS'].std()
df['Total_PAS_Z'] = (df['Total_PAS'] - pas_mean) / pas_std

# Select all relevant columns (original + normalized)
output_df = df[[
    'FY',
    # Original variables
    'Policy_Count',
    'Total_Civilians',
    'Total_PAS',
    'O5_LtColCDR_Pct',
    'O4_MajorLTCDR_Pct',
    'E5_Pct',
    'O6_ColCAPT_Pct',
    'GDP_Growth',
    'Major_Conflict',
    # Normalized/transformed variables
    'Policy_Count_Log',
    'Total_Civilians_Z',
    'Total_PAS_Z'
]].copy()

# Save to Excel
output_path = 'data/analysis/normalized_dataset_v10.6.xlsx'
output_df.to_excel(output_path, index=False)

print(f"\n[OK] Saved to: {output_path}")
print(f"\nColumns included:")
print("  Original: Policy_Count, Total_Civilians, Total_PAS, O5-O6-O4-E5 Pcts, GDP_Growth, Major_Conflict")
print("  Normalized: Policy_Count_Log, Total_Civilians_Z, Total_PAS_Z")
print(f"\nRows: {len(output_df)}")

# Print summary
print("\n" + "=" * 80)
print("NORMALIZATION SUMMARY:")
print("=" * 80)
print(f"\nPolicy_Count_Log = log(Policy_Count + 1)")
print(f"  Range: {output_df['Policy_Count_Log'].min():.3f} to {output_df['Policy_Count_Log'].max():.3f}")

print(f"\nTotal_Civilians_Z = (Total_Civilians - {civ_mean:.0f}) / {civ_std:.0f}")
print(f"  Range: {output_df['Total_Civilians_Z'].min():.3f} to {output_df['Total_Civilians_Z'].max():.3f}")

print(f"\nTotal_PAS_Z = (Total_PAS - {pas_mean:.0f}) / {pas_std:.0f}")
print(f"  Range: {output_df['Total_PAS_Z'].min():.3f} to {output_df['Total_PAS_Z'].max():.3f}")

print("\n" + "=" * 80)
