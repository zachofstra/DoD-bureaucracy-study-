"""
Save ALL variables in normalized/transformed forms to Excel
Includes all ranks, totals, policy data, FOIA, etc.
Excludes service-specific breakdowns
"""

import pandas as pd
import numpy as np

print("Creating comprehensive normalized dataset...")

# Load data
df = pd.read_excel('data/analysis/complete_relative_dataset.xlsx')

print(f"\nOriginal dataset columns: {len(df.columns)}")
print("\nAll columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# Calculate Total_Civilians
df['Total_Civilians'] = df['Civ_Army'] + df['Civ_Navy'] + df['Civ_AirForce']

# Create normalized versions
print("\n" + "=" * 100)
print("CREATING NORMALIZED VARIABLES")
print("=" * 100)

# Log-transform Policy_Count
df['Policy_Count_Log'] = np.log(df['Policy_Count'] + 1)
print(f"\nPolicy_Count_Log = log(Policy_Count + 1)")
print(f"  Range: {df['Policy_Count_Log'].min():.3f} to {df['Policy_Count_Log'].max():.3f}")

# Z-score normalize Total_Civilians
civ_mean = df['Total_Civilians'].mean()
civ_std = df['Total_Civilians'].std()
df['Total_Civilians_Z'] = (df['Total_Civilians'] - civ_mean) / civ_std
print(f"\nTotal_Civilians_Z = (Total_Civilians - {civ_mean:.0f}) / {civ_std:.0f}")
print(f"  Range: {df['Total_Civilians_Z'].min():.3f} to {df['Total_Civilians_Z'].max():.3f}")

# Z-score normalize Total_PAS
pas_mean = df['Total_PAS'].mean()
pas_std = df['Total_PAS'].std()
df['Total_PAS_Z'] = (df['Total_PAS'] - pas_mean) / pas_std
print(f"\nTotal_PAS_Z = (Total_PAS - {pas_mean:.0f}) / {pas_std:.0f}")
print(f"  Range: {df['Total_PAS_Z'].min():.3f} to {df['Total_PAS_Z'].max():.3f}")

# Get all columns EXCEPT service-specific breakdowns
exclude_patterns = ['Civ_Army', 'Civ_Navy', 'Civ_AirForce']

# Get all columns
all_cols = list(df.columns)

# Filter out service-specific columns
selected_cols = [col for col in all_cols if not any(pattern in col for pattern in exclude_patterns)]

# Make sure we have the normalized ones
if 'Policy_Count_Log' not in selected_cols:
    selected_cols.append('Policy_Count_Log')
if 'Total_Civilians_Z' not in selected_cols:
    selected_cols.append('Total_Civilians_Z')
if 'Total_PAS_Z' not in selected_cols:
    selected_cols.append('Total_PAS_Z')

# Create output dataframe
output_df = df[selected_cols].copy()

# Save to Excel
output_path = 'data/analysis/complete_normalized_dataset_v10.6.xlsx'
output_df.to_excel(output_path, index=False)

print("\n" + "=" * 100)
print(f"[OK] Saved to: {output_path}")
print("=" * 100)

print(f"\nTotal columns: {len(output_df.columns)}")
print(f"Total rows: {len(output_df)}")

print("\n" + "=" * 100)
print("COLUMNS INCLUDED:")
print("=" * 100)
for i, col in enumerate(output_df.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "=" * 100)
print("COLUMNS EXCLUDED:")
print("=" * 100)
excluded = [col for col in all_cols if col not in selected_cols]
for i, col in enumerate(excluded, 1):
    print(f"  {i}. {col}")
