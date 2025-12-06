"""
Create Comprehensive Normalized Dataset v10.6
Includes:
- All rank percentages (E1-E7, O1-O10)
- Total variables
- FOIA data (z-scored)
- Political/exogenous variables
- Normalized versions (z-scores and log transforms)
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("CREATING COMPREHENSIVE NORMALIZED DATASET v10.6")
print("=" * 100)

# =============================================================================
# 1. LOAD BASE DATASET
# =============================================================================
print("\n[1/5] Loading base dataset...")
df_base = pd.read_excel('data/analysis/complete_relative_dataset.xlsx')

# Calculate Total_Civilians
df_base['Total_Civilians'] = df_base['Civ_Army'] + df_base['Civ_Navy'] + df_base['Civ_AirForce']

print(f"  Base dataset: {len(df_base)} rows, {len(df_base.columns)} columns")

# =============================================================================
# 2. EXTRACT O1-O3 AND O7-O10 FROM AD_STRENGTHS
# =============================================================================
print("\n[2/5] Extracting O1-O3 and O7-O10 from AD Strengths data...")

# Load all years and extract officer ranks
officer_ranks_data = []

xls = pd.ExcelFile('data/AD_strengths/AD_Strengths_FY1987-2024_normalized.xlsx')

for year in range(1987, 2025):
    sheet_name = str(year)
    if sheet_name in xls.sheet_names:
        df_year = pd.read_excel(xls, sheet_name=sheet_name)

        # Get Total DoD column
        total_col = 'Total DoD'
        if total_col not in df_year.columns:
            continue

        # Extract officer counts by rank
        officer_data = {'FY': year}

        # Map rank names to counts (using exact string matching)
        rank_map = {
            '2ND LIEUTENANT - ENSIGN': 'O1',
            '1ST LIEUTENANT - LIEUTENANT (JG)': 'O2',
            'CAPTAIN - LIEUTENANT': 'O3',
            'BRIG GENERAL - REAR ADMIRAL (L)': 'O7',
            'MAJ GENERAL - REAR ADMIRAL (U)': 'O8',
            'LT GENERAL - VICE ADMIRAL': 'O9',
            'GENERAL - ADMIRAL': 'O10',
            'E-8': 'E8',
            'E-9': 'E9'
        }

        for rank_label, rank_code in rank_map.items():
            # Use exact string match instead of contains to avoid regex issues
            mask = df_year['Rank/Grade'].astype(str).str.upper().str.strip() == rank_label.upper()
            if mask.any():
                value = df_year.loc[mask, total_col].values[0]
                try:
                    officer_data[rank_code] = float(value)
                except:
                    officer_data[rank_code] = np.nan

        # Get total military for percentage calc
        total_mask = df_year['Rank/Grade'].astype(str).str.contains('Total DoD', na=False)
        if total_mask.any():
            total_mil = df_year.loc[total_mask, total_col].values[0]
            try:
                officer_data['Total_Military_Check'] = float(total_mil)
            except:
                pass

        officer_ranks_data.append(officer_data)

df_officers = pd.DataFrame(officer_ranks_data)

# Calculate percentages
for rank in ['O1', 'O2', 'O3', 'O7', 'O8', 'O9', 'O10', 'E8', 'E9']:
    if rank in df_officers.columns:
        # Use Total_Military from base dataset for consistency
        df_officers = df_officers.merge(df_base[['FY', 'Total_Military']], on='FY', how='left')
        df_officers[f'{rank}_Pct'] = (df_officers[rank] / df_officers['Total_Military']) * 100
        df_officers.drop(columns=[rank, 'Total_Military'], inplace=True, errors='ignore')

# Drop the check column
df_officers.drop(columns=['Total_Military_Check'], inplace=True, errors='ignore')

print(f"  Officer ranks extracted: {len(df_officers)} years")
print(f"  Columns: {list(df_officers.columns)}")

# =============================================================================
# 3. LOAD FOIA DATA
# =============================================================================
print("\n[3/5] Loading FOIA data...")

df_foia = pd.read_csv('data/analysis/foia-processed-requests-response-time-for-all-processed-perfected-requests.csv',
                      encoding='latin1')

# Clean column names
df_foia.columns = ['FY', 'FOIA_Simple_Days', 'FOIA_Complex_Days', 'Unnamed_3', 'Unnamed_4']
df_foia = df_foia[['FY', 'FOIA_Simple_Days', 'FOIA_Complex_Days']]

# Convert to numeric
df_foia['FOIA_Simple_Days'] = pd.to_numeric(df_foia['FOIA_Simple_Days'], errors='coerce')
df_foia['FOIA_Complex_Days'] = pd.to_numeric(df_foia['FOIA_Complex_Days'], errors='coerce')

print(f"  FOIA data: {len(df_foia)} rows")
print(f"  Simple Days range: {df_foia['FOIA_Simple_Days'].min():.1f} to {df_foia['FOIA_Simple_Days'].max():.1f}")
print(f"  Complex Days range: {df_foia['FOIA_Complex_Days'].min():.1f} to {df_foia['FOIA_Complex_Days'].max():.1f}")

# =============================================================================
# 4. MERGE ALL DATASETS
# =============================================================================
print("\n[4/5] Merging all datasets...")

# Start with base
df = df_base.copy()

# Merge officers
df = df.merge(df_officers, on='FY', how='left')

# Merge FOIA
df = df.merge(df_foia, on='FY', how='left')

print(f"  Merged dataset: {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# 5. CREATE NORMALIZED VARIABLES
# =============================================================================
print("\n[5/5] Creating normalized variables...")

# Log transform Policy_Count
df['Policy_Count_Log'] = np.log(df['Policy_Count'] + 1)
print(f"\n  Policy_Count_Log: {df['Policy_Count_Log'].min():.3f} to {df['Policy_Count_Log'].max():.3f}")

# Z-score Total_Civilians
civ_mean = df['Total_Civilians'].mean()
civ_std = df['Total_Civilians'].std()
df['Total_Civilians_Z'] = (df['Total_Civilians'] - civ_mean) / civ_std
print(f"  Total_Civilians_Z (mean={civ_mean:.0f}, std={civ_std:.0f}): {df['Total_Civilians_Z'].min():.3f} to {df['Total_Civilians_Z'].max():.3f}")

# Z-score Total_PAS
pas_mean = df['Total_PAS'].mean()
pas_std = df['Total_PAS'].std()
df['Total_PAS_Z'] = (df['Total_PAS'] - pas_mean) / pas_std
print(f"  Total_PAS_Z (mean={pas_mean:.0f}, std={pas_std:.0f}): {df['Total_PAS_Z'].min():.3f} to {df['Total_PAS_Z'].max():.3f}")

# Z-score FOIA Simple Days
if 'FOIA_Simple_Days' in df.columns:
    foia_simple_mean = df['FOIA_Simple_Days'].mean()
    foia_simple_std = df['FOIA_Simple_Days'].std()
    df['FOIA_Simple_Days_Z'] = (df['FOIA_Simple_Days'] - foia_simple_mean) / foia_simple_std
    print(f"  FOIA_Simple_Days_Z (mean={foia_simple_mean:.0f}, std={foia_simple_std:.0f}): {df['FOIA_Simple_Days_Z'].min():.3f} to {df['FOIA_Simple_Days_Z'].max():.3f}")

# Z-score FOIA Complex Days
if 'FOIA_Complex_Days' in df.columns:
    foia_complex_mean = df['FOIA_Complex_Days'].mean()
    foia_complex_std = df['FOIA_Complex_Days'].std()
    df['FOIA_Complex_Days_Z'] = (df['FOIA_Complex_Days'] - foia_complex_mean) / foia_complex_std
    print(f"  FOIA_Complex_Days_Z (mean={foia_complex_mean:.0f}, std={foia_complex_std:.0f}): {df['FOIA_Complex_Days_Z'].min():.3f} to {df['FOIA_Complex_Days_Z'].max():.3f}")

# =============================================================================
# SELECT FINAL COLUMNS (EXCLUDE SERVICE-SPECIFIC)
# =============================================================================
print("\n" + "=" * 100)
print("FINAL DATASET COMPOSITION")
print("=" * 100)

# Exclude service-specific breakdowns
exclude_cols = ['Civ_Army', 'Civ_Navy', 'Civ_AirForce']
final_cols = [col for col in df.columns if col not in exclude_cols]

df_final = df[final_cols].copy()

# Save
output_path = 'data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx'
df_final.to_excel(output_path, index=False)

print(f"\n[OK] Saved to: {output_path}")
print(f"\nRows: {len(df_final)}")
print(f"Columns: {len(df_final.columns)}")

print("\n" + "=" * 100)
print("COLUMNS INCLUDED ({})".format(len(df_final.columns)))
print("=" * 100)

# Group columns by category
original_vars = ['FY', 'Total_Military', 'Total_Civilians', 'Total_PAS', 'Policy_Count',
                 'HOR_Republican', 'Senate_Republican', 'President_Republican',
                 'GDP_Growth', 'Major_Conflict']
enlisted_pcts = [c for c in df_final.columns if c.startswith('E') and c.endswith('_Pct')]
officer_pcts = [c for c in df_final.columns if c.startswith('O') and c.endswith('_Pct')]
foia_vars = [c for c in df_final.columns if 'FOIA' in c]
normalized_vars = [c for c in df_final.columns if c.endswith('_Z') or c.endswith('_Log')]

print("\nORIGINAL VARIABLES:")
for col in original_vars:
    if col in df_final.columns:
        print(f"  - {col}")

print(f"\nENLISTED PERCENTAGES ({len(enlisted_pcts)}):")
for col in sorted(enlisted_pcts):
    print(f"  - {col}")

print(f"\nOFFICER PERCENTAGES ({len(officer_pcts)}):")
for col in sorted(officer_pcts):
    print(f"  - {col}")

print(f"\nFOIA VARIABLES ({len(foia_vars)}):")
for col in foia_vars:
    print(f"  - {col}")

print(f"\nNORMALIZED VARIABLES ({len(normalized_vars)}):")
for col in normalized_vars:
    print(f"  - {col}")

print("\n" + "=" * 100)
print("NORMALIZATION PARAMETERS")
print("=" * 100)

norm_params = pd.DataFrame({
    'Variable': ['Policy_Count', 'Total_Civilians', 'Total_PAS', 'FOIA_Simple_Days', 'FOIA_Complex_Days'],
    'Transform': ['Log', 'Z-score', 'Z-score', 'Z-score', 'Z-score'],
    'Mean': [np.nan, civ_mean, pas_mean, foia_simple_mean, foia_complex_mean],
    'Std': [np.nan, civ_std, pas_std, foia_simple_std, foia_complex_std],
    'Normalized_Column': ['Policy_Count_Log', 'Total_Civilians_Z', 'Total_PAS_Z',
                          'FOIA_Simple_Days_Z', 'FOIA_Complex_Days_Z']
})

print(norm_params.to_string(index=False))

norm_params.to_excel('data/analysis/normalization_parameters_v10.6_FULL.xlsx', index=False)

print("\n" + "=" * 100)
print("COMPLETE!")
print("=" * 100)
