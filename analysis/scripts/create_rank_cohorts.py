"""
Group military ranks into cohorts and calculate percentages + z-scores

Cohorts:
- Junior Enlisted: E1, E2, E3, E4
- Middle Enlisted: E5, E6
- Senior Enlisted: E7, E8, E9
- Company Grade Officers: O1, O2, O3
- Field Grade Officers: O4, O5, O6
- GOFOs (General/Flag Officers): O7, O8, O9, O10
- Warrant Officers: W1, W2, W3, W4, W5
"""

import pandas as pd
import numpy as np

print("=" * 100)
print("CREATING MILITARY RANK COHORTS")
print("=" * 100)

# Define rank mappings (exact labels from the file)
rank_mappings = {
    'Junior_Enlisted': ['E-1', 'E-2', 'E-3', 'E-4'],
    'Middle_Enlisted': ['E-5', 'E-6'],
    'Senior_Enlisted': ['E-7', 'E-8', 'E-9'],
    'Company_Grade_Officers': [
        '2ND LIEUTENANT - ENSIGN',
        '1ST LIEUTENANT - LIEUTENANT (JG)',
        'CAPTAIN - LIEUTENANT'
    ],
    'Field_Grade_Officers': [
        'MAJOR - LT COMMANDER',
        'LIEUTENANT COL - COMMANDER',
        'COLONEL - CAPTAIN'
    ],
    'GOFOs': [
        'BRIG GENERAL - REAR ADMIRAL (L)',
        'MAJ GENERAL - REAR ADMIRAL (U)',
        'LT GENERAL - VICE ADMIRAL',
        'GENERAL - ADMIRAL'
    ],
    'Warrant_Officers': [
        'WARRANT OFFICER W-1',
        'CHIEF WARRANT OFFICER W-2',
        'CHIEF WARRANT OFFICER W-3',
        'CHIEF WARRANT OFFICER W-4',
        'CHIEF WARRANT OFFICER W-5'
    ]
}

# Load the Excel file
print("\n[1/4] Loading AD Strengths data...")
xls = pd.ExcelFile('data/AD_strengths/AD_Strengths_FY1987-2024_normalized.xlsx')

all_years_data = []

for year in range(1987, 2025):
    sheet_name = str(year)
    if sheet_name not in xls.sheet_names:
        continue

    df_year = pd.read_excel(xls, sheet_name=sheet_name)

    # Get Total DoD column
    total_col = 'Total DoD'
    if total_col not in df_year.columns:
        continue

    print(f"  Processing FY {year}...", end='')

    year_data = {'FY': year}

    # Get total military (TOTAL OFFICER + TOTAL ENLISTED, excluding cadets)
    total_officer_mask = df_year['Rank/Grade'].astype(str).str.strip() == 'TOTAL OFFICER'
    total_enlisted_mask = df_year['Rank/Grade'].astype(str).str.strip() == 'TOTAL ENLISTED'

    total_military = 0
    if total_officer_mask.any():
        total_officer = float(df_year.loc[total_officer_mask, total_col].values[0])
        total_military += total_officer
    if total_enlisted_mask.any():
        total_enlisted = float(df_year.loc[total_enlisted_mask, total_col].values[0])
        total_military += total_enlisted

    if total_military == 0:
        print(" [SKIP - No total found]")
        continue

    # Extract counts for each cohort
    for cohort_name, rank_labels in rank_mappings.items():
        cohort_total = 0

        for rank_label in rank_labels:
            # Clean and match rank names
            mask = df_year['Rank/Grade'].astype(str).str.upper().str.strip() == rank_label.upper()

            if mask.any():
                value = df_year.loc[mask, total_col].values[0]
                try:
                    cohort_total += float(value)
                except:
                    pass

        # Calculate percentage
        year_data[f'{cohort_name}_Pct'] = (cohort_total / total_military) * 100
        year_data[f'{cohort_name}_Count'] = cohort_total

    year_data['Total_Military'] = total_military
    all_years_data.append(year_data)
    print(f" [OK - Total: {total_military:,.0f}]")

# Create dataframe
df_cohorts = pd.DataFrame(all_years_data)

print(f"\n[2/4] Extracted {len(df_cohorts)} years of cohort data")
print(f"  Cohorts: {len(rank_mappings)}")

# Calculate z-scores for each cohort
print("\n[3/4] Calculating z-scores for each cohort...")
print("  " + "-" * 96)

for cohort in rank_mappings.keys():
    pct_col = f'{cohort}_Pct'
    z_col = f'{cohort}_Z'

    if pct_col in df_cohorts.columns:
        mean_val = df_cohorts[pct_col].mean()
        std_val = df_cohorts[pct_col].std()

        df_cohorts[z_col] = (df_cohorts[pct_col] - mean_val) / std_val

        print(f"  {cohort:30s} (mean={mean_val:6.2f}%, std={std_val:5.2f}%)")

# Save to Excel
print("\n[4/4] Saving cohort data...")

# Full dataset
output_path = 'data/analysis/rank_cohorts_full.xlsx'
df_cohorts.to_excel(output_path, index=False)
print(f"  [OK] Full dataset: {output_path}")

# Simplified dataset (FY, percentages, z-scores only)
simplified_cols = ['FY']
for cohort in rank_mappings.keys():
    simplified_cols.append(f'{cohort}_Pct')
    simplified_cols.append(f'{cohort}_Z')

df_simplified = df_cohorts[simplified_cols].copy()
output_path_simple = 'data/analysis/rank_cohorts_simplified.xlsx'
df_simplified.to_excel(output_path_simple, index=False)
print(f"  [OK] Simplified dataset: {output_path_simple}")

# Display summary
print("\n" + "=" * 100)
print("COHORT SUMMARY")
print("=" * 100)

for cohort in rank_mappings.keys():
    pct_col = f'{cohort}_Pct'
    if pct_col in df_cohorts.columns:
        mean_pct = df_cohorts[pct_col].mean()
        min_pct = df_cohorts[pct_col].min()
        max_pct = df_cohorts[pct_col].max()
        print(f"\n{cohort}:")
        print(f"  Mean: {mean_pct:6.2f}%")
        print(f"  Range: {min_pct:6.2f}% to {max_pct:6.2f}%")

print("\n" + "=" * 100)
print(f"COMPLETE - {len(df_cohorts)} years, {len(rank_mappings)} cohorts")
print("=" * 100)
print("\nCohort definitions:")
print("  Junior_Enlisted: E1, E2, E3, E4")
print("  Middle_Enlisted: E5, E6")
print("  Senior_Enlisted: E7, E8, E9")
print("  Company_Grade_Officers: O1, O2, O3")
print("  Field_Grade_Officers: O4, O5, O6")
print("  GOFOs: O7, O8, O9, O10")
print("  Warrant_Officers: W1, W2, W3, W4, W5")
