"""
Check Stationarity of All Variables
Determine if VAR differencing approach is correct
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("STATIONARITY TEST: Should All Variables Be Differenced?")
print("=" * 100)

# Load data
df = pd.read_excel('data/analysis/complete_relative_dataset.xlsx')
df['Total_Civilians'] = df['Civ_Army'] + df['Civ_Navy'] + df['Civ_AirForce']
df['Policy_Count_Log'] = np.log(df['Policy_Count'] + 1)

variables = [
    'Policy_Count_Log', 'Total_Civilians', 'O5_LtColCDR_Pct',
    'O4_MajorLTCDR_Pct', 'E5_Pct', 'O6_ColCAPT_Pct',
    'GDP_Growth', 'Major_Conflict', 'Total_PAS'
]

# Current differencing approach
diff_vars_current = ['Policy_Count_Log', 'O5_LtColCDR_Pct', 'O4_MajorLTCDR_Pct',
                     'E5_Pct', 'O6_ColCAPT_Pct', 'Major_Conflict', 'Total_PAS']

print("\nCURRENT APPROACH:")
print(f"  Differenced: {', '.join(diff_vars_current)}")
print(f"  NOT Differenced: Total_Civilians, GDP_Growth")

# =============================================================================
# STATIONARITY TESTS
# =============================================================================
print("\n" + "=" * 100)
print("STATIONARITY TESTS (LEVELS)")
print("=" * 100)
print("\nADF Test: H0 = Unit Root (non-stationary)")
print("KPSS Test: H0 = Stationary")
print("\nIf ADF p-value < 0.05 AND KPSS p-value > 0.05 --> STATIONARY (use levels)")
print("Otherwise --> NON-STATIONARY (need to difference)")
print("-" * 100)

results = []

for var in variables:
    data = df[var].dropna()

    # ADF test (H0: unit root exists)
    try:
        adf_result = adfuller(data, maxlag=4, regression='ct')
        adf_stat = adf_result[0]
        adf_pval = adf_result[1]
        adf_conclusion = "Stationary" if adf_pval < 0.05 else "Non-Stationary"
    except Exception as e:
        adf_stat = np.nan
        adf_pval = np.nan
        adf_conclusion = "ERROR"

    # KPSS test (H0: stationary)
    try:
        kpss_result = kpss(data, regression='ct', nlags=4)
        kpss_stat = kpss_result[0]
        kpss_pval = kpss_result[1]
        kpss_conclusion = "Stationary" if kpss_pval > 0.05 else "Non-Stationary"
    except Exception as e:
        kpss_stat = np.nan
        kpss_pval = np.nan
        kpss_conclusion = "ERROR"

    # Final decision
    if adf_pval < 0.05 and kpss_pval > 0.05:
        final = "STATIONARY - Use LEVELS"
    elif adf_pval >= 0.05 or kpss_pval <= 0.05:
        final = "NON-STATIONARY - DIFFERENCE"
    else:
        final = "INCONCLUSIVE"

    # Check against current approach
    currently_differenced = var in diff_vars_current
    should_difference = final == "NON-STATIONARY - DIFFERENCE"

    if currently_differenced == should_difference:
        match = "[OK] CORRECT"
    else:
        match = "[X] WRONG!"

    results.append({
        'Variable': var,
        'ADF_pvalue': adf_pval,
        'ADF_Conclusion': adf_conclusion,
        'KPSS_pvalue': kpss_pval,
        'KPSS_Conclusion': kpss_conclusion,
        'Final_Decision': final,
        'Currently_Differenced': 'YES' if currently_differenced else 'NO',
        'Match': match
    })

    print(f"\n{var}:")
    print(f"  ADF p-value: {adf_pval:.4f} --> {adf_conclusion}")
    print(f"  KPSS p-value: {kpss_pval:.4f} --> {kpss_conclusion}")
    print(f"  DECISION: {final}")
    print(f"  Current approach: {'DIFFERENCED' if currently_differenced else 'LEVELS'} {match}")

results_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("SUMMARY TABLE")
print("=" * 100)
print(results_df[['Variable', 'Final_Decision', 'Currently_Differenced', 'Match']].to_string(index=False))

# =============================================================================
# TEST FIRST DIFFERENCES
# =============================================================================
print("\n" + "=" * 100)
print("STATIONARITY TESTS (FIRST DIFFERENCES)")
print("=" * 100)
print("Testing if differencing makes non-stationary variables stationary")
print("-" * 100)

diff_results = []

for var in variables:
    data = df[var].dropna()
    data_diff = data.diff().dropna()

    try:
        adf_result = adfuller(data_diff, maxlag=4, regression='ct')
        adf_pval = adf_result[1]
        adf_conclusion = "Stationary" if adf_pval < 0.05 else "Non-Stationary"
    except:
        adf_pval = np.nan
        adf_conclusion = "ERROR"

    try:
        kpss_result = kpss(data_diff, regression='ct', nlags=4)
        kpss_pval = kpss_result[1]
        kpss_conclusion = "Stationary" if kpss_pval > 0.05 else "Non-Stationary"
    except:
        kpss_pval = np.nan
        kpss_conclusion = "ERROR"

    if adf_pval < 0.05 and kpss_pval > 0.05:
        final = "STATIONARY after differencing"
    else:
        final = "Still non-stationary"

    diff_results.append({
        'Variable': var,
        'ADF_pvalue_diff': adf_pval,
        'KPSS_pvalue_diff': kpss_pval,
        'Diff_Conclusion': final
    })

    print(f"\n{var} (differenced):")
    print(f"  ADF p-value: {adf_pval:.4f}")
    print(f"  KPSS p-value: {kpss_pval:.4f}")
    print(f"  CONCLUSION: {final}")

diff_df = pd.DataFrame(diff_results)

# =============================================================================
# RECOMMENDATION
# =============================================================================
print("\n" + "=" * 100)
print("RECOMMENDATION")
print("=" * 100)

wrong_count = len(results_df[results_df['Match'].str.contains('WRONG')])

if wrong_count > 0:
    print(f"\n[!] WARNING: {wrong_count} variables are incorrectly specified!")
    print("\nVariables with WRONG treatment:")
    wrong_vars = results_df[results_df['Match'].str.contains('WRONG')]
    for _, row in wrong_vars.iterrows():
        print(f"  - {row['Variable']}: {row['Final_Decision']}, but currently {row['Currently_Differenced']}")

    print("\n" + "=" * 100)
    print("RECOMMENDED FIX:")
    print("=" * 100)

    # Find which variables should be differenced
    should_diff = results_df[results_df['Final_Decision'].str.contains('DIFFERENCE')]['Variable'].tolist()
    should_level = results_df[results_df['Final_Decision'].str.contains('LEVELS')]['Variable'].tolist()

    if len(should_level) == 0:
        print("\n[OK] ALL VARIABLES ARE NON-STATIONARY")
        print("[OK] Difference ALL 9 variables for a consistent VAR specification")
        print("\nRecommended code:")
        print("```python")
        print("# Difference ALL variables")
        print("diff_vars = [")
        for var in variables:
            print(f"    '{var}',")
        print("]")
        print("for var in diff_vars:")
        print("    data[var] = data[var].diff()")
        print("```")
    else:
        print("\n[!] MIXED specification may be valid IF:")
        print(f"   Stationary variables ({len(should_level)}): {', '.join(should_level)}")
        print(f"   Non-stationary variables ({len(should_diff)}): {', '.join(should_diff)}")
        print("\nHowever, this requires careful interpretation and may cause NetLogo instability.")
else:
    print("\n[OK] Current differencing approach is CORRECT based on stationarity tests")

# Save results
results_df.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/stationarity_check_UPDATED.xlsx', index=False)
print("\n[OK] Results saved to stationarity_check_UPDATED.xlsx")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
