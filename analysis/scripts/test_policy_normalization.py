"""
Test Policy_Count Normalization
Compare linear vs log transformation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from scipy import stats

df = pd.read_excel('data/analysis/complete_relative_dataset.xlsx')
df['Total_Civilians'] = df['Civ_Army'] + df['Civ_Navy'] + df['Civ_AirForce']

print("=" * 100)
print("POLICY_COUNT NORMALIZATION ANALYSIS")
print("=" * 100)

# Check Policy_Count distribution
policy = df['Policy_Count'].dropna()

print("\nPOLICY_COUNT STATISTICS:")
print("-" * 100)
print(policy.describe())
print(f"\nRange: {policy.min():.0f} to {policy.max():.0f} (factor of {policy.max()/policy.min():.1f}x)")
print(f"Coefficient of Variation: {policy.std() / policy.mean():.2f}")
print(f"Skewness: {stats.skew(policy):.2f}")
print(f"Kurtosis: {stats.kurtosis(policy):.2f}")

# Create transformations
df['Policy_Count_Linear'] = df['Policy_Count']
df['Policy_Count_Log'] = np.log(df['Policy_Count'] + 1)  # +1 to handle zeros
df['Policy_Count_Sqrt'] = np.sqrt(df['Policy_Count'])

print("\nCOMPARING TRANSFORMATIONS:")
print("-" * 100)
for transform in ['Policy_Count_Linear', 'Policy_Count_Log', 'Policy_Count_Sqrt']:
    data = df[transform].dropna()
    print(f"\n{transform}:")
    print(f"  Mean: {data.mean():.2f}")
    print(f"  Std: {data.std():.2f}")
    print(f"  CV: {data.std()/data.mean():.2f}")
    print(f"  Skewness: {stats.skew(data):.2f}")
    print(f"  Range: {data.min():.2f} to {data.max():.2f}")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')

# Row 1: Time series
ax = axes[0, 0]
ax.plot(df['FY'], df['Policy_Count_Linear'], 'o-', linewidth=2, markersize=6)
ax.set_title('Linear Policy Count', fontsize=12, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(df['FY'], df['Policy_Count_Log'], 'o-', linewidth=2, markersize=6, color='orange')
ax.set_title('Log(Policy Count + 1)', fontsize=12, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Log(Count + 1)')
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.plot(df['FY'], df['Policy_Count_Sqrt'], 'o-', linewidth=2, markersize=6, color='green')
ax.set_title('Sqrt(Policy Count)', fontsize=12, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Sqrt(Count)')
ax.grid(True, alpha=0.3)

# Row 2: Distributions
ax = axes[1, 0]
ax.hist(df['Policy_Count_Linear'].dropna(), bins=20, edgecolor='black', alpha=0.7)
ax.set_title('Linear Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Policy Count')
ax.set_ylabel('Frequency')
ax.axvline(df['Policy_Count_Linear'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax.legend()

ax = axes[1, 1]
ax.hist(df['Policy_Count_Log'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='orange')
ax.set_title('Log Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Log(Policy Count + 1)')
ax.set_ylabel('Frequency')
ax.axvline(df['Policy_Count_Log'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax.legend()

ax = axes[1, 2]
ax.hist(df['Policy_Count_Sqrt'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='green')
ax.set_title('Sqrt Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Sqrt(Policy Count)')
ax.set_ylabel('Frequency')
ax.axvline(df['Policy_Count_Sqrt'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax.legend()

fig.suptitle('Policy Count Transformation Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/policy_count_transformations.png',
           dpi=300, bbox_inches='tight')

print("\n[OK] Visualization saved: policy_count_transformations.png")

# =============================================================================
# RE-RUN VAR WITH LOG-TRANSFORMED POLICY_COUNT
# =============================================================================
print("\n" + "=" * 100)
print("RE-RUNNING VAR ANALYSIS WITH LOG(POLICY_COUNT)")
print("=" * 100)

variables_log = [
    'Policy_Count_Log', 'Total_Civilians', 'O5_LtColCDR_Pct',
    'O4_MajorLTCDR_Pct', 'E5_Pct', 'O6_ColCAPT_Pct',
    'GDP_Growth', 'Major_Conflict', 'Total_PAS'
]

data_log = df[variables_log].copy()

# Apply differencing (same as before)
diff_vars = ['Policy_Count_Log', 'O5_LtColCDR_Pct', 'O4_MajorLTCDR_Pct',
             'E5_Pct', 'O6_ColCAPT_Pct', 'Major_Conflict', 'Total_PAS']

for var in diff_vars:
    data_log[var] = data_log[var].diff()

data_log = data_log.dropna()

print(f"Data prepared: {len(data_log)} observations")

# Test multiple lags
results_summary = []

for lag in [1, 2, 3, 4, 5]:
    try:
        model = VAR(data_log)
        result = model.fit(lag)
        results_summary.append({
            'Lag': lag,
            'AIC': result.aic,
            'BIC': result.bic,
            'Status': 'SUCCESS'
        })
        print(f"Lag {lag}: AIC={result.aic:.4f}, BIC={result.bic:.4f} - SUCCESS")
    except Exception as e:
        results_summary.append({
            'Lag': lag,
            'AIC': np.nan,
            'BIC': np.nan,
            'Status': f'FAILED'
        })
        print(f"Lag {lag}: FAILED - {str(e)[:50]}")

results_df = pd.DataFrame(results_summary)
print("\n" + "=" * 100)
print("LAG SELECTION WITH LOG(POLICY_COUNT):")
print("=" * 100)
print(results_df.to_string(index=False))

# Find optimal lag
valid = results_df[results_df['Status'] == 'SUCCESS']
if len(valid) > 0:
    optimal_aic = valid.loc[valid['AIC'].idxmin()]
    optimal_bic = valid.loc[valid['BIC'].idxmin()]
    print(f"\nOptimal by AIC: Lag {int(optimal_aic['Lag'])} (AIC={optimal_aic['AIC']:.4f})")
    print(f"Optimal by BIC: Lag {int(optimal_bic['Lag'])} (BIC={optimal_bic['BIC']:.4f})")

results_df.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/policy_log_lag_selection.xlsx',
                   index=False)

# =============================================================================
# COMPARE ORIGINAL VS LOG MODELS
# =============================================================================
print("\n" + "=" * 100)
print("COMPARISON: LINEAR vs LOG POLICY_COUNT")
print("=" * 100)

# Original with linear Policy_Count
variables_linear = [
    'Policy_Count', 'Total_Civilians', 'O5_LtColCDR_Pct',
    'O4_MajorLTCDR_Pct', 'E5_Pct', 'O6_ColCAPT_Pct',
    'GDP_Growth', 'Major_Conflict', 'Total_PAS'
]

data_linear = df[variables_linear].copy()
for var in diff_vars:
    if var == 'Policy_Count_Log':
        data_linear['Policy_Count'] = data_linear['Policy_Count'].diff()
    elif var in data_linear.columns:
        data_linear[var] = data_linear[var].diff()

data_linear = data_linear.dropna()

print("\nTesting Lag 2 with both transformations:")
print("-" * 100)

# Linear
try:
    model_linear = VAR(data_linear)
    result_linear = model_linear.fit(2)
    print(f"LINEAR Policy_Count - Lag 2:")
    print(f"  AIC: {result_linear.aic:.4f}")
    print(f"  BIC: {result_linear.bic:.4f}")
    print(f"  Status: SUCCESS")
    linear_success = True
except Exception as e:
    print(f"LINEAR Policy_Count - Lag 2: FAILED - {e}")
    linear_success = False

# Log
try:
    model_log = VAR(data_log)
    result_log = model_log.fit(2)
    print(f"\nLOG Policy_Count - Lag 2:")
    print(f"  AIC: {result_log.aic:.4f}")
    print(f"  BIC: {result_log.bic:.4f}")
    print(f"  Status: SUCCESS")
    log_success = True
except Exception as e:
    print(f"\nLOG Policy_Count - Lag 2: FAILED - {e}")
    log_success = False

if linear_success and log_success:
    print("\n" + "=" * 100)
    print("RECOMMENDATION:")
    print("=" * 100)
    if result_log.aic < result_linear.aic:
        print("USE LOG(POLICY_COUNT) - Better model fit (lower AIC)")
    else:
        print("KEEP LINEAR POLICY_COUNT - Better model fit (lower AIC)")

    print(f"\nAIC difference: {abs(result_log.aic - result_linear.aic):.4f}")
    print(f"BIC difference: {abs(result_log.bic - result_linear.bic):.4f}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
