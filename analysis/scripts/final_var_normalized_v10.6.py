"""
FINAL VAR ANALYSIS v10.6 - NORMALIZED
Z-score normalization of Total_Civilians and Total_PAS
Complete re-run with proper scaling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera
import networkx as nx
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 100)
print("FINAL VAR ANALYSIS v10.6 - NORMALIZED")
print("Z-Score Normalization: Total_Civilians + Total_PAS")
print("=" * 100)

# Create output directory
output_dir = 'data/analysis/FINAL_v10.6_NORMALIZED'
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# DATA PREPARATION
# =============================================================================
df = pd.read_excel('data/analysis/complete_relative_dataset.xlsx')
df['Total_Civilians'] = df['Civ_Army'] + df['Civ_Navy'] + df['Civ_AirForce']
df['Policy_Count_Log'] = np.log(df['Policy_Count'] + 1)

print("\n" + "=" * 100)
print("DATA PREPARATION")
print("=" * 100)

# Z-score normalization for count variables
print("\nNormalizing count variables with Z-scores:")
print("-" * 100)

# Total_Civilians
civ_mean = df['Total_Civilians'].mean()
civ_std = df['Total_Civilians'].std()
df['Total_Civilians_Z'] = (df['Total_Civilians'] - civ_mean) / civ_std
print(f"\nTotal_Civilians:")
print(f"  Original range: {df['Total_Civilians'].min():.0f} to {df['Total_Civilians'].max():.0f}")
print(f"  Mean: {civ_mean:.0f}, Std: {civ_std:.0f}")
print(f"  Z-scored range: {df['Total_Civilians_Z'].min():.3f} to {df['Total_Civilians_Z'].max():.3f}")

# Total_PAS
pas_mean = df['Total_PAS'].mean()
pas_std = df['Total_PAS'].std()
df['Total_PAS_Z'] = (df['Total_PAS'] - pas_mean) / pas_std
print(f"\nTotal_PAS:")
print(f"  Original range: {df['Total_PAS'].min():.0f} to {df['Total_PAS'].max():.0f}")
print(f"  Mean: {pas_mean:.0f}, Std: {pas_std:.0f}")
print(f"  Z-scored range: {df['Total_PAS_Z'].min():.3f} to {df['Total_PAS_Z'].max():.3f}")

# Define variables (using z-scored versions)
variables = [
    'Policy_Count_Log', 'Total_Civilians_Z', 'O5_LtColCDR_Pct',
    'O4_MajorLTCDR_Pct', 'E5_Pct', 'O6_ColCAPT_Pct',
    'GDP_Growth', 'Major_Conflict', 'Total_PAS_Z'
]

var_names_clean = {
    'Policy_Count_Log': 'Log_Policy_Count',
    'Total_Civilians_Z': 'Total_Civilians_Z',
    'O5_LtColCDR_Pct': 'O5_LtColCDR_Pct',
    'O4_MajorLTCDR_Pct': 'O4_MajorLTCDR_Pct',
    'E5_Pct': 'E5_Pct',
    'O6_ColCAPT_Pct': 'O6_ColCAPT_Pct',
    'GDP_Growth': 'GDP_Growth',
    'Major_Conflict': 'Major_Conflict',
    'Total_PAS_Z': 'Total_PAS_Z'
}

data = df[variables].copy()

# Apply differencing (all except GDP_Growth which is already stationary)
print("\n" + "=" * 100)
print("DIFFERENCING FOR STATIONARITY")
print("=" * 100)

diff_vars = ['Policy_Count_Log', 'Total_Civilians_Z', 'O5_LtColCDR_Pct',
             'O4_MajorLTCDR_Pct', 'E5_Pct', 'O6_ColCAPT_Pct',
             'Major_Conflict', 'Total_PAS_Z']

print(f"\nDifferenced (8 variables): {', '.join(diff_vars)}")
print(f"Kept in levels (1 variable): GDP_Growth")

for var in diff_vars:
    data[var] = data[var].diff()

data = data.dropna()

print(f"\nData prepared: {len(data)} observations after differencing")
print(f"Variables: {len(variables)}")

# Save normalization parameters for NetLogo
norm_params = pd.DataFrame({
    'Variable': ['Total_Civilians', 'Total_PAS'],
    'Mean': [civ_mean, pas_mean],
    'Std': [civ_std, pas_std],
    'Min_Original': [df['Total_Civilians'].min(), df['Total_PAS'].min()],
    'Max_Original': [df['Total_Civilians'].max(), df['Total_PAS'].max()]
})
norm_params.to_excel(f'{output_dir}/normalization_parameters.xlsx', index=False)

# =============================================================================
# STATIONARITY VERIFICATION
# =============================================================================
print("\n" + "=" * 100)
print("STATIONARITY VERIFICATION (Post-Transformation)")
print("=" * 100)

stationarity_results = []

for var in variables:
    test_data = data[var].dropna()

    # ADF test
    try:
        adf_result = adfuller(test_data, maxlag=4, regression='ct')
        adf_pval = adf_result[1]
        adf_stat = "STATIONARY" if adf_pval < 0.05 else "Non-Stationary"
    except:
        adf_pval = np.nan
        adf_stat = "ERROR"

    # KPSS test
    try:
        kpss_result = kpss(test_data, regression='ct', nlags=4)
        kpss_pval = kpss_result[1]
        kpss_stat = "STATIONARY" if kpss_pval > 0.05 else "Non-Stationary"
    except:
        kpss_pval = np.nan
        kpss_stat = "ERROR"

    conclusion = "STATIONARY" if (adf_pval < 0.05 and kpss_pval > 0.05) else "CHECK"

    stationarity_results.append({
        'Variable': var_names_clean[var],
        'ADF_pvalue': adf_pval,
        'ADF_Status': adf_stat,
        'KPSS_pvalue': kpss_pval,
        'KPSS_Status': kpss_stat,
        'Final_Status': conclusion
    })

    print(f"\n{var}:")
    print(f"  ADF: p={adf_pval:.4f} ({adf_stat})")
    print(f"  KPSS: p={kpss_pval:.4f} ({kpss_stat})")
    print(f"  --> {conclusion}")

stationarity_df = pd.DataFrame(stationarity_results)
stationarity_df.to_excel(f'{output_dir}/stationarity_verification.xlsx', index=False)

# =============================================================================
# LAG ORDER SELECTION
# =============================================================================
print("\n" + "=" * 100)
print("LAG ORDER SELECTION")
print("=" * 100)

lag_results = []

for lag in range(1, 6):
    try:
        model = VAR(data)
        result = model.fit(lag)
        lag_results.append({
            'Lag': lag,
            'AIC': result.aic,
            'BIC': result.bic,
            'HQIC': result.hqic,
            'Status': 'SUCCESS'
        })
        print(f"Lag {lag}: AIC={result.aic:.4f}, BIC={result.bic:.4f}, HQIC={result.hqic:.4f}")
    except Exception as e:
        lag_results.append({
            'Lag': lag,
            'AIC': np.nan,
            'BIC': np.nan,
            'HQIC': np.nan,
            'Status': f'FAILED: {str(e)[:30]}'
        })
        print(f"Lag {lag}: FAILED - {str(e)[:50]}")

lag_df = pd.DataFrame(lag_results)
valid_lags = lag_df[lag_df['Status'] == 'SUCCESS']

if len(valid_lags) > 0:
    optimal_aic = valid_lags.loc[valid_lags['AIC'].idxmin()]
    optimal_bic = valid_lags.loc[valid_lags['BIC'].idxmin()]

    print("\n" + "=" * 100)
    print("OPTIMAL LAG SELECTION:")
    print("=" * 100)
    print(f"By AIC: Lag {int(optimal_aic['Lag'])} (AIC={optimal_aic['AIC']:.4f})")
    print(f"By BIC: Lag {int(optimal_bic['Lag'])} (BIC={optimal_bic['BIC']:.4f})")

    # Use lag with lowest AIC
    optimal_lag = int(optimal_aic['Lag'])
    print(f"\nUsing Lag {optimal_lag} (AIC-optimal)")
else:
    print("\n[ERROR] No valid lag orders!")
    optimal_lag = 4  # fallback

lag_df.to_excel(f'{output_dir}/lag_selection.xlsx', index=False)

# =============================================================================
# ESTIMATE VAR MODEL
# =============================================================================
print("\n" + "=" * 100)
print(f"ESTIMATING VAR({optimal_lag}) MODEL")
print("=" * 100)

model = VAR(data)
results = model.fit(optimal_lag)

print(f"\nModel estimated successfully!")
print(f"  Lag order: {optimal_lag}")
print(f"  Observations used: {results.nobs}")
print(f"  AIC: {results.aic:.4f}")
print(f"  BIC: {results.bic:.4f}")
print(f"  Log-Likelihood: {results.llf:.4f}")

# Save model summary
with open(f'{output_dir}/model_summary.txt', 'w') as f:
    f.write(f"VAR({optimal_lag}) MODEL - NORMALIZED VERSION v10.6\n")
    f.write("=" * 100 + "\n")
    f.write("Z-Score Normalized: Total_Civilians, Total_PAS\n")
    f.write("Differenced: 8 variables (all except GDP_Growth)\n\n")
    f.write(str(results.summary()))

print("[OK] Model summary saved")

# =============================================================================
# GRANGER CAUSALITY ANALYSIS
# =============================================================================
print("\n" + "=" * 100)
print("GRANGER CAUSALITY ANALYSIS (Lag-by-Lag)")
print("=" * 100)

granger_results = []

for cause_var in variables:
    for effect_var in variables:
        if cause_var == effect_var:
            continue

        try:
            test_result = grangercausalitytests(
                data[[effect_var, cause_var]].dropna(),
                maxlag=optimal_lag,
                verbose=False
            )

            for lag in range(1, optimal_lag + 1):
                if lag in test_result:
                    f_stat = test_result[lag][0]['ssr_ftest'][0]
                    p_value = test_result[lag][0]['ssr_ftest'][1]

                    granger_results.append({
                        'Cause': var_names_clean[cause_var],
                        'Effect': var_names_clean[effect_var],
                        'Lag': lag,
                        'F_statistic': f_stat,
                        'p_value': p_value,
                        'Significant_10pct': p_value < 0.10,
                        'Significant_5pct': p_value < 0.05,
                        'Significant_1pct': p_value < 0.01,
                        'Confidence': 'Very High (p<0.01)' if p_value < 0.01 else
                                     'High (p<0.05)' if p_value < 0.05 else
                                     'Moderate (p<0.10)' if p_value < 0.10 else
                                     'Not Significant'
                    })
        except Exception as e:
            continue

granger_df = pd.DataFrame(granger_results)

print(f"\nTotal tests: {len(granger_df)}")
print(f"Significant at 10%: {len(granger_df[granger_df['Significant_10pct']==True])}")
print(f"Significant at 5%: {len(granger_df[granger_df['Significant_5pct']==True])}")
print(f"Significant at 1%: {len(granger_df[granger_df['Significant_1pct']==True])}")

granger_df.to_excel(f'{output_dir}/granger_by_lag_all.xlsx', index=False)

granger_sig = granger_df[granger_df['Significant_5pct'] == True].copy()
granger_sig = granger_sig.sort_values('p_value')
granger_sig.to_excel(f'{output_dir}/granger_by_lag_significant.xlsx', index=False)

print("\nTop 20 Most Significant Relationships:")
print("-" * 100)
if len(granger_sig) > 0:
    print(granger_sig.head(20)[['Cause', 'Effect', 'Lag', 'F_statistic', 'p_value']].to_string(index=False))
else:
    print("No significant relationships found!")

# Aggregate by relationship
relationship_summary = []

for (cause, effect), group in granger_df.groupby(['Cause', 'Effect']):
    sig_lags = group[group['Significant_5pct'] == True]['Lag'].tolist()

    if len(sig_lags) > 0:
        strongest = group.loc[group['F_statistic'].idxmax()]

        relationship_summary.append({
            'Cause': cause,
            'Effect': effect,
            'Significant_Lags': ','.join(map(str, sig_lags)),
            'Num_Significant_Lags': len(sig_lags),
            'Strongest_Lag': int(strongest['Lag']),
            'Max_F_statistic': strongest['F_statistic'],
            'Min_p_value': group['p_value'].min()
        })

relationship_df = pd.DataFrame(relationship_summary)
relationship_df = relationship_df.sort_values('Max_F_statistic', ascending=False)
relationship_df.to_excel(f'{output_dir}/granger_summary_by_relationship.xlsx', index=False)

print(f"\n[OK] {len(relationship_df)} significant causal relationships identified")

# =============================================================================
# FORECAST ERROR VARIANCE DECOMPOSITION
# =============================================================================
print("\n" + "=" * 100)
print("FORECAST ERROR VARIANCE DECOMPOSITION")
print("=" * 100)

try:
    fevd = results.fevd(10)
    max_step = fevd.decomp.shape[0] - 1

    fevd_data = []
    for target_var in variables:
        target_idx = variables.index(target_var)
        fevd_values = fevd.decomp[max_step, target_idx, :] * 100

        for source_idx, source_var in enumerate(variables):
            fevd_data.append({
                'Target': var_names_clean[target_var],
                'Source': var_names_clean[source_var],
                'Variance_Explained_Pct': fevd_values[source_idx],
                'Step': max_step + 1
            })

    fevd_df = pd.DataFrame(fevd_data)
    fevd_pivot = fevd_df.pivot(index='Target', columns='Source', values='Variance_Explained_Pct')
    fevd_pivot = fevd_pivot.round(2)

    print(f"\nFEVD at Step {max_step + 1}:")
    print("-" * 100)
    print(fevd_pivot.to_string())

    fevd_df.to_excel(f'{output_dir}/fevd_all_variables.xlsx', index=False)
    fevd_pivot.to_excel(f'{output_dir}/fevd_matrix.xlsx')

    # O4 breakdown
    o4_fevd = fevd_pivot.loc['O4_MajorLTCDR_Pct'].sort_values(ascending=False)
    print("\n" + "=" * 100)
    print("O4 BUREAUCRATIC BLOAT DRIVERS:")
    print("=" * 100)
    print(o4_fevd.to_string())

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    sns.heatmap(fevd_pivot, annot=True, fmt='.1f', cmap='YlOrRd',
               cbar_kws={'label': 'Variance Explained (%)'}, ax=ax,
               linewidths=0.5, linecolor='gray')
    ax.set_title(f'Forecast Error Variance Decomposition (Step {max_step + 1})\nVAR({optimal_lag}) - Normalized v10.6',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Source Variable', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Variable', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fevd_heatmap.png', dpi=300, bbox_inches='tight')

    print("\n[OK] FEVD analysis complete")
    fevd_success = True

except Exception as e:
    print(f"\n[WARNING] FEVD failed: {e}")
    fevd_success = False

# =============================================================================
# RESIDUAL DIAGNOSTICS
# =============================================================================
print("\n" + "=" * 100)
print("RESIDUAL DIAGNOSTICS")
print("=" * 100)

residuals = results.resid
diagnostics = []

for var_idx, var in enumerate(variables):
    resid = residuals.iloc[:, var_idx].dropna()
    var_clean = var_names_clean[var]

    # Ljung-Box
    lb_test = acorr_ljungbox(resid, lags=[5], return_df=True)
    lb_pval = lb_test.loc[5, 'lb_pvalue']

    # ARCH
    try:
        arch_test = het_arch(resid, nlags=5)
        arch_pval = arch_test[1]
    except:
        arch_pval = np.nan

    # Jarque-Bera
    jb_stat, jb_pval = jarque_bera(resid)

    diagnostics.append({
        'Variable': var_clean,
        'LB_PValue': lb_pval,
        'ARCH_PValue': arch_pval,
        'JB_PValue': jb_pval,
        'Autocorr': 'PASS' if lb_pval >= 0.05 else 'FAIL',
        'Heterosked': 'PASS' if arch_pval >= 0.05 else 'FAIL',
        'Normality': 'PASS' if jb_pval >= 0.05 else 'FAIL'
    })

diag_df = pd.DataFrame(diagnostics)
print("\nDiagnostic Test Results (PASS = p >= 0.05):")
print("-" * 100)
print(diag_df[['Variable', 'Autocorr', 'Heterosked', 'Normality']].to_string(index=False))

diag_df.to_excel(f'{output_dir}/residual_diagnostics.xlsx', index=False)

# =============================================================================
# NETWORK VISUALIZATION
# =============================================================================
print("\n" + "=" * 100)
print("GENERATING NETWORK DIAGRAM")
print("=" * 100)

G = nx.DiGraph()
G.add_nodes_from([var_names_clean[v] for v in variables])

for _, row in relationship_df.iterrows():
    G.add_edge(row['Cause'], row['Effect'],
               weight=row['Max_F_statistic'],
               lags=row['Significant_Lags'])

# Variable categories
categories = {
    'Exogenous': ['GDP_Growth', 'Major_Conflict'],
    'Administrative': ['Log_Policy_Count', 'Total_PAS_Z', 'Total_Civilians_Z'],
    'Military_Officers': ['O4_MajorLTCDR_Pct', 'O5_LtColCDR_Pct', 'O6_ColCAPT_Pct'],
    'Military_Enlisted': ['E5_Pct']
}

colors = {
    'Exogenous': '#e74c3c',
    'Administrative': '#3498db',
    'Military_Officers': '#2ecc71',
    'Military_Enlisted': '#f39c12'
}

node_to_category = {}
for category, nodes in categories.items():
    for node in nodes:
        node_to_category[node] = category

# Calculate centrality
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())
betweenness = nx.betweenness_centrality(G)

centrality_df = pd.DataFrame({
    'Variable': list(G.nodes()),
    'In_Degree': [in_degree[n] for n in G.nodes()],
    'Out_Degree': [out_degree[n] for n in G.nodes()],
    'Total_Degree': [in_degree[n] + out_degree[n] for n in G.nodes()],
    'Betweenness': [betweenness[n] for n in G.nodes()]
}).sort_values('Total_Degree', ascending=False)

centrality_df.to_excel(f'{output_dir}/network_centrality.xlsx', index=False)

print("\nNetwork Statistics:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Density: {nx.density(G):.3f}")

print("\nTop 5 by Total Degree:")
print(centrality_df.head()[['Variable', 'In_Degree', 'Out_Degree', 'Total_Degree']].to_string(index=False))

# Create network plot
fig, ax = plt.subplots(figsize=(20, 14), facecolor='white')

# Hierarchical layout
pos = {}
exog_vars = [v for v in G.nodes() if node_to_category.get(v) == 'Exogenous']
admin_vars = [v for v in G.nodes() if node_to_category.get(v) == 'Administrative']
officer_vars = [v for v in G.nodes() if node_to_category.get(v) == 'Military_Officers']
enlisted_vars = [v for v in G.nodes() if node_to_category.get(v) == 'Military_Enlisted']

for i, var in enumerate(exog_vars):
    pos[var] = (i * 3 - 1.5, 3)
for i, var in enumerate(admin_vars):
    pos[var] = (i * 2.5 - 2.5, 2)
for i, var in enumerate(officer_vars):
    pos[var] = (i * 2.5 - 2.5, 1)
for i, var in enumerate(enlisted_vars):
    pos[var] = (0, 0)

# Draw edges
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
max_weight = max(weights) if weights else 1
edge_widths = [2 + (w / max_weight) * 6 for w in weights]

for (u, v), width in zip(edges, edge_widths):
    x1, y1 = pos[u]
    x2, y2 = pos[v]

    arrow = plt.Arrow(x1, y1, x2-x1, y2-y1, width=width/15,
                     color='gray', alpha=0.6)
    ax.add_patch(arrow)

# Draw nodes
for node in G.nodes():
    category = node_to_category.get(node, 'Administrative')
    color = colors[category]
    degree = G.degree(node)
    node_size = 0.35 + degree * 0.05

    x, y = pos[node]
    circle = plt.Circle((x, y), node_size, color=color, alpha=0.8,
                       edgecolor='white', linewidth=3, zorder=2)
    ax.add_patch(circle)

    label = node.replace('_', '\n')
    ax.text(x, y, label, fontsize=9, fontweight='bold',
           ha='center', va='center', color='white', zorder=3)

ax.set_xlim(-6, 6)
ax.set_ylim(-1.5, 4)
ax.axis('off')

ax.text(0, 4.2, 'DoD Bureaucratic Growth Causal Network\nVAR - Normalized v10.6',
       fontsize=18, fontweight='bold', ha='center')
ax.text(0, 3.85, f'{len(relationship_df)} Significant Causal Relationships (p<0.05)',
       fontsize=12, ha='center', style='italic', color='gray')

plt.tight_layout()
plt.savefig(f'{output_dir}/network_diagram.png', dpi=300, bbox_inches='tight')

print("[OK] Network diagram saved")

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("CREATING EXECUTIVE SUMMARY")
print("=" * 100)

summary = []
summary.append("=" * 100)
summary.append("EXECUTIVE SUMMARY - VAR MODEL v10.6 NORMALIZED")
summary.append("=" * 100)
summary.append("")
summary.append("MODEL SPECIFICATION:")
summary.append(f"  VAR Order: {optimal_lag}")
summary.append(f"  Observations: {results.nobs}")
summary.append(f"  AIC: {results.aic:.4f}")
summary.append(f"  BIC: {results.bic:.4f}")
summary.append("")
summary.append("NORMALIZATION:")
summary.append(f"  Total_Civilians: Z-scored (mean={civ_mean:.0f}, std={civ_std:.0f})")
summary.append(f"  Total_PAS: Z-scored (mean={pas_mean:.0f}, std={pas_std:.0f})")
summary.append("")
summary.append("DIFFERENCING:")
summary.append(f"  Differenced: {', '.join([var_names_clean[v] for v in diff_vars])}")
summary.append("  Kept in levels: GDP_Growth (already stationary)")
summary.append("")
summary.append("GRANGER CAUSALITY:")
summary.append(f"  Total relationships tested: {len(granger_df)}")
summary.append(f"  Significant (p<0.05): {len(granger_sig)}")
summary.append(f"  Aggregated relationships: {len(relationship_df)}")
summary.append("")

if len(relationship_df) > 0:
    summary.append("TOP 10 CAUSAL RELATIONSHIPS:")
    summary.append("-" * 100)
    for idx, row in relationship_df.head(10).iterrows():
        summary.append(f"  {row['Cause']} -> {row['Effect']}")
        summary.append(f"    Lags: {row['Significant_Lags']}, F={row['Max_F_statistic']:.2f}, p={row['Min_p_value']:.4f}")
    summary.append("")

if fevd_success:
    summary.append("O4 BUREAUCRATIC BLOAT - TOP DRIVERS (FEVD):")
    summary.append("-" * 100)
    top5 = o4_fevd.head(5)
    for var, val in top5.items():
        summary.append(f"  {var}: {val:.2f}%")
    summary.append("")

summary.append("DIAGNOSTICS:")
summary.append("-" * 100)
autocorr_pass = len(diag_df[diag_df['Autocorr'] == 'PASS'])
summary.append(f"  Autocorrelation: {autocorr_pass}/{len(diag_df)} variables PASS")
hetero_pass = len(diag_df[diag_df['Heterosked'] == 'PASS'])
summary.append(f"  Heteroskedasticity: {hetero_pass}/{len(diag_df)} variables PASS")
summary.append("")

summary.append("NETWORK TOPOLOGY:")
summary.append("-" * 100)
top_hub = centrality_df.iloc[0]
summary.append(f"  Primary hub: {top_hub['Variable']} (degree={int(top_hub['Total_Degree'])})")
summary.append(f"  Network density: {nx.density(G):.3f}")
summary.append("")

summary.append("=" * 100)
summary.append("FILES GENERATED:")
summary.append("=" * 100)
summary.append(f"  Output directory: {output_dir}/")
summary.append("  1. model_summary.txt - Full regression output")
summary.append("  2. normalization_parameters.xlsx - Z-score parameters")
summary.append("  3. stationarity_verification.xlsx - Post-transformation tests")
summary.append("  4. lag_selection.xlsx - AIC/BIC comparison")
summary.append("  5. granger_by_lag_all.xlsx - All Granger tests")
summary.append("  6. granger_by_lag_significant.xlsx - Significant only")
summary.append("  7. granger_summary_by_relationship.xlsx - Aggregated")
if fevd_success:
    summary.append("  8. fevd_all_variables.xlsx - FEVD data")
    summary.append("  9. fevd_matrix.xlsx - FEVD pivot table")
    summary.append(" 10. fevd_heatmap.png - Visualization")
summary.append(" 11. residual_diagnostics.xlsx - Model diagnostics")
summary.append(" 12. network_centrality.xlsx - Network measures")
summary.append(" 13. network_diagram.png - Causal network")
summary.append(" 14. executive_summary.txt - This file")
summary.append("=" * 100)

summary_text = '\n'.join(summary)
print(summary_text)

with open(f'{output_dir}/executive_summary.txt', 'w') as f:
    f.write(summary_text)

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
print(f"\nAll outputs saved to: {output_dir}/")
