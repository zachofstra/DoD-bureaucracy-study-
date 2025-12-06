"""
FINAL VAR ANALYSIS
LOG(Policy_Count) with Lag 4 (Optimal)
Complete lag-by-lag Granger causality breakdown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("FINAL VAR ANALYSIS: LOG(POLICY_COUNT) + LAG 4")
print("Complete Lag-by-Lag Causal Relationship Breakdown")
print("=" * 100)

# =============================================================================
# DATA PREPARATION
# =============================================================================
df = pd.read_excel('data/analysis/complete_relative_dataset.xlsx')
df['Total_Civilians'] = df['Civ_Army'] + df['Civ_Navy'] + df['Civ_AirForce']
df['Policy_Count_Log'] = np.log(df['Policy_Count'] + 1)

variables = [
    'Policy_Count_Log', 'Total_Civilians', 'O5_LtColCDR_Pct',
    'O4_MajorLTCDR_Pct', 'E5_Pct', 'O6_ColCAPT_Pct',
    'GDP_Growth', 'Major_Conflict', 'Total_PAS'
]

# Variable names for output (replace log notation)
var_names_clean = {
    'Policy_Count_Log': 'Log_Policy_Count',
    'Total_Civilians': 'Total_Civilians',
    'O5_LtColCDR_Pct': 'O5_LtColCDR_Pct',
    'O4_MajorLTCDR_Pct': 'O4_MajorLTCDR_Pct',
    'E5_Pct': 'E5_Pct',
    'O6_ColCAPT_Pct': 'O6_ColCAPT_Pct',
    'GDP_Growth': 'GDP_Growth',
    'Major_Conflict': 'Major_Conflict',
    'Total_PAS': 'Total_PAS'
}

data = df[variables].copy()

# Apply differencing
diff_vars = ['Policy_Count_Log', 'O5_LtColCDR_Pct', 'O4_MajorLTCDR_Pct',
             'E5_Pct', 'O6_ColCAPT_Pct', 'Major_Conflict', 'Total_PAS']

for var in diff_vars:
    data[var] = data[var].diff()

data = data.dropna()

print(f"\nData: {len(data)} observations")
print(f"Variables: {len(variables)}")

# =============================================================================
# ESTIMATE VAR(4) MODEL
# =============================================================================
print("\n" + "=" * 100)
print("ESTIMATING VAR(4) MODEL")
print("=" * 100)

model = VAR(data)
results = model.fit(4)

print(f"\nModel estimated successfully!")
print(f"  Lag order: 4")
print(f"  Observations used: {results.nobs}")
print(f"  AIC: {results.aic:.4f}")
print(f"  BIC: {results.bic:.4f}")
print(f"  Log-Likelihood: {results.llf:.4f}")

# Save model summary
with open('data/analysis/FINAL_LAG4_LOG/model_summary.txt', 'w') as f:
    f.write("VAR(4) MODEL WITH LOG(POLICY_COUNT)\n")
    f.write("=" * 100 + "\n\n")
    f.write(str(results.summary()))

print("[OK] Model summary saved")

# =============================================================================
# LAG-BY-LAG GRANGER CAUSALITY ANALYSIS
# =============================================================================
print("\n" + "=" * 100)
print("LAG-BY-LAG GRANGER CAUSALITY ANALYSIS")
print("=" * 100)

granger_results = []

for cause_var in variables:
    for effect_var in variables:
        if cause_var == effect_var:
            continue

        try:
            # Run Granger test with maxlag=4
            test_result = grangercausalitytests(
                data[[effect_var, cause_var]].dropna(),
                maxlag=4,
                verbose=False
            )

            # Extract results for EACH individual lag
            for lag in [1, 2, 3, 4]:
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

# Save all results
granger_df.to_excel('data/analysis/FINAL_LAG4_LOG/granger_by_lag_all.xlsx', index=False)

# Save significant only
granger_sig = granger_df[granger_df['Significant_5pct'] == True].copy()
granger_sig = granger_sig.sort_values('p_value')
granger_sig.to_excel('data/analysis/FINAL_LAG4_LOG/granger_by_lag_significant.xlsx', index=False)

print("\nTop 20 Most Significant Lag-Specific Relationships:")
print("-" * 100)
print(granger_sig.head(20)[['Cause', 'Effect', 'Lag', 'F_statistic', 'p_value', 'Confidence']].to_string(index=False))

# =============================================================================
# AGGREGATE GRANGER RESULTS BY RELATIONSHIP
# =============================================================================
print("\n" + "=" * 100)
print("AGGREGATED GRANGER CAUSALITY (Any Lag Significant)")
print("=" * 100)

# Group by cause-effect pair
relationship_summary = []

for (cause, effect), group in granger_df.groupby(['Cause', 'Effect']):
    # Find which lags are significant
    sig_lags = group[group['Significant_5pct'] == True]['Lag'].tolist()

    if len(sig_lags) > 0:
        # Get strongest effect
        strongest = group.loc[group['F_statistic'].idxmax()]

        relationship_summary.append({
            'Cause': cause,
            'Effect': effect,
            'Significant_Lags': ','.join(map(str, sig_lags)),
            'Num_Significant_Lags': len(sig_lags),
            'Strongest_Lag': int(strongest['Lag']),
            'Max_F_statistic': strongest['F_statistic'],
            'Min_p_value': group['p_value'].min(),
            'Overall_Confidence': 'Very High' if group['p_value'].min() < 0.01 else
                                 'High' if group['p_value'].min() < 0.05 else
                                 'Moderate'
        })

relationship_df = pd.DataFrame(relationship_summary)
relationship_df = relationship_df.sort_values('Max_F_statistic', ascending=False)

print(f"\nTotal significant relationships (any lag): {len(relationship_df)}")
print("\nTop 15 Causal Relationships (by strongest F-statistic):")
print("-" * 100)
print(relationship_df.head(15)[['Cause', 'Effect', 'Significant_Lags', 'Strongest_Lag',
                                'Max_F_statistic', 'Min_p_value']].to_string(index=False))

relationship_df.to_excel('data/analysis/FINAL_LAG4_LOG/granger_summary_by_relationship.xlsx', index=False)

# =============================================================================
# LAG PATTERN ANALYSIS
# =============================================================================
print("\n" + "=" * 100)
print("LAG PATTERN ANALYSIS")
print("=" * 100)

# Count significant effects by lag
lag_counts = granger_sig.groupby('Lag').size()
print("\nSignificant effects by lag:")
for lag, count in lag_counts.items():
    print(f"  Lag {lag}: {count} significant relationships")

# Identify immediate vs delayed effects
immediate = relationship_df[relationship_df['Strongest_Lag'] == 1]
delayed = relationship_df[relationship_df['Strongest_Lag'].isin([3, 4])]

print(f"\nImmediate effects (strongest at lag 1): {len(immediate)}")
print(f"Delayed effects (strongest at lag 3-4): {len(delayed)}")

print("\nImmediate effects (Lag 1 strongest):")
print(immediate[['Cause', 'Effect', 'Max_F_statistic']].head(10).to_string(index=False))

print("\nDelayed effects (Lag 3-4 strongest):")
print(delayed[['Cause', 'Effect', 'Strongest_Lag', 'Max_F_statistic']].head(10).to_string(index=False))

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

    fevd_df.to_excel('data/analysis/FINAL_LAG4_LOG/fevd_all_variables.xlsx', index=False)
    fevd_pivot.to_excel('data/analysis/FINAL_LAG4_LOG/fevd_matrix.xlsx')

    # O4 breakdown
    o4_fevd = fevd_pivot.loc['O4_MajorLTCDR_Pct'].sort_values(ascending=False)
    print("\n" + "=" * 100)
    print("O4 BUREAUCRATIC BLOAT DRIVERS:")
    print("=" * 100)
    print(o4_fevd.to_string())

    exog_total = o4_fevd.get('GDP_Growth', 0) + o4_fevd.get('Major_Conflict', 0)
    print(f"\nExogenous (GDP + Conflict): {exog_total:.2f}%")
    print(f"Self-perpetuation: {o4_fevd.get('O4_MajorLTCDR_Pct', 0):.2f}%")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    sns.heatmap(fevd_pivot, annot=True, fmt='.1f', cmap='YlOrRd',
               cbar_kws={'label': 'Variance Explained (%)'}, ax=ax,
               linewidths=0.5, linecolor='gray')
    ax.set_title(f'Forecast Error Variance Decomposition (Step {max_step + 1})\nVAR(4) with LOG(Policy_Count)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Source Variable', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Variable', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('data/analysis/FINAL_LAG4_LOG/fevd_heatmap.png', dpi=300, bbox_inches='tight')

    print("\n[OK] FEVD analysis complete")
    fevd_success = True

except Exception as e:
    print(f"\n[WARNING] FEVD failed: {e}")
    print("Continuing with other analyses...")
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

diag_df.to_excel('data/analysis/FINAL_LAG4_LOG/residual_diagnostics.xlsx', index=False)

# =============================================================================
# NETWORK VISUALIZATION
# =============================================================================
print("\n" + "=" * 100)
print("GENERATING NETWORK DIAGRAM")
print("=" * 100)

# Use aggregated relationships (any lag significant)
G = nx.DiGraph()
G.add_nodes_from([var_names_clean[v] for v in variables])

for _, row in relationship_df.iterrows():
    G.add_edge(row['Cause'], row['Effect'],
               weight=row['Max_F_statistic'],
               lags=row['Significant_Lags'])

# Variable categories
categories = {
    'Exogenous': ['GDP_Growth', 'Major_Conflict'],
    'Administrative': ['Log_Policy_Count', 'Total_PAS', 'Total_Civilians'],
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

centrality_df.to_excel('data/analysis/FINAL_LAG4_LOG/network_centrality.xlsx', index=False)

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

    label = node.replace('_', '\n').replace('Log\nPolicy\nCount', 'LOG\nPolicy')
    ax.text(x, y, label, fontsize=9, fontweight='bold',
           ha='center', va='center', color='white', zorder=3)

ax.set_xlim(-6, 6)
ax.set_ylim(-1.5, 4)
ax.axis('off')

ax.text(0, 4.2, 'DoD Bureaucratic Growth Causal Network\nVAR(4) with LOG(Policy_Count)',
       fontsize=18, fontweight='bold', ha='center')
ax.text(0, 3.85, f'{len(relationship_df)} Significant Causal Relationships (p<0.05)',
       fontsize=12, ha='center', style='italic', color='gray')

plt.tight_layout()
plt.savefig('data/analysis/FINAL_LAG4_LOG/network_diagram.png', dpi=300, bbox_inches='tight')

print("[OK] Network diagram saved")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 100)
print("ANALYSIS COMPLETE - SUMMARY")
print("=" * 100)

print("\nMODEL SPECIFICATION:")
print(f"  Transformation: LOG(Policy_Count + 1)")
print(f"  Lag order: 4 (includes lags 1, 2, 3, 4)")
print(f"  AIC: {results.aic:.4f}")
print(f"  BIC: {results.bic:.4f}")

print("\nGRANGER CAUSALITY:")
print(f"  Total relationships tested: {len(granger_df)}")
print(f"  Significant (p<0.05): {len(granger_sig)}")
print(f"  Aggregated relationships: {len(relationship_df)}")

if fevd_success:
    print("\nFEVD (O4 Bureaucratic Bloat):")
    top3 = o4_fevd.head(3)
    for i, (var, val) in enumerate(top3.items(), 1):
        print(f"  {i}. {var}: {val:.2f}%")

print("\nDIAGNOSTICS:")
autocorr_pass = len(diag_df[diag_df['Autocorr'] == 'PASS'])
print(f"  Autocorrelation: {autocorr_pass}/{len(diag_df)} variables pass")
hetero_pass = len(diag_df[diag_df['Heterosked'] == 'PASS'])
print(f"  Heteroskedasticity: {hetero_pass}/{len(diag_df)} variables pass")

print("\nNETWORK TOPOLOGY:")
top_hub = centrality_df.iloc[0]
print(f"  Primary hub: {top_hub['Variable']} (degree={int(top_hub['Total_Degree'])})")
print(f"  Network density: {nx.density(G):.3f}")

print("\n" + "=" * 100)
print("FILES GENERATED:")
print("=" * 100)
print("  1. model_summary.txt")
print("  2. granger_by_lag_all.xlsx")
print("  3. granger_by_lag_significant.xlsx")
print("  4. granger_summary_by_relationship.xlsx")
if fevd_success:
    print("  5. fevd_all_variables.xlsx")
    print("  6. fevd_matrix.xlsx")
    print("  7. fevd_heatmap.png")
print("  8. residual_diagnostics.xlsx")
print("  9. network_centrality.xlsx")
print(" 10. network_diagram.png")
print("\n" + "=" * 100)
