"""
Bootstrap Analysis for Network Centrality Measures
Tests robustness of network topology findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats

print("=" * 100)
print("BOOTSTRAP ANALYSIS: NETWORK CENTRALITY ROBUSTNESS")
print("=" * 100)

# Load Granger causality results
granger = pd.read_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/granger_significant.xlsx')

# Filter for 5% significance
granger_sig = granger[granger['Significant_5pct'] == True].copy()

print(f"\nSignificant relationships: {len(granger_sig)}")
print(f"Bootstrap replications: 10,000")
print("\nStarting bootstrap analysis...")

# Bootstrap function
def bootstrap_network(df, n_bootstrap=10000, seed=42):
    """
    Bootstrap network centrality measures by resampling Granger relationships
    """
    np.random.seed(seed)

    # Store bootstrap results
    centrality_results = {
        'Variable': [],
        'Measure': [],
        'Original': [],
        'Bootstrap_Mean': [],
        'Bootstrap_Std': [],
        'CI_Lower_2.5': [],
        'CI_Upper_97.5': [],
        'Robust': []
    }

    # Get original network
    G_orig = nx.DiGraph()
    all_vars = list(set(df['Cause'].tolist() + df['Effect'].tolist()))
    G_orig.add_nodes_from(all_vars)

    for _, row in df.iterrows():
        G_orig.add_edge(row['Cause'], row['Effect'])

    # Calculate original centrality
    orig_in_deg = dict(G_orig.in_degree())
    orig_out_deg = dict(G_orig.out_degree())
    orig_betweenness = nx.betweenness_centrality(G_orig)
    orig_eigenvector = nx.eigenvector_centrality(G_orig, max_iter=1000)

    # Bootstrap loop
    boot_in_deg = {var: [] for var in all_vars}
    boot_out_deg = {var: [] for var in all_vars}
    boot_between = {var: [] for var in all_vars}
    boot_eigen = {var: [] for var in all_vars}

    for boot_iter in range(n_bootstrap):
        if boot_iter % 1000 == 0:
            print(f"  Bootstrap iteration {boot_iter}/{n_bootstrap}...")

        # Resample edges with replacement
        boot_sample = df.sample(n=len(df), replace=True)

        # Build bootstrap network
        G_boot = nx.DiGraph()
        G_boot.add_nodes_from(all_vars)

        for _, row in boot_sample.iterrows():
            G_boot.add_edge(row['Cause'], row['Effect'])

        # Calculate centrality
        in_deg_boot = dict(G_boot.in_degree())
        out_deg_boot = dict(G_boot.out_degree())
        between_boot = nx.betweenness_centrality(G_boot)

        try:
            eigen_boot = nx.eigenvector_centrality(G_boot, max_iter=1000)
        except:
            eigen_boot = {var: 0 for var in all_vars}

        # Store results
        for var in all_vars:
            boot_in_deg[var].append(in_deg_boot.get(var, 0))
            boot_out_deg[var].append(out_deg_boot.get(var, 0))
            boot_between[var].append(between_boot.get(var, 0))
            boot_eigen[var].append(eigen_boot.get(var, 0))

    # Calculate statistics
    for var in all_vars:
        # In-Degree
        boot_mean = np.mean(boot_in_deg[var])
        boot_std = np.std(boot_in_deg[var])
        ci_lower = np.percentile(boot_in_deg[var], 2.5)
        ci_upper = np.percentile(boot_in_deg[var], 97.5)
        robust = 'YES' if ci_lower > 0 or ci_upper < 0 else 'NO'

        centrality_results['Variable'].append(var)
        centrality_results['Measure'].append('In-Degree')
        centrality_results['Original'].append(orig_in_deg[var])
        centrality_results['Bootstrap_Mean'].append(boot_mean)
        centrality_results['Bootstrap_Std'].append(boot_std)
        centrality_results['CI_Lower_2.5'].append(ci_lower)
        centrality_results['CI_Upper_97.5'].append(ci_upper)
        centrality_results['Robust'].append(robust)

        # Out-Degree
        boot_mean = np.mean(boot_out_deg[var])
        boot_std = np.std(boot_out_deg[var])
        ci_lower = np.percentile(boot_out_deg[var], 2.5)
        ci_upper = np.percentile(boot_out_deg[var], 97.5)
        robust = 'YES' if ci_lower > 0 or ci_upper < 0 else 'NO'

        centrality_results['Variable'].append(var)
        centrality_results['Measure'].append('Out-Degree')
        centrality_results['Original'].append(orig_out_deg[var])
        centrality_results['Bootstrap_Mean'].append(boot_mean)
        centrality_results['Bootstrap_Std'].append(boot_std)
        centrality_results['CI_Lower_2.5'].append(ci_lower)
        centrality_results['CI_Upper_97.5'].append(ci_upper)
        centrality_results['Robust'].append(robust)

        # Betweenness
        boot_mean = np.mean(boot_between[var])
        boot_std = np.std(boot_between[var])
        ci_lower = np.percentile(boot_between[var], 2.5)
        ci_upper = np.percentile(boot_between[var], 97.5)
        robust = 'YES' if ci_lower > 0 or ci_upper < 0 else 'NO'

        centrality_results['Variable'].append(var)
        centrality_results['Measure'].append('Betweenness')
        centrality_results['Original'].append(orig_betweenness[var])
        centrality_results['Bootstrap_Mean'].append(boot_mean)
        centrality_results['Bootstrap_Std'].append(boot_std)
        centrality_results['CI_Lower_2.5'].append(ci_lower)
        centrality_results['CI_Upper_97.5'].append(ci_upper)
        centrality_results['Robust'].append(robust)

        # Eigenvector
        boot_mean = np.mean(boot_eigen[var])
        boot_std = np.std(boot_eigen[var])
        ci_lower = np.percentile(boot_eigen[var], 2.5)
        ci_upper = np.percentile(boot_eigen[var], 97.5)
        robust = 'YES' if ci_lower > 0 or ci_upper < 0 else 'NO'

        centrality_results['Variable'].append(var)
        centrality_results['Measure'].append('Eigenvector')
        centrality_results['Original'].append(orig_eigenvector[var])
        centrality_results['Bootstrap_Mean'].append(boot_mean)
        centrality_results['Bootstrap_Std'].append(boot_std)
        centrality_results['CI_Lower_2.5'].append(ci_lower)
        centrality_results['CI_Upper_97.5'].append(ci_upper)
        centrality_results['Robust'].append(robust)

    return pd.DataFrame(centrality_results), boot_in_deg, boot_out_deg

# Run bootstrap
boot_df, boot_in, boot_out = bootstrap_network(granger_sig)

# Save results
boot_df.to_excel('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/bootstrap_centrality.xlsx', index=False)

print("\n" + "=" * 100)
print("BOOTSTRAP RESULTS SUMMARY")
print("=" * 100)

# Print key findings
print("\nIN-DEGREE CENTRALITY (Number of incoming causal edges):")
print("-" * 100)
in_deg_results = boot_df[boot_df['Measure'] == 'In-Degree'].copy()
in_deg_results = in_deg_results.sort_values('Original', ascending=False)
print(f"{'Variable':<25} {'Original':<10} {'Bootstrap Mean':<15} {'95% CI':<25} {'Robust':<10}")
print("-" * 100)
for _, row in in_deg_results.iterrows():
    ci_str = f"[{row['CI_Lower_2.5']:.2f}, {row['CI_Upper_97.5']:.2f}]"
    print(f"{row['Variable']:<25} {row['Original']:<10} {row['Bootstrap_Mean']:<15.2f} {ci_str:<25} {row['Robust']:<10}")

print("\nOUT-DEGREE CENTRALITY (Number of outgoing causal edges):")
print("-" * 100)
out_deg_results = boot_df[boot_df['Measure'] == 'Out-Degree'].copy()
out_deg_results = out_deg_results.sort_values('Original', ascending=False)
print(f"{'Variable':<25} {'Original':<10} {'Bootstrap Mean':<15} {'95% CI':<25} {'Robust':<10}")
print("-" * 100)
for _, row in out_deg_results.iterrows():
    ci_str = f"[{row['CI_Lower_2.5']:.2f}, {row['CI_Upper_97.5']:.2f}]"
    print(f"{row['Variable']:<25} {row['Original']:<10} {row['Bootstrap_Mean']:<15.2f} {ci_str:<25} {row['Robust']:<10}")

print("\nBETWEENNESS CENTRALITY (Bridge role in network):")
print("-" * 100)
between_results = boot_df[boot_df['Measure'] == 'Betweenness'].copy()
between_results = between_results.sort_values('Original', ascending=False).head(5)
print(f"{'Variable':<25} {'Original':<12} {'Bootstrap Mean':<15} {'95% CI':<30} {'Robust':<10}")
print("-" * 100)
for _, row in between_results.iterrows():
    ci_str = f"[{row['CI_Lower_2.5']:.4f}, {row['CI_Upper_97.5']:.4f}]"
    print(f"{row['Variable']:<25} {row['Original']:<12.4f} {row['Bootstrap_Mean']:<15.4f} {ci_str:<30} {row['Robust']:<10}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='white')

# Plot 1: In-Degree with confidence intervals
ax = axes[0, 0]
in_deg_plot = in_deg_results.sort_values('Bootstrap_Mean', ascending=True)
y_pos = np.arange(len(in_deg_plot))

ax.barh(y_pos, in_deg_plot['Bootstrap_Mean'], color='steelblue', alpha=0.7, label='Bootstrap Mean')
ax.errorbar(in_deg_plot['Bootstrap_Mean'], y_pos,
            xerr=[in_deg_plot['Bootstrap_Mean'] - in_deg_plot['CI_Lower_2.5'],
                  in_deg_plot['CI_Upper_97.5'] - in_deg_plot['Bootstrap_Mean']],
            fmt='none', ecolor='black', capsize=5, alpha=0.8)
ax.scatter(in_deg_plot['Original'], y_pos, color='red', s=100, zorder=5,
          marker='D', label='Original', edgecolors='darkred', linewidths=1.5)

ax.set_yticks(y_pos)
ax.set_yticklabels([v.replace('_', ' ') for v in in_deg_plot['Variable']], fontsize=9)
ax.set_xlabel('In-Degree', fontsize=11, fontweight='bold')
ax.set_title('In-Degree Centrality with 95% Bootstrap CI', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Plot 2: Out-Degree with confidence intervals
ax = axes[0, 1]
out_deg_plot = out_deg_results.sort_values('Bootstrap_Mean', ascending=True)
y_pos = np.arange(len(out_deg_plot))

ax.barh(y_pos, out_deg_plot['Bootstrap_Mean'], color='coral', alpha=0.7, label='Bootstrap Mean')
ax.errorbar(out_deg_plot['Bootstrap_Mean'], y_pos,
            xerr=[out_deg_plot['Bootstrap_Mean'] - out_deg_plot['CI_Lower_2.5'],
                  out_deg_plot['CI_Upper_97.5'] - out_deg_plot['Bootstrap_Mean']],
            fmt='none', ecolor='black', capsize=5, alpha=0.8)
ax.scatter(out_deg_plot['Original'], y_pos, color='red', s=100, zorder=5,
          marker='D', label='Original', edgecolors='darkred', linewidths=1.5)

ax.set_yticks(y_pos)
ax.set_yticklabels([v.replace('_', ' ') for v in out_deg_plot['Variable']], fontsize=9)
ax.set_xlabel('Out-Degree', fontsize=11, fontweight='bold')
ax.set_title('Out-Degree Centrality with 95% Bootstrap CI', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Plot 3: Bootstrap distribution for Total_Civilians (highest centrality)
ax = axes[1, 0]
ax.hist(boot_in['Total_Civilians'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(np.mean(boot_in['Total_Civilians']), color='red', linestyle='--',
          linewidth=2, label=f"Mean = {np.mean(boot_in['Total_Civilians']):.2f}")
ax.axvline(np.percentile(boot_in['Total_Civilians'], 2.5), color='orange',
          linestyle=':', linewidth=2, label='2.5%ile')
ax.axvline(np.percentile(boot_in['Total_Civilians'], 97.5), color='orange',
          linestyle=':', linewidth=2, label='97.5%ile')
ax.set_xlabel('In-Degree', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Bootstrap Distribution: Total_Civilians In-Degree\n(10,000 replications)',
            fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Bootstrap distribution for E5_Pct (high out-degree)
ax = axes[1, 1]
ax.hist(boot_out['E5_Pct'], bins=50, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(np.mean(boot_out['E5_Pct']), color='red', linestyle='--',
          linewidth=2, label=f"Mean = {np.mean(boot_out['E5_Pct']):.2f}")
ax.axvline(np.percentile(boot_out['E5_Pct'], 2.5), color='orange',
          linestyle=':', linewidth=2, label='2.5%ile')
ax.axvline(np.percentile(boot_out['E5_Pct'], 97.5), color='orange',
          linestyle=':', linewidth=2, label='97.5%ile')
ax.set_xlabel('Out-Degree', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Bootstrap Distribution: E5_Pct Out-Degree\n(10,000 replications)',
            fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Bootstrap Analysis of Network Centrality Measures\n10,000 Replications with Replacement',
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/analysis/FINAL_TOP9_WITH_EXOGENOUS/bootstrap_centrality_plots.png',
           dpi=300, bbox_inches='tight')

print("\n[OK] Bootstrap analysis complete")
print("[OK] Files saved:")
print("     - bootstrap_centrality.xlsx")
print("     - bootstrap_centrality_plots.png")
print("\n" + "=" * 100)
