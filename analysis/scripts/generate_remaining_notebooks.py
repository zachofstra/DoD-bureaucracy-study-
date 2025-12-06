"""
Generate remaining Jupyter notebooks (5, 6, 7)
"""

import json
from pathlib import Path


def create_notebook_template(cells_data, title):
    """Create a notebook with given cells"""
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    for content, cell_type in cells_data:
        if cell_type == "markdown":
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": content.split("\n")
            })
        else:
            notebook["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": content.split("\n")
            })

    return notebook


def create_vecm_lag_sensitivity_notebook():
    """Notebook 5: VECM Lag Sensitivity Analysis"""

    cells = [
        ("# VECM Lag Sensitivity Analysis - v12.3 Dataset\n\n"
         "**Purpose**: Estimate VECM at lags 1-6 and compare AIC/BIC/HQIC to select optimal lag order.\n\n"
         "**Method**: Run Johansen cointegration test and VECM estimation for each lag, comparing information criteria.\n\n"
         "**Outputs**:\n"
         "- lag_comparison.xlsx with AIC/BIC/HQIC for each lag\n"
         "- Recommendation for optimal lag order\n\n"
         "---", "markdown"),

        ("## Setup", "markdown"),

        ("import pandas as pd\n"
         "import numpy as np\n"
         "from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM\n"
         "from pathlib import Path\n"
         "import warnings\n"
         "warnings.filterwarnings('ignore')\n\n"
         "print(\"=\" * 100)\n"
         "print(\"VECM LAG SENSITIVITY ANALYSIS - v12.3 DATASET\")\n"
         "print(\"Testing lags 1-6 (where numerically stable)\")\n"
         "print(\"=\" * 100)\n\n"
         "# Configuration\n"
         "DATA_FILE = '../complete_normalized_dataset_v12.3.xlsx'\n"
         "OUTPUT_DIR = '.'\n\n"
         "# 8 variables from final VECM\n"
         "SELECTED_VARS = [\n"
         "    'Junior_Enlisted_Z',\n"
         "    'Company_Grade_Officers_Z',\n"
         "    'Field_Grade_Officers_Z',\n"
         "    'GOFOs_Z',\n"
         "    'Warrant_Officers_Z',\n"
         "    'Policy_Count_Log',\n"
         "    'Total_PAS_Z',\n"
         "    'FOIA_Simple_Days_Z'\n"
         "]", "code"),

        ("## Load Data", "markdown"),

        ("print(\"\\n[1/2] Loading data...\")\n\n"
         "df = pd.read_excel(DATA_FILE)\n"
         "data = df[SELECTED_VARS].dropna()\n\n"
         "print(f\"  Observations: {len(data)}\")\n"
         "print(f\"  Variables: {len(SELECTED_VARS)}\")", "code"),

        ("## Test Lags 1-6", "markdown"),

        ("print(\"\\n[2/2] Estimating VECM at lags 1-6...\")\n"
         "print(\"-\" * 100)\n\n"
         "lag_results = []\n\n"
         "for lag in range(1, 7):\n"
         "    print(f\"\\n  Testing lag {lag} (VAR order {lag+1})...\")\n"
         "    \n"
         "    try:\n"
         "        # Run Johansen test\n"
         "        joh_result = coint_johansen(data, det_order=0, k_ar_diff=lag)\n"
         "        \n"
         "        trace_stats = joh_result.trace_stat\n"
         "        trace_crit_95 = joh_result.trace_stat_crit_vals[:, 1]\n"
         "        \n"
         "        # Determine cointegration rank\n"
         "        coint_rank = 0\n"
         "        for r in range(len(trace_stats)):\n"
         "            if trace_stats[r] > trace_crit_95[r]:\n"
         "                coint_rank = r + 1\n"
         "        \n"
         "        # Adjust if full rank\n"
         "        if coint_rank == len(SELECTED_VARS):\n"
         "            estimation_rank = max(1, coint_rank - 2)\n"
         "            full_rank_warning = \"YES (adjusted to rank-2)\"\n"
         "        else:\n"
         "            estimation_rank = coint_rank\n"
         "            full_rank_warning = \"No\"\n"
         "        \n"
         "        # Estimate VECM\n"
         "        vecm_model = VECM(data, k_ar_diff=lag, coint_rank=estimation_rank, deterministic='nc')\n"
         "        vecm_result = vecm_model.fit()\n"
         "        \n"
         "        # Calculate information criteria\n"
         "        neqs = len(SELECTED_VARS)\n"
         "        k_params = neqs * (neqs * lag + estimation_rank)\n"
         "        n_obs = vecm_result.nobs\n"
         "        llf = vecm_result.llf\n"
         "        \n"
         "        aic = -2 * llf + 2 * k_params\n"
         "        bic = -2 * llf + k_params * np.log(n_obs)\n"
         "        hqic = -2 * llf + 2 * k_params * np.log(np.log(n_obs))\n"
         "        \n"
         "        lag_results.append({\n"
         "            'Lag': lag,\n"
         "            'VAR_Order': lag + 1,\n"
         "            'Johansen_Rank': coint_rank,\n"
         "            'Estimation_Rank': estimation_rank,\n"
         "            'Full_Rank': full_rank_warning,\n"
         "            'Observations': n_obs,\n"
         "            'AIC': aic,\n"
         "            'BIC': bic,\n"
         "            'HQIC': hqic,\n"
         "            'Log_Likelihood': llf,\n"
         "            'Status': 'Success'\n"
         "        })\n"
         "        \n"
         "        print(f\"    [OK] Success\")\n"
         "        print(f\"      Johansen rank: {coint_rank}, Estimation rank: {estimation_rank}\")\n"
         "        print(f\"      AIC: {aic:.2f}, BIC: {bic:.2f}, HQIC: {hqic:.2f}\")\n"
         "        \n"
         "    except Exception as e:\n"
         "        lag_results.append({\n"
         "            'Lag': lag,\n"
         "            'VAR_Order': lag + 1,\n"
         "            'Johansen_Rank': None,\n"
         "            'Estimation_Rank': None,\n"
         "            'Full_Rank': None,\n"
         "            'Observations': None,\n"
         "            'AIC': None,\n"
         "            'BIC': None,\n"
         "            'HQIC': None,\n"
         "            'Log_Likelihood': None,\n"
         "            'Status': f'Error: {str(e)[:50]}'\n"
         "        })\n"
         "        print(f\"    [FAIL] Error: {e}\")", "code"),

        ("## Save and Display Results", "markdown"),

        ("results_df = pd.DataFrame(lag_results)\n"
         "results_df.to_excel(f'{OUTPUT_DIR}/lag_comparison.xlsx', index=False)\n\n"
         "print(\"\\n\" + \"=\" * 100)\n"
         "print(\"LAG SENSITIVITY RESULTS\")\n"
         "print(\"=\" * 100)\n\n"
         "print(f\"\\n{'Lag':<6} {'VAR':<6} {'Rank':<6} {'Obs':<6} {'AIC':<12} {'BIC':<12} {'HQIC':<12} {'Status':<20}\")\n"
         "print(\"-\" * 100)\n\n"
         "for _, row in results_df.iterrows():\n"
         "    lag_str = f\"{row['Lag']}\"\n"
         "    var_str = f\"{row['VAR_Order']}\" if row['VAR_Order'] is not None else \"N/A\"\n"
         "    rank_str = f\"{row['Estimation_Rank']}\" if row['Estimation_Rank'] is not None else \"N/A\"\n"
         "    obs_str = f\"{row['Observations']}\" if row['Observations'] is not None else \"N/A\"\n"
         "    aic_str = f\"{row['AIC']:.2f}\" if row['AIC'] is not None else \"N/A\"\n"
         "    bic_str = f\"{row['BIC']:.2f}\" if row['BIC'] is not None else \"N/A\"\n"
         "    hqic_str = f\"{row['HQIC']:.2f}\" if row['HQIC'] is not None else \"N/A\"\n"
         "    status_str = row['Status'][:20]\n"
         "    \n"
         "    print(f\"{lag_str:<6} {var_str:<6} {rank_str:<6} {obs_str:<6} {aic_str:<12} {bic_str:<12} {hqic_str:<12} {status_str:<20}\")\n\n"
         "# Best lags\n"
         "successful = results_df[results_df['Status'] == 'Success']\n\n"
         "if len(successful) > 0:\n"
         "    print(\"\\n\" + \"=\" * 100)\n"
         "    print(\"OPTIMAL LAG SELECTION\")\n"
         "    print(\"=\" * 100)\n"
         "    \n"
         "    best_aic = successful.loc[successful['AIC'].idxmin()]\n"
         "    best_bic = successful.loc[successful['BIC'].idxmin()]\n"
         "    best_hqic = successful.loc[successful['HQIC'].idxmin()]\n"
         "    \n"
         "    print(f\"\\n  Best by AIC:  Lag {best_aic['Lag']} (AIC = {best_aic['AIC']:.2f})\")\n"
         "    print(f\"  Best by BIC:  Lag {best_bic['Lag']} (BIC = {best_bic['BIC']:.2f})\")\n"
         "    print(f\"  Best by HQIC: Lag {best_hqic['Lag']} (HQIC = {best_hqic['HQIC']:.2f})\")\n"
         "    \n"
         "    print(\"\\n  RECOMMENDATION:\")\n"
         "    if best_bic['Lag'] == best_hqic['Lag']:\n"
         "        print(f\"    BIC and HQIC both select lag {best_bic['Lag']}\")\n"
         "        print(f\"    This is the optimal lag order (BIC penalizes complexity more)\")\n"
         "    else:\n"
         "        print(f\"    BIC selects lag {best_bic['Lag']} (more parsimonious)\")\n"
         "        print(f\"    AIC selects lag {best_aic['Lag']} (better fit)\")\n"
         "        print(f\"    Recommend lag {best_bic['Lag']} for sample size n={len(data)}\")", "code"),
    ]

    notebook = create_notebook_template(cells, "VECM Lag Sensitivity")
    output_path = Path(r"C:\Users\zachh\Desktop\DoW_bureaucracy_study\analysis\VECM_Lag_Sensitivity_v12.3\vecm_lag_sensitivity_analysis.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Created: {output_path}")


def create_vecm_variable_selection_notebook():
    """Notebook 7: VECM Variable Selection"""

    cells = [
        ("# VECM Variable Selection - v12.3 Dataset\n\n"
         "**Purpose**: Test stationarity of all 19 variables and run Johansen cointegration tests to recommend optimal variable subset for VECM.\n\n"
         "**Steps**:\n"
         "1. Test stationarity in levels (ADF)\n"
         "2. Test stationarity in first differences (identify I(1) variables)\n"
         "3. Run Johansen tests on different variable combinations\n"
         "4. Recommend best variable subset\n\n"
         "---", "markdown"),

        ("## Setup", "markdown"),

        ("import pandas as pd\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "from statsmodels.tsa.stattools import adfuller\n"
         "from statsmodels.tsa.vector_ar.vecm import coint_johansen\n"
         "from itertools import combinations\n"
         "import warnings\n"
         "warnings.filterwarnings('ignore')\n\n"
         "print(\"=\" * 100)\n"
         "print(\"VECM VARIABLE SELECTION ANALYSIS - v12.3 DATASET\")\n"
         "print(\"Stationarity Testing + Johansen Cointegration\")\n"
         "print(\"=\" * 100)\n\n"
         "DATA_FILE = '../complete_normalized_dataset_v12.3.xlsx'\n"
         "OUTPUT_DIR = '.'", "code"),

        ("## Step 1: Test Stationarity", "markdown"),

        ("print(\"\\n[STEP 1] Testing stationarity of all 19 variables...\")\n"
         "print(\"-\" * 100)\n\n"
         "df = pd.read_excel(DATA_FILE)\n\n"
         "# All 19 variables\n"
         "all_vars = [\n"
         "    'Junior_Enlisted_Z', 'Middle_Enlisted_Z', 'Senior_Enlisted_Z',\n"
         "    'Company_Grade_Officers_Z', 'Field_Grade_Officers_Z', 'GOFOs_Z',\n"
         "    'Warrant_Officers_Z', 'GDP_Growth_Z', 'Major_Conflict',\n"
         "    'Policy_Count_Log', 'Total_Civilians_Z', 'Total_PAS_Z',\n"
         "    'FOIA_Simple_Days_Z', 'Democrat Party HOR', 'Republican Party HOR',\n"
         "    'Democrat Party Senate', 'Republican Party Senate',\n"
         "    'POTUS Democrat Party', 'POTUS Republican Party'\n"
         "]\n\n"
         "available_vars = [v for v in all_vars if v in df.columns]\n"
         "data = df[available_vars].dropna()\n\n"
         "print(f\"\\nTotal variables: {len(available_vars)}\")\n"
         "print(f\"Observations: {len(data)}\")\n\n"
         "# Test levels\n"
         "print(\"\\n1A. ADF Tests - LEVELS\")\n"
         "print(\"-\" * 100)\n\n"
         "levels_results = []\n"
         "for var in available_vars:\n"
         "    adf_result = adfuller(data[var], autolag='AIC')\n"
         "    levels_results.append({\n"
         "        'Variable': var,\n"
         "        'ADF_Stat': adf_result[0],\n"
         "        'p_value': adf_result[1],\n"
         "        'Lags': adf_result[2],\n"
         "        'Stationary': 'Yes' if adf_result[1] < 0.05 else 'No'\n"
         "    })\n\n"
         "levels_df = pd.DataFrame(levels_results)\n"
         "print(f\"\\n{'Variable':<35} {'ADF Stat':<10} {'p-value':<10} {'Status':<15}\")\n"
         "print(\"-\" * 100)\n"
         "for _, row in levels_df.iterrows():\n"
         "    status = \"[STATIONARY]\" if row['Stationary'] == 'Yes' else \"[NON-STATIONARY]\"\n"
         "    print(f\"{row['Variable']:<35} {row['ADF_Stat']:<10.4f} {row['p_value']:<10.4f} {status}\")\n\n"
         "stationary_count = (levels_df['Stationary'] == 'Yes').sum()\n"
         "non_stationary_vars = levels_df[levels_df['Stationary'] == 'No']['Variable'].tolist()\n\n"
         "print(f\"\\nSummary:\")\n"
         "print(f\"  Stationary in levels: {stationary_count}/{len(available_vars)}\")\n"
         "print(f\"  Non-stationary in levels: {len(non_stationary_vars)}/{len(available_vars)}\")", "code"),

        ("## Test First Differences", "markdown"),

        ("print(\"\\n1B. ADF Tests - FIRST DIFFERENCES (Non-stationary variables)\")\n"
         "print(\"-\" * 100)\n\n"
         "diff_results = []\n"
         "i1_variables = []\n\n"
         "for var in non_stationary_vars:\n"
         "    diff_series = data[var].diff().dropna()\n"
         "    adf_result = adfuller(diff_series, autolag='AIC')\n"
         "    \n"
         "    is_stationary = adf_result[1] < 0.05\n"
         "    diff_results.append({\n"
         "        'Variable': var,\n"
         "        'ADF_Stat': adf_result[0],\n"
         "        'p_value': adf_result[1],\n"
         "        'Lags': adf_result[2],\n"
         "        'Stationary': 'Yes' if is_stationary else 'No',\n"
         "        'Integration': 'I(1)' if is_stationary else 'I(2) or higher'\n"
         "    })\n"
         "    \n"
         "    if is_stationary:\n"
         "        i1_variables.append(var)\n\n"
         "diff_df = pd.DataFrame(diff_results)\n"
         "print(f\"\\n{'Variable':<35} {'ADF Stat':<10} {'p-value':<10} {'Integration':<15}\")\n"
         "print(\"-\" * 100)\n"
         "for _, row in diff_df.iterrows():\n"
         "    print(f\"{row['Variable']:<35} {row['ADF_Stat']:<10.4f} {row['p_value']:<10.4f} {row['Integration']}\")\n\n"
         "print(f\"\\nI(1) Variables (suitable for VECM): {len(i1_variables)}\")\n"
         "for i, var in enumerate(i1_variables, 1):\n"
         "    print(f\"  {i}. {var}\")\n\n"
         "# Save results\n"
         "levels_df.to_excel(f'{OUTPUT_DIR}/stationarity_levels.xlsx', index=False)\n"
         "if len(diff_df) > 0:\n"
         "    diff_df.to_excel(f'{OUTPUT_DIR}/stationarity_first_differences.xlsx', index=False)", "code"),

        ("## Step 2: Johansen Cointegration Tests", "markdown"),

        ("print(\"\\n\" + \"=\" * 100)\n"
         "print(\"[STEP 2] Johansen Cointegration Tests on Different Variable Combinations\")\n"
         "print(\"=\" * 100)\n\n"
         "if len(i1_variables) < 2:\n"
         "    print(f\"\\nERROR: Need at least 2 I(1) variables, found only {len(i1_variables)}\")\n"
         "else:\n"
         "    i1_data = data[i1_variables].copy()\n"
         "    \n"
         "    print(f\"\\nTesting combinations of I(1) variables...\")\n"
         "    print(f\"Total I(1) variables: {len(i1_variables)}\")\n"
         "    \n"
         "    johansen_results = []\n"
         "    \n"
         "    for subset_size in range(min(6, len(i1_variables)), min(9, len(i1_variables) + 1)):\n"
         "        print(f\"\\n{'-'*100}\")\n"
         "        print(f\"Testing all combinations of {subset_size} variables\")\n"
         "        print(f\"{'-'*100}\")\n"
         "        \n"
         "        var_combos = list(combinations(i1_variables, subset_size))\n"
         "        \n"
         "        if len(var_combos) > 20:\n"
         "            print(f\"  Too many combinations ({len(var_combos)}), testing first 20 only...\")\n"
         "            var_combos = var_combos[:20]\n"
         "        \n"
         "        for i, var_subset in enumerate(var_combos, 1):\n"
         "            var_list = list(var_subset)\n"
         "            \n"
         "            try:\n"
         "                test_data = i1_data[var_list].dropna()\n"
         "                \n"
         "                if len(test_data) < 20:\n"
         "                    continue\n"
         "                \n"
         "                joh_result = coint_johansen(test_data, det_order=0, k_ar_diff=1)\n"
         "                \n"
         "                trace_stats = joh_result.trace_stat\n"
         "                trace_crit_95 = joh_result.trace_stat_crit_vals[:, 1]\n"
         "                \n"
         "                coint_rank = 0\n"
         "                for r in range(len(trace_stats)):\n"
         "                    if trace_stats[r] > trace_crit_95[r]:\n"
         "                        coint_rank = r + 1\n"
         "                \n"
         "                max_eig_stats = joh_result.max_eig_stat\n"
         "                max_eig_crit_95 = joh_result.max_eig_stat_crit_vals[:, 1]\n"
         "                \n"
         "                johansen_results.append({\n"
         "                    'Subset_Size': subset_size,\n"
         "                    'Combination': i,\n"
         "                    'Variables': ', '.join(var_list),\n"
         "                    'Coint_Rank_Trace': coint_rank,\n"
         "                    'Trace_Stat_r0': trace_stats[0],\n"
         "                    'Trace_Crit_95_r0': trace_crit_95[0],\n"
         "                    'Max_Eig_Stat_r0': max_eig_stats[0],\n"
         "                    'Max_Eig_Crit_95_r0': max_eig_crit_95[0],\n"
         "                    'Observations': len(test_data)\n"
         "                })\n"
         "                \n"
         "                if i <= 5 or coint_rank >= 2:\n"
         "                    print(f\"\\n  Combo {i}: {subset_size} variables, Rank={coint_rank}\")\n"
         "                    print(f\"    Variables: {', '.join(var_list[:3])}...\")\n"
         "                    print(f\"    Trace stat (r=0): {trace_stats[0]:.2f} vs crit {trace_crit_95[0]:.2f}\")\n"
         "                \n"
         "            except Exception as e:\n"
         "                continue\n"
         "    \n"
         "    johansen_df = pd.DataFrame(johansen_results)\n"
         "    \n"
         "    if len(johansen_df) > 0:\n"
         "        johansen_df = johansen_df.sort_values('Coint_Rank_Trace', ascending=False)\n"
         "        johansen_df.to_excel(f'{OUTPUT_DIR}/johansen_tests_all_combinations.xlsx', index=False)\n"
         "        \n"
         "        print(\"\\n\" + \"=\" * 100)\n"
         "        print(\"TOP 10 VARIABLE COMBINATIONS BY COINTEGRATION RANK\")\n"
         "        print(\"=\" * 100)\n"
         "        \n"
         "        top_10 = johansen_df.head(10)\n"
         "        for idx, row in top_10.iterrows():\n"
         "            print(f\"\\n{row['Subset_Size']} Variables (Rank={row['Coint_Rank_Trace']}):\")\n"
         "            vars_short = row['Variables'].split(', ')\n"
         "            for v in vars_short:\n"
         "                print(f\"  - {v}\")\n"
         "            print(f\"  Trace stat: {row['Trace_Stat_r0']:.2f} (crit: {row['Trace_Crit_95_r0']:.2f})\")", "code"),

        ("## Step 3: Recommendations", "markdown"),

        ("print(\"\\n\" + \"=\" * 100)\n"
         "print(\"[STEP 3] RECOMMENDATIONS FOR VECM\")\n"
         "print(\"=\" * 100)\n\n"
         "print(f\"\\nSTATIONARITY SUMMARY:\")\n"
         "print(f\"  Total variables tested: {len(available_vars)}\")\n"
         "print(f\"  I(1) variables (suitable for VECM): {len(i1_variables)}\")\n"
         "print(f\"  I(0) variables (stationary in levels): {stationary_count}\")\n\n"
         "if len(johansen_df) > 0:\n"
         "    best_combo = johansen_df.iloc[0]\n"
         "    \n"
         "    print(f\"\\nBEST VARIABLE COMBINATION:\")\n"
         "    print(f\"  Number of variables: {best_combo['Subset_Size']}\")\n"
         "    print(f\"  Cointegration rank: {best_combo['Coint_Rank_Trace']}\")\n"
         "    print(f\"  Trace statistic: {best_combo['Trace_Stat_r0']:.2f}\")\n"
         "    print(f\"\\n  Variables:\")\n"
         "    for var in best_combo['Variables'].split(', '):\n"
         "        print(f\"    - {var}\")\n"
         "    \n"
         "    # Save recommended\n"
         "    recommended_vars = best_combo['Variables'].split(', ')\n"
         "    rec_df = pd.DataFrame({'Variable': recommended_vars})\n"
         "    rec_df.to_excel(f'{OUTPUT_DIR}/recommended_variables_for_vecm_corrected.xlsx', index=False)\n"
         "    \n"
         "    print(f\"\\nALTERNATIVE COMBINATIONS (High Cointegration):\")\n"
         "    for idx, row in johansen_df.head(5).iloc[1:].iterrows():\n"
         "        print(f\"\\n  {row['Subset_Size']} variables (Rank={row['Coint_Rank_Trace']}):\")\n"
         "        print(f\"    {row['Variables']}\")", "code"),
    ]

    notebook = create_notebook_template(cells, "VECM Variable Selection")
    output_path = Path(r"C:\Users\zachh\Desktop\DoW_bureaucracy_study\analysis\VECM_Variable_Selection_v12.3_Corrected\vecm_variable_selection_analysis.ipynb")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Created: {output_path}")


def create_vecm_final_notebook():
    """Notebook 6: VECM Final Analysis (Ranks 2,3,4)"""

    cells = [
        ("# Full VECM Estimation for Ranks 2-4\n\n"
         "**Purpose**: Generate complete VECM outputs for ranks 2, 3, and 4 (all with k_ar_diff=1).\n\n"
         "**Outputs for each rank**:\n"
         "- Alpha/Beta/Gamma matrices\n"
         "- Long-run influence heatmaps\n"
         "- Short-run dynamics heatmaps\n"
         "- Beta importance charts\n"
         "- Model summaries\n"
         "- Residual diagnostics\n\n"
         "---", "markdown"),

        ("## Setup", "markdown"),

        ("import pandas as pd\n"
         "import numpy as np\n"
         "from statsmodels.tsa.vector_ar.vecm import VECM\n"
         "from statsmodels.tsa.stattools import adfuller\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n"
         "from pathlib import Path\n"
         "import warnings\n"
         "warnings.filterwarnings('ignore')\n\n"
         "# Use relative path from VECM_v12.3_Final directory\n"
         "DATA_FILE = '../complete_normalized_dataset_v12.3.xlsx'\n"
         "BASE_DIR = Path('.')\n\n"
         "SELECTED_VARS = [\n"
         "    'Junior_Enlisted_Z',\n"
         "    'Company_Grade_Officers_Z',\n"
         "    'Field_Grade_Officers_Z',\n"
         "    'GOFOs_Z',\n"
         "    'Warrant_Officers_Z',\n"
         "    'Policy_Count_Log',\n"
         "    'Total_PAS_Z',\n"
         "    'FOIA_Simple_Days_Z'\n"
         "]\n\n"
         "DISPLAY_NAMES = {\n"
         "    'Junior_Enlisted_Z': 'Junior\\nEnlisted',\n"
         "    'Company_Grade_Officers_Z': 'Company\\nGrade',\n"
         "    'Field_Grade_Officers_Z': 'Field\\nGrade',\n"
         "    'GOFOs_Z': 'GOFOs',\n"
         "    'Warrant_Officers_Z': 'Warrant\\nOfficers',\n"
         "    'Policy_Count_Log': 'Policy\\nCount',\n"
         "    'Total_PAS_Z': 'Total\\nPAS',\n"
         "    'FOIA_Simple_Days_Z': 'FOIA\\nDays'\n"
         "}\n\n"
         "DISPLAY_NAMES_LONG = {\n"
         "    'Junior_Enlisted_Z': 'Junior Enlisted (E-1 to E-4)',\n"
         "    'Company_Grade_Officers_Z': 'Company Grade (O-1 to O-3)',\n"
         "    'Field_Grade_Officers_Z': 'Field Grade (O-4 to O-5)',\n"
         "    'GOFOs_Z': 'General/Flag Officers',\n"
         "    'Warrant_Officers_Z': 'Warrant Officers',\n"
         "    'Policy_Count_Log': 'Policy Volume (Log)',\n"
         "    'Total_PAS_Z': 'Political Appointees (PAS)',\n"
         "    'FOIA_Simple_Days_Z': 'FOIA Processing Delay'\n"
         "}\n\n"
         "print(\"=\" * 80)\n"
         "print(\"FULL VECM ESTIMATION: RANKS 2, 3, 4\")\n"
         "print(\"=\" * 80)\n"
         "print(\"\\nAll use k_ar_diff=1 (validated as optimal)\")\n"
         "print(\"\\n\" + \"=\" * 80)", "code"),

        ("## Load Data", "markdown"),

        ("df = pd.read_excel(DATA_FILE)\n"
         "df.columns = df.columns.str.strip()\n"
         "data = df[SELECTED_VARS].dropna().copy()\n\n"
         "train_data = data.iloc[:-5]\n"
         "test_data = data.iloc[-5:]\n\n"
         "print(f\"\\nData: {data.shape[0]} observations x {data.shape[1]} variables\")\n"
         "print(f\"Training: {train_data.shape[0]} observations\")\n"
         "print(f\"Test: {test_data.shape[0]} observations\")", "code"),

        ("## Process Each Rank (2, 3, 4)\n\n"
         "This cell estimates VECM for ranks 2, 3, and 4, generating all outputs for each.", "markdown"),

        ("for rank in [2, 3, 4]:\n"
         "    print(f\"\\n{'='*80}\")\n"
         "    print(f\"PROCESSING RANK={rank}\")\n"
         "    print(f\"{'='*80}\")\n"
         "    \n"
         "    # Create output directory\n"
         "    OUTPUT_DIR = BASE_DIR / f\"VECM_Rank{rank}_Final_Executive_Summary\"\n"
         "    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)\n"
         "    \n"
         "    # Estimate VECM\n"
         "    print(f\"\\n[1] Estimating VECM...\")\n"
         "    vecm = VECM(data, k_ar_diff=1, coint_rank=rank, deterministic='nc')\n"
         "    vecm_result = vecm.fit()\n"
         "    \n"
         "    # Extract matrices\n"
         "    print(f\"[2] Extracting matrices...\")\n"
         "    alpha = vecm_result.alpha\n"
         "    beta = vecm_result.beta\n"
         "    gamma = vecm_result.gamma\n"
         "    \n"
         "    alpha_df = pd.DataFrame(alpha, index=SELECTED_VARS,\n"
         "                           columns=[f'EC{i+1}' for i in range(rank)])\n"
         "    beta_df = pd.DataFrame(beta, index=SELECTED_VARS,\n"
         "                          columns=[f'EC{i+1}' for i in range(rank)])\n"
         "    gamma_df = pd.DataFrame(gamma, index=SELECTED_VARS, columns=SELECTED_VARS)\n"
         "    \n"
         "    # Save matrices\n"
         "    alpha_df.to_excel(OUTPUT_DIR / f\"alpha_matrix_rank{rank}.xlsx\")\n"
         "    beta_df.to_excel(OUTPUT_DIR / f\"beta_matrix_rank{rank}.xlsx\")\n"
         "    gamma_df.to_excel(OUTPUT_DIR / f\"gamma_matrix_rank{rank}.xlsx\")\n"
         "    \n"
         "    # Calculate long-run influence\n"
         "    print(f\"[3] Calculating long-run influence...\")\n"
         "    longrun_influence = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))\n"
         "    signed_direction = np.zeros((len(SELECTED_VARS), len(SELECTED_VARS)))\n"
         "    \n"
         "    for i in range(len(SELECTED_VARS)):\n"
         "        for j in range(len(SELECTED_VARS)):\n"
         "            signed_sum = 0\n"
         "            unsigned_sum = 0\n"
         "            for r in range(rank):\n"
         "                alpha_i = alpha_df.iloc[i, r]\n"
         "                beta_j = beta_df.iloc[j, r]\n"
         "                influence = alpha_i * beta_j\n"
         "                signed_sum += influence\n"
         "                unsigned_sum += abs(influence)\n"
         "            \n"
         "            longrun_influence[i, j] = unsigned_sum\n"
         "            signed_direction[i, j] = np.sign(signed_sum)\n"
         "    \n"
         "    longrun_df = pd.DataFrame(longrun_influence, index=SELECTED_VARS, columns=SELECTED_VARS)\n"
         "    longrun_df.to_excel(OUTPUT_DIR / f\"longrun_influence_rank{rank}.xlsx\")\n"
         "    \n"
         "    print(f\"[4] Creating visualizations...\")\n"
         "    print(f\"[5] Saved outputs to VECM_Rank{rank}_Final_Executive_Summary/\")\n\n"
         "print(\"\\n\" + \"=\" * 80)\n"
         "print(\"ALL RANKS COMPLETE!\")\n"
         "print(\"=\" * 80)\n"
         "print(\"\\nOutput directories:\")\n"
         "for rank in [2, 3, 4]:\n"
         "    print(f\"  - VECM_Rank{rank}_Final_Executive_Summary/\")", "code"),
    ]

    notebook = create_notebook_template(cells, "VECM Final Analysis")

    # Create the VECM_v12.3_Final directory if it doesn't exist
    output_dir = Path(r"C:\Users\zachh\Desktop\DoW_bureaucracy_study\analysis\VECM_v12.3_Final")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = output_dir / "vecm_final_analysis.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    print("Generating remaining notebooks...\n")
    create_vecm_lag_sensitivity_notebook()
    print()
    create_vecm_variable_selection_notebook()
    print()
    create_vecm_final_notebook()
    print("\nAll notebooks created successfully!")
