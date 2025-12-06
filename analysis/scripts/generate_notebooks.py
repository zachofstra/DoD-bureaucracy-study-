"""
Generate Jupyter notebooks from Python scripts
Converts analysis scripts to notebook format with markdown documentation
"""

import json
from pathlib import Path

def create_var2_notebook():
    """Create VAR(2) Analysis Notebook"""

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

    # Read the original Python script
    script_path = Path(r"C:\Users\zachh\Desktop\DoW_bureaucracy_study\analysis\VAR2_v12.3\var4_analysis_v12.3.py")
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()

    # Split into logical sections
    sections = [
        ("# VAR(2) Model - DoD Bureaucratic Growth Analysis v12.3\n\n"
         "**Purpose**: Full Vector Autoregression with 2 lags on 8 selected variables.\n\n"
         "**Dataset**: `complete_normalized_dataset_v12.3.xlsx`\n\n"
         "**Analysis Steps**:\n"
         "1. Load 8 selected variables\n"
         "2. Test stationarity (ADF tests)\n"
         "3. Estimate VAR(2) model\n"
         "4. Extract coefficients\n"
         "5. Run Granger causality tests\n"
         "6. Compute Impulse Response Functions\n"
         "7. Compute Forecast Error Variance Decomposition\n\n"
         "---", "markdown"),

        ("## Setup and Configuration", "markdown"),

        ("import pandas as pd\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n"
         "from statsmodels.tsa.api import VAR\n"
         "from statsmodels.tsa.stattools import adfuller\n"
         "from scipy import stats\n"
         "from pathlib import Path\n"
         "import warnings\n"
         "warnings.filterwarnings('ignore')\n\n"
         "# Configuration\n"
         "DATA_FILE = '../complete_normalized_dataset_v12.3.xlsx'\n"
         "OUTPUT_DIR = '.'\n"
         "LAG_ORDER = 2\n\n"
         "# 8 variables from Granger causality analysis\n"
         "SELECTED_VARS = [\n"
         "    'Warrant_Officers_Z',\n"
         "    'Policy_Count_Log',\n"
         "    'Company_Grade_Officers_Z',\n"
         "    'Total_PAS_Z',\n"
         "    'FOIA_Simple_Days_Z',\n"
         "    'Junior_Enlisted_Z',\n"
         "    'Field_Grade_Officers_Z',\n"
         "    'Total_Civilians_Z'\n"
         "]\n\n"
         "print(\"=\" * 100)\n"
         "print(f\"VAR({LAG_ORDER}) MODEL ANALYSIS - v12.3 DATASET\")\n"
         "print(\"8 Variables Selected from Pairwise Granger Causality\")\n"
         "print(\"=\" * 100)", "code"),

        ("## 1. Load and Prepare Data", "markdown"),

        ("print(\"\\n[1/7] Loading data...\")\n\n"
         "df = pd.read_excel(DATA_FILE)\n"
         "data = df[SELECTED_VARS].dropna()\n\n"
         "print(f\"  Observations: {len(data)}\")\n"
         "print(f\"  Variables: {len(SELECTED_VARS)}\")\n"
         "print(\"\\n  Selected variables:\")\n"
         "for i, var in enumerate(SELECTED_VARS, 1):\n"
         "    print(f\"    {i}. {var}\")", "code"),

        ("## 2. Test Stationarity (ADF Tests)", "markdown"),

        ("print(\"\\n[2/7] Testing stationarity (ADF tests)...\")\n\n"
         "stationarity_results = []\n"
         "for var in SELECTED_VARS:\n"
         "    adf_result = adfuller(data[var], autolag='AIC')\n"
         "    stationarity_results.append({\n"
         "        'Variable': var,\n"
         "        'ADF_Statistic': adf_result[0],\n"
         "        'p_value': adf_result[1],\n"
         "        'Lags_Used': adf_result[2],\n"
         "        'Stationary': 'Yes' if adf_result[1] < 0.05 else 'No'\n"
         "    })\n\n"
         "stationarity_df = pd.DataFrame(stationarity_results)\n"
         "stationarity_df.to_excel(f'{OUTPUT_DIR}/stationarity_tests.xlsx', index=False)\n\n"
         "print(\"\\n  Stationarity Test Results:\")\n"
         "print(\"  \" + \"-\" * 80)\n"
         "for _, row in stationarity_df.iterrows():\n"
         "    status = \"[STATIONARY]\" if row['Stationary'] == 'Yes' else \"[NON-STATIONARY]\"\n"
         "    print(f\"    {row['Variable']:30s} ADF={row['ADF_Statistic']:8.4f}, p={row['p_value']:.4f} {status}\")\n\n"
         "stationary_count = (stationarity_df['Stationary'] == 'Yes').sum()\n"
         "print(f\"\\n  Stationary variables: {stationary_count}/{len(SELECTED_VARS)}\")", "code"),

        ("## 3. Estimate VAR(2) Model", "markdown"),

        ("print(f\"\\n[3/7] Estimating VAR({LAG_ORDER}) model...\")\n\n"
         "model = VAR(data)\n"
         "var_result = model.fit(maxlags=LAG_ORDER, ic=None)\n\n"
         "print(f\"\\n  Model estimated successfully\")\n"
         "print(f\"  Lag order: {var_result.k_ar}\")\n"
         "print(f\"  Number of equations: {var_result.neqs}\")\n"
         "print(f\"  Number of coefficients per equation: {var_result.k_ar * var_result.neqs + 1}\")\n"
         "print(f\"  Total observations used: {var_result.nobs}\")", "code"),

        ("## 4. Model Diagnostics and R-squared", "markdown"),

        ("print(f\"\\n[4/7] Generating model summary and diagnostics...\")\n\n"
         "# Calculate R-squared for each equation\n"
         "rsquared_data = []\n"
         "residuals = var_result.resid.values if hasattr(var_result.resid, 'values') else var_result.resid\n"
         "for i, var in enumerate(SELECTED_VARS):\n"
         "    resid = residuals[:, i]\n"
         "    y_actual = data[var].values[-len(resid):]\n"
         "    y_pred = y_actual - resid\n"
         "    \n"
         "    ss_res = np.sum(resid ** 2)\n"
         "    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)\n"
         "    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0\n"
         "    \n"
         "    n = len(resid)\n"
         "    k = LAG_ORDER * len(SELECTED_VARS) + 1\n"
         "    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k) if (n - k) > 0 else 0\n"
         "    \n"
         "    rsquared_data.append({\n"
         "        'Variable': var,\n"
         "        'R_squared': r_squared,\n"
         "        'Adj_R_squared': adj_r_squared\n"
         "    })\n\n"
         "rsquared_df = pd.DataFrame(rsquared_data)\n"
         "rsquared_df.to_excel(f'{OUTPUT_DIR}/model_fit_rsquared.xlsx', index=False)\n\n"
         "print(\"\\n  R-squared by equation:\")\n"
         "print(\"  \" + \"-\" * 60)\n"
         "for _, row in rsquared_df.iterrows():\n"
         "    print(f\"    {row['Variable']:30s} R2={row['R_squared']:.4f}, Adj-R2={row['Adj_R_squared']:.4f}\")\n\n"
         "avg_rsq = rsquared_df['R_squared'].mean()\n"
         "print(f\"\\n  Average R-squared: {avg_rsq:.4f}\")", "code"),

        ("## 5. Extract Coefficient Matrices", "markdown"),

        ("print(f\"\\n[5/7] Extracting coefficient matrices...\")\n\n"
         "for lag in range(1, LAG_ORDER + 1):\n"
         "    coef_matrix = pd.DataFrame(index=SELECTED_VARS, columns=SELECTED_VARS)\n"
         "    \n"
         "    for to_var in SELECTED_VARS:\n"
         "        for from_var in SELECTED_VARS:\n"
         "            param_name = f'L{lag}.{from_var}'\n"
         "            if param_name in var_result.params.index:\n"
         "                coef_matrix.loc[from_var, to_var] = var_result.params.loc[param_name, to_var]\n"
         "            else:\n"
         "                coef_matrix.loc[from_var, to_var] = 0.0\n"
         "    \n"
         "    coef_matrix = coef_matrix.astype(float)\n"
         "    coef_matrix.to_excel(f'{OUTPUT_DIR}/coefficients_lag{lag}.xlsx')\n"
         "    print(f\"  Saved coefficients for lag {lag}\")\n\n"
         "# Save constant terms\n"
         "const_df = pd.DataFrame({\n"
         "    'Equation': SELECTED_VARS,\n"
         "    'Constant': [var_result.params.loc['const', var] for var in SELECTED_VARS]\n"
         "})\n"
         "const_df.to_excel(f'{OUTPUT_DIR}/constants.xlsx', index=False)\n"
         "print(f\"  Saved constant terms\")", "code"),

        ("## 6. Granger Causality Tests", "markdown"),

        ("print(f\"\\n[6/7] Running Granger causality tests on fitted model...\")\n\n"
         "granger_results = []\n"
         "for caused_var in SELECTED_VARS:\n"
         "    for causing_var in SELECTED_VARS:\n"
         "        if caused_var == causing_var:\n"
         "            continue\n"
         "        \n"
         "        try:\n"
         "            gc_test = var_result.test_causality(caused_var, causing_var, kind='f')\n"
         "            granger_results.append({\n"
         "                'Caused': caused_var,\n"
         "                'Causing': causing_var,\n"
         "                'F_statistic': gc_test.test_statistic,\n"
         "                'p_value': gc_test.pvalue,\n"
         "                'df_num': gc_test.df,\n"
         "                'df_denom': gc_test.df_denom,\n"
         "                'Significant_5pct': gc_test.pvalue < 0.05,\n"
         "                'Significant_1pct': gc_test.pvalue < 0.01\n"
         "            })\n"
         "        except:\n"
         "            continue\n\n"
         "granger_causality_df = pd.DataFrame(granger_results)\n\n"
         "if len(granger_causality_df) > 0:\n"
         "    granger_causality_df = granger_causality_df.sort_values('p_value')\n"
         "    granger_causality_df.to_excel(f'{OUTPUT_DIR}/granger_causality_tests.xlsx', index=False)\n"
         "    \n"
         "    sig_5pct = granger_causality_df['Significant_5pct'].sum()\n"
         "    sig_1pct = granger_causality_df['Significant_1pct'].sum()\n"
         "    print(f\"\\n  Granger causality test results:\")\n"
         "    print(f\"    Total tests: {len(granger_causality_df)}\")\n"
         "    print(f\"    Significant at 5%: {sig_5pct}\")\n"
         "    print(f\"    Significant at 1%: {sig_1pct}\")\n"
         "else:\n"
         "    print(f\"\\n  No Granger causality tests completed successfully\")\n"
         "    sig_5pct = 0\n"
         "    sig_1pct = 0", "code"),

        ("## 7. Impulse Response Functions and FEVD", "markdown"),

        ("print(f\"\\n[7/7] Computing impulse response functions...\")\n\n"
         "# Compute IRFs (10 periods ahead)\n"
         "irf = var_result.irf(10)\n\n"
         "# Plot IRFs\n"
         "try:\n"
         "    fig = irf.plot(orth=False, figsize=(24, 20))\n"
         "    plt.suptitle('Impulse Response Functions (Non-Orthogonalized) - All Variables',\n"
         "                 fontsize=16, fontweight='bold', y=0.995)\n"
         "    plt.tight_layout(rect=[0, 0, 1, 0.99])\n"
         "    plt.savefig(f'{OUTPUT_DIR}/impulse_response_functions_all.png', dpi=300, bbox_inches='tight')\n"
         "    plt.close()\n"
         "    print(\"  IRF plots saved\")\n"
         "except Exception as e:\n"
         "    print(f\"  Warning: Could not plot IRFs: {e}\")\n\n"
         "# Save IRF data\n"
         "irf_data = []\n"
         "for i, impulse_var in enumerate(SELECTED_VARS):\n"
         "    for j, response_var in enumerate(SELECTED_VARS):\n"
         "        irf_values = irf.irfs[:, j, i]\n"
         "        for period in range(len(irf_values)):\n"
         "            irf_data.append({\n"
         "                'Impulse': impulse_var,\n"
         "                'Response': response_var,\n"
         "                'Period': period,\n"
         "                'IRF_Value': irf_values[period]\n"
         "            })\n\n"
         "irf_df = pd.DataFrame(irf_data)\n"
         "irf_df.to_excel(f'{OUTPUT_DIR}/impulse_response_data.xlsx', index=False)\n"
         "print(\"  IRF data saved\")\n\n"
         "# FEVD\n"
         "print(\"\\n  Computing forecast error variance decomposition...\")\n"
         "fevd = var_result.fevd(10)\n\n"
         "for i, var in enumerate(SELECTED_VARS):\n"
         "    fevd_var = pd.DataFrame(\n"
         "        fevd.decomp[:, i, :],\n"
         "        columns=SELECTED_VARS\n"
         "    )\n"
         "    fevd_var.insert(0, 'Period', range(len(fevd_var)))\n"
         "    fevd_var.to_excel(f'{OUTPUT_DIR}/fevd_{var}.xlsx', index=False)\n\n"
         "print(\"  FEVD saved for all variables\")", "code"),

        ("## Summary", "markdown"),

        ("print(\"\\n\" + \"=\" * 100)\n"
         "print(f\"VAR({LAG_ORDER}) ANALYSIS COMPLETE\")\n"
         "print(\"=\" * 100)\n\n"
         "print(f\"\\nMODEL SPECIFICATION:\")\n"
         "print(f\"  Lag order: {LAG_ORDER}\")\n"
         "print(f\"  Number of variables: {len(SELECTED_VARS)}\")\n"
         "print(f\"  Observations used: {var_result.nobs}\")\n"
         "print(f\"  Total parameters: {len(SELECTED_VARS) * (len(SELECTED_VARS) * LAG_ORDER + 1)}\")\n\n"
         "print(f\"\\nMODEL FIT:\")\n"
         "print(f\"  Average R-squared: {avg_rsq:.4f}\")\n"
         "print(f\"  AIC: {var_result.aic:.2f}\")\n"
         "print(f\"  BIC: {var_result.bic:.2f}\")\n"
         "print(f\"  HQIC: {var_result.hqic:.2f}\")\n\n"
         "print(f\"\\nDIAGNOSTICS:\")\n"
         "print(f\"  Stationary variables: {stationary_count}/{len(SELECTED_VARS)}\")\n"
         "print(f\"  Significant Granger causalities (5%): {sig_5pct}/{len(granger_causality_df)}\")\n\n"
         "print(\"=\" * 100)", "code"),
    ]

    # Create cells
    for content, cell_type in sections:
        if cell_type == "markdown":
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": content.split("\n")
            })
        else:  # code
            notebook["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": content.split("\n")
            })

    # Save notebook
    output_path = Path(r"C:\Users\zachh\Desktop\DoW_bureaucracy_study\analysis\VAR2_v12.3\var2_analysis.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

    print(f"Created: {output_path}")
    return True

if __name__ == "__main__":
    create_var2_notebook()
    print("Notebook 4 created successfully!")
