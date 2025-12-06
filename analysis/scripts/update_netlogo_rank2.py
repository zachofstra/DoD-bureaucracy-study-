"""
Update NetLogo Model with Rank=2 VECM Coefficients
===================================================
Read alpha, beta, gamma from VECM_Rank2_Final_Executive_Summary
and update the NetLogo model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Paths
BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
INPUT_DIR = BASE_DIR / "VECM_Rank2_Final_Executive_Summary"
NETLOGO_DIR = BASE_DIR / "netlogo"
NETLOGO_FILE = NETLOGO_DIR / "DoD_Bureaucracy_VECM_v12.3.nlogo"
OUTPUT_FILE = NETLOGO_DIR / "DoD_Bureaucracy_VECM_Rank2.nlogo"

SELECTED_VARS = [
    'Junior_Enlisted_Z',
    'Company_Grade_Officers_Z',
    'Field_Grade_Officers_Z',
    'GOFOs_Z',
    'Warrant_Officers_Z',
    'Policy_Count_Log',
    'Total_PAS_Z',
    'FOIA_Simple_Days_Z'
]

print("=" * 80)
print("UPDATING NETLOGO MODEL WITH RANK=2 COEFFICIENTS")
print("=" * 80)

# Load coefficients
print("\n[1] Loading VECM coefficients...")
alpha_df = pd.read_excel(INPUT_DIR / "alpha_matrix_rank2.xlsx", index_col=0)
beta_df = pd.read_excel(INPUT_DIR / "beta_matrix_rank2.xlsx", index_col=0)
gamma_df = pd.read_excel(INPUT_DIR / "gamma_matrix_rank2.xlsx", index_col=0)

print(f"    Alpha: {alpha_df.shape} (error correction speeds)")
print(f"    Beta: {beta_df.shape} (cointegration vectors)")
print(f"    Gamma: {gamma_df.shape} (short-run VAR)")

# Sanitize variable names for NetLogo
def sanitize_varname(name):
    """Convert variable name to valid NetLogo identifier."""
    name = name.replace('_Z', '')
    name = name.replace('_', '-')
    name = name.lower()
    return name

nl_varnames = [sanitize_varname(v) for v in SELECTED_VARS]
var_mapping = dict(zip(SELECTED_VARS, nl_varnames))

print("\n[2] Variable mapping:")
for orig, nl in var_mapping.items():
    print(f"    {orig:30s} -> {nl}")

# Generate error correction procedure
print("\n[3] Generating error correction procedure...")

error_correction_code = "to apply-error-correction\n"
error_correction_code += "  ; Calculate equilibrium deviations (2 cointegrating vectors)\n"

for eq in range(2):  # rank=2
    error_correction_code += f"  set equilibrium-{eq+1} ("
    terms = []
    for i, var in enumerate(SELECTED_VARS):
        coef = beta_df.iloc[i, eq]
        nl_var = var_mapping[var]
        if abs(coef) > 0.01:
            terms.append(f"({coef:.4f} * {nl_var})")
    error_correction_code += " + ".join(terms)
    error_correction_code += ")\n"
    error_correction_code += f"  set error-{eq+1} equilibrium-{eq+1}\n"

error_correction_code += "\n  ; Apply error correction\n"
for i, var in enumerate(SELECTED_VARS):
    nl_var = var_mapping[var]
    correction_terms = []
    for eq in range(2):
        alpha_coef = alpha_df.iloc[i, eq]
        if abs(alpha_coef) > 0.01:
            correction_terms.append(f"({alpha_coef:.4f} * error-correction-strength * error-{eq+1} * 0.01)")

    if correction_terms:
        error_correction_code += f"  set {nl_var} {nl_var} + " + " + ".join(correction_terms) + "\n"

error_correction_code += "end\n"

print("    Error correction procedure generated")
print(f"    Lines: {len(error_correction_code.split(chr(10)))}")

# Generate VAR dynamics from gamma matrix
print("\n[4] Generating VAR dynamics procedure...")

var_dynamics_code = "to apply-var-dynamics\n"
var_dynamics_code += "  ; VAR dynamics from gamma matrix (short-run effects)\n"
var_dynamics_code += "  let vs var-strength\n\n"

for i, var_to in enumerate(SELECTED_VARS):
    nl_var_to = var_mapping[var_to]

    # Find significant coefficients
    significant_effects = []
    for j, var_from in enumerate(SELECTED_VARS):
        if i == j:  # Skip diagonal
            continue
        nl_var_from = var_mapping[var_from]
        coef = gamma_df.iloc[i, j]

        if abs(coef) > 0.05:  # Only include coefficients > 0.05
            significant_effects.append((nl_var_from, coef))

    if significant_effects:
        var_dynamics_code += f"  ; Effects on {nl_var_to}\n"
        for nl_var_from, coef in significant_effects:
            var_dynamics_code += f"  set {nl_var_to} {nl_var_to} + ({coef:.4f} * vs * lag1-{nl_var_from} * 0.01)\n"
        var_dynamics_code += "\n"

var_dynamics_code += "end\n"

print("    VAR dynamics procedure generated")
print(f"    Lines: {len(var_dynamics_code.split(chr(10)))}")

# Read existing NetLogo file
print("\n[5] Reading existing NetLogo model...")
with open(NETLOGO_FILE, 'r', encoding='utf-8') as f:
    netlogo_content = f.read()

print(f"    File size: {len(netlogo_content):,} characters")

# Replace error correction procedure
print("\n[6] Updating error correction procedure...")
pattern_ec = r'to apply-error-correction\n.*?end\n'
netlogo_content_updated = re.sub(pattern_ec, error_correction_code, netlogo_content, flags=re.DOTALL)

if netlogo_content == netlogo_content_updated:
    print("    WARNING: Error correction procedure not found - adding it")
    # Find where to insert (after apply-var-dynamics or at end of code section)
    interface_marker = "@#$#@#$#@"
    code_end = netlogo_content.find(interface_marker)
    if code_end > 0:
        netlogo_content_updated = netlogo_content[:code_end] + "\n" + error_correction_code + "\n" + netlogo_content[code_end:]
else:
    print("    [OK] Error correction procedure updated")

# Replace VAR dynamics procedure
print("\n[7] Updating VAR dynamics procedure...")
pattern_var = r'to apply-var-dynamics\n.*?end\n'
netlogo_content_final = re.sub(pattern_var, var_dynamics_code, netlogo_content_updated, flags=re.DOTALL)

if netlogo_content_updated == netlogo_content_final:
    print("    WARNING: VAR dynamics procedure not found - adding it")
    interface_marker = "@#$#@#$#@"
    code_end = netlogo_content_updated.find(interface_marker)
    if code_end > 0:
        netlogo_content_final = netlogo_content_updated[:code_end] + "\n" + var_dynamics_code + "\n" + netlogo_content_updated[code_end:]
else:
    print("    [OK] VAR dynamics procedure updated")

# Update header comment
print("\n[8] Updating model metadata...")
header_pattern = r'; Auto-generated from.*?\n; \=+'
new_header = f"""; Auto-generated from VECM Rank=2 Analysis
; Updated with rank=2 coefficients (2 cointegration vectors)
; k_ar_diff=1, BIC=128, MAPE=53%
; ={'=' * 76}"""
netlogo_content_final = re.sub(header_pattern, new_header, netlogo_content_final, flags=re.DOTALL)

print("    [OK] Metadata updated")

# Write updated file
print("\n[9] Writing updated NetLogo model...")
with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='\n') as f:
    f.write(netlogo_content_final)

print(f"    [OK] Written to: {OUTPUT_FILE}")
print(f"    File size: {len(netlogo_content_final):,} characters")

# Summary
print("\n" + "=" * 80)
print("UPDATE COMPLETE!")
print("=" * 80)
print(f"\nNew NetLogo model: {OUTPUT_FILE.name}")
print(f"\nUpdates applied:")
print(f"  - Error correction with rank=2 (2 equilibrium vectors)")
print(f"  - Alpha coefficients (8×2 matrix)")
print(f"  - Beta coefficients (8×2 matrix)")
print(f"  - Gamma VAR dynamics (8×8 matrix, lag=1)")
print(f"\nModel parameters:")
print(f"  - Cointegration rank: 2")
print(f"  - Lag order: 1 (k_ar_diff=1)")
print(f"  - Variables: {len(SELECTED_VARS)}")
print(f"  - Out-of-sample MAPE: 53.0%")
print(f"  - BIC: 128")
print("\nYou can now open {OUTPUT_FILE.name} in NetLogo 6.4!")
print("=" * 80)
