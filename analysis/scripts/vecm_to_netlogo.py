"""
VECM Lag 2 -> NetLogo Model Generator
Converts VECM lag 2 analysis results into executable NetLogo agent-based model

Based on data_to_netlogo_pipeline.ipynb methodology
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM

print("=" * 100)
print("VECM LAG 2 -> NETLOGO MODEL GENERATOR")
print("=" * 100)

# =============================================================================
# LOAD DATA AND RE-ESTIMATE VECM
# =============================================================================
print("\n[1/5] Loading data and re-estimating VECM lag 2...")

df = pd.read_excel('data/analysis/complete_normalized_dataset_v10.6_FULL.xlsx')

endog_vars = [
    'Junior_Enlisted_Z',
    'FOIA_Simple_Days_Z',
    'Total_PAS_Z',
    'Total_Civilians_Z',
    'Policy_Count_Log',
    'Field_Grade_Officers_Z',
    'GOFOs_Z'
]

exog_vars = ['GDP_Growth', 'Major_Conflict']

data = df[endog_vars + exog_vars].dropna()
endog_data = data[endog_vars]
exog_data = data[exog_vars]

# Estimate VECM
vecm_model = VECM(endog_data, exog=exog_data, k_ar_diff=2, coint_rank=4, deterministic='ci')
vecm_result = vecm_model.fit()

# Extract coefficients
alpha = vecm_result.alpha  # Error correction (7x4)
beta = vecm_result.beta    # Cointegrating vectors (7x4)
gamma_full = vecm_result.gamma  # Short-run dynamics (7x14)

# Reshape gamma into list of lag matrices
n_vars = len(endog_vars)
n_lags = 2
gamma = [gamma_full[:, i*n_vars:(i+1)*n_vars] for i in range(n_lags)]

print(f"  Variables: {n_vars} endogenous, {len(exog_vars)} exogenous")
print(f"  Alpha shape: {alpha.shape}")
print(f"  Beta shape: {beta.shape}")
print(f"  Gamma lags: {len(gamma)}")

# =============================================================================
# NETLOGO VARIABLE NAMES
# =============================================================================
print("\n[2/5] Creating NetLogo variable mappings...")

def sanitize_varname(name):
    """Convert to valid NetLogo identifier."""
    name = name.replace('_Z', '')
    name = name.replace('_', '-')
    name = name.lower()
    return name

nl_varnames = [sanitize_varname(v) for v in endog_vars]
var_mapping = dict(zip(endog_vars, nl_varnames))

print("  Variable mapping:")
for orig, nl in var_mapping.items():
    print(f"    {orig:30s} -> {nl}")

# =============================================================================
# GENERATE NETLOGO CODE SECTIONS
# =============================================================================
print("\n[3/5] Generating NetLogo code sections...")

# --- GLOBALS ---
def generate_globals(nl_varnames, n_lags, coint_rank):
    code = "globals [\n"
    code += "  ; === ENDOGENOUS VARIABLES (normalized) ===\n"
    for var in nl_varnames:
        code += f"  {var}\n"

    code += "\n  ; === LAG VARIABLES ===\n"
    for lag in range(1, n_lags + 1):
        code += f"  ; Lag {lag}\n"
        for var in nl_varnames:
            code += f"  lag{lag}-{var}\n"

    code += "\n  ; === EQUILIBRIUM & ERROR CORRECTION ===\n"
    for eq in range(coint_rank):
        code += f"  equilibrium-{eq+1}\n"
        code += f"  error-{eq+1}\n"

    code += "\n  ; === TIME TRACKING ===\n"
    code += "  year\n"
    code += "  tick-count\n"

    code += "\n  ; === HISTORY TRACKING ===\n"
    for var in nl_varnames:
        code += f"  history-{var}\n"

    code += "]\n"
    return code

# --- SETUP ---
def generate_setup(nl_varnames):
    code = "to setup\n"
    code += "  clear-all\n"
    code += "  reset-ticks\n\n"
    code += "  set year 1987\n"
    code += "  set tick-count 0\n\n"
    code += "  ; Initialize to baseline (normalized 0)\n"
    for var in nl_varnames:
        code += f"  set {var} 0\n"
    code += "\n  update-lags\n\n"
    code += "  ; Initialize history\n"
    for var in nl_varnames:
        code += f"  set history-{var} []\n"
    code += "  record-history\n\n"
    code += "  setup-patches\n"
    code += "end\n\n"
    code += "to setup-patches\n"
    code += "  ask patches [ set pcolor white ]\n"
    code += "end\n"
    return code

# --- UPDATE LAGS ---
def generate_update_lags(nl_varnames, n_lags):
    code = "to update-lags\n"
    # Shift backwards
    for lag in range(n_lags, 1, -1):
        code += f"  ; Shift lag{lag-1} to lag{lag}\n"
        for var in nl_varnames:
            code += f"  set lag{lag}-{var} lag{lag-1}-{var}\n"
    code += "  ; Shift current to lag1\n"
    for var in nl_varnames:
        code += f"  set lag1-{var} {var}\n"
    code += "end\n"
    return code

# --- ERROR CORRECTION ---
def generate_error_correction(nl_varnames, alpha, beta, coint_rank):
    code = "to apply-error-correction\n"
    code += "  ; CRITICAL: Calculate equilibrium deviations using LAGGED values (beta' * Y_{t-1})\n"
    code += "  ; This is the correct VECM specification: ECT_t = beta' * Y_{t-1}\n"
    for eq in range(coint_rank):
        code += f"  set equilibrium-{eq+1} ("
        terms = []
        for i, var in enumerate(nl_varnames):
            coef = beta[i, eq]
            if abs(coef) > 0.01:
                terms.append(f"({coef:.6f} * lag1-{var})")  # FIX: Use lag1 values!
        code += " + ".join(terms) if terms else "0"
        code += ")\n"
        code += f"  set error-{eq+1} equilibrium-{eq+1}\n"

    code += "\n  ; Apply error correction (alpha * error) with reduced scaling for stability\n"
    code += "  ; Note: Some alpha coefficients are very large (e.g., FOIA = -3.066)\n"
    for i, var in enumerate(nl_varnames):
        correction_terms = []
        for eq in range(coint_rank):
            alpha_coef = alpha[i, eq]
            if abs(alpha_coef) > 0.01:
                correction_terms.append(f"({alpha_coef:.6f} * error-correction-strength * error-{eq+1} * 0.05)")  # FIX: Use 0.05 scaling
        if correction_terms:
            code += f"  set {var} {var} + (" + " + ".join(correction_terms) + ")\n"
    code += "end\n"
    return code

# --- VAR DYNAMICS (SHORT-RUN) ---
def generate_var_dynamics(nl_varnames, gamma):
    code = "to apply-var-dynamics\n"
    code += "  let vs var-strength\n\n"

    for lag_idx, gamma_matrix in enumerate(gamma):
        lag_num = lag_idx + 1
        code += f"  ; === LAG {lag_num} SHORT-RUN DYNAMICS ===\n"
        for i, var_to in enumerate(nl_varnames):
            terms = []
            for j, var_from in enumerate(nl_varnames):
                coef = gamma_matrix[i, j]
                if abs(coef) > 0.05:  # Only significant coefficients
                    terms.append(f"({coef:.6f} * lag{lag_num}-{var_from})")
            if terms:
                code += f"  set {var_to} {var_to} + (vs * 0.1 * (" + " + ".join(terms) + "))\n"
        code += "\n"

    code += "end\n"
    return code

# --- POLYNOMIAL INTERACTIONS ---
def generate_interactions(nl_varnames):
    code = "to apply-polynomial-interactions\n"
    code += "  if not use-interactions? [ stop ]\n\n"
    code += "  let is interaction-strength\n\n"

    code += "  ; Second-order: Officer growth * Policy growth -> Bureaucratic expansion\n"
    code += "  set total-pas total-pas + (lag1-field-grade-officers * lag1-policy-count-log * 0.15 * is * 0.1)\n"
    code += "  set total-civilians total-civilians + (lag1-gofos * lag1-policy-count-log * 0.12 * is * 0.1)\n\n"

    code += "  ; Third-order: Officers * Civilians * Policy -> Bureaucratic inertia\n"
    code += "  if is > 0.3 [\n"
    code += "    set foia-simple-days foia-simple-days + (lag1-field-grade-officers * lag1-total-civilians * lag1-policy-count-log * 0.08 * is * 0.1)\n"
    code += "  ]\n"
    code += "end\n"
    return code

# --- UTILITIES ---
def generate_utilities(nl_varnames):
    code = "to add-noise\n"
    for var in nl_varnames:
        code += f"  set {var} {var} + random-normal 0 (noise-level * 0.01)\n"
    code += "end\n\n"

    code += "to clip-variables\n"
    code += "  ; Clip to prevent numerical overflow\n"
    for var in nl_varnames:
        code += f"  if {var} > 10 [ set {var} 10 ]\n"
        code += f"  if {var} < -10 [ set {var} -10 ]\n"
    code += "end\n\n"

    code += "to record-history\n"
    for var in nl_varnames:
        code += f"  set history-{var} lput {var} history-{var}\n"
    code += "end\n"
    return code

# --- MAIN LOOP ---
def generate_go():
    code = "to go\n"
    code += "  if max-year > 0 and year >= max-year [ stop ]\n\n"
    code += "  update-lags\n"
    code += "  apply-error-correction\n"
    code += "  apply-var-dynamics\n"
    code += "  apply-polynomial-interactions\n"
    code += "  add-noise\n"
    code += "  clip-variables\n\n"
    code += "  set year year + 1\n"
    code += "  set tick-count tick-count + 1\n"
    code += "  record-history\n"
    code += "  tick\n"
    code += "end\n"
    return code

# --- INTERFACE ---
def generate_interface(nl_varnames):
    code = "@#$#@#$#@\n"

    # Graphics window
    code += """GRAPHICS-WINDOW
210
10
647
448
-1
-1
13.0
1
10
1
1
1
0
1
1
1
-16
16
-16
16
0
0
1
ticks
30.0

"""

    # Buttons
    code += """BUTTON
15
15
88
48
Setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
95
15
168
48
Go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

"""

    # Sliders
    code += """SLIDER
15
95
200
128
error-correction-strength
error-correction-strength
0
2
1.0
0.1
1
NIL
HORIZONTAL

SLIDER
15
135
200
168
var-strength
var-strength
0
2
1.0
0.1
1
NIL
HORIZONTAL

SLIDER
15
175
200
208
noise-level
noise-level
0
5
1.0
0.5
1
NIL
HORIZONTAL

SLIDER
15
215
200
248
interaction-strength
interaction-strength
0
1
0.5
0.1
1
NIL
HORIZONTAL

SLIDER
15
255
200
288
max-year
max-year
0
2050
0
10
1
NIL
HORIZONTAL

SWITCH
15
295
200
328
use-interactions?
use-interactions?
0
1
-1000

"""

    # Year monitor
    code += """MONITOR
670
15
750
60
Year
year
0
1
11

"""

    # Variable monitors
    monitor_y = 70
    for i, var in enumerate(nl_varnames):
        col = i % 2
        row = i // 2
        x1 = 670 + (col * 100)
        x2 = x1 + 90
        y1 = monitor_y + (row * 50)
        y2 = y1 + 45
        label = var.replace('-', ' ').title()

        code += f"""MONITOR
{x1}
{y1}
{x2}
{y2}
{label}
{var}
3
1
11

"""

    # Plot with exact color format from DoD model
    distinct_colors = [
        -13345367,  # Red (Junior Enlisted)
        -2674135,   # Orange (Field Grade Officers)
        -955883,    # Brown (GOFOs)
        -13840069,  # Blue (Total Civilians)
        -8630108,   # Green (Policy Count)
        -5825686,   # Purple (Total PAS)
        -7500403,   # Cyan (FOIA Days)
    ]

    code += """PLOT
15
460
870
680
DoD Bureaucratic Growth (VECM Lag 2)
Year
Value (normalized)
0.0
40.0
-3.0
3.0
true
true
"" ""
PENS
"""
    for i, var in enumerate(nl_varnames):
        color = distinct_colors[i]
        label = var.replace('-', ' ').title()
        code += f'"{label}" 1.0 0 {color} true "" "plot {var}"\n'

    code += "\n@#$#@#$#@\n"
    return code

# Generate all sections
print("  Generating globals...")
globals_code = generate_globals(nl_varnames, n_lags, 4)

print("  Generating setup...")
setup_code = generate_setup(nl_varnames)

print("  Generating update lags...")
lags_code = generate_update_lags(nl_varnames, n_lags)

print("  Generating error correction...")
error_corr_code = generate_error_correction(nl_varnames, alpha, beta, 4)

print("  Generating VAR dynamics...")
var_code = generate_var_dynamics(nl_varnames, gamma)

print("  Generating polynomial interactions...")
interactions_code = generate_interactions(nl_varnames)

print("  Generating utilities...")
utils_code = generate_utilities(nl_varnames)

print("  Generating main loop...")
go_code = generate_go()

print("  Generating interface...")
interface_code = generate_interface(nl_varnames)

# =============================================================================
# ASSEMBLE COMPLETE MODEL
# =============================================================================
print("\n[4/5] Assembling complete NetLogo model...")

netlogo_model = f"""; {'=' * 80}
; DoD Bureaucratic Growth - VECM Lag 2 Agent-Based Model
; {'=' * 80}
;
; Auto-generated from Vector Error Correction Model (VECM) analysis
;
; Data: DoD Bureaucracy (1987-2024)
; Variables: {n_vars} endogenous (normalized)
; Cointegration rank: 4 long-run equilibrium relationships
; Lag order: 2 (short-run dynamics)
;
; Model Structure:
;   - Error correction mechanisms (4 cointegrating vectors)
;   - Short-run VAR dynamics (2 lags)
;   - Polynomial interactions (bureaucratic complexity)
;
; Theoretical Framework: Max Weber's Iron Cage of Bureaucracy
; Research Focus: O-4 staff officer growth as bureaucratic bloat indicator
;
; {'=' * 80}

{globals_code}

; {'=' * 80}
; SETUP PROCEDURES
; {'=' * 80}

{setup_code}

; {'=' * 80}
; MAIN SIMULATION LOOP
; {'=' * 80}

{go_code}

; {'=' * 80}
; LAG UPDATE
; {'=' * 80}

{lags_code}

; {'=' * 80}
; ERROR CORRECTION (VECM - Long-run Equilibria)
; {'=' * 80}

{error_corr_code}

; {'=' * 80}
; VAR DYNAMICS (Short-run Dynamics)
; {'=' * 80}

{var_code}

; {'=' * 80}
; POLYNOMIAL INTERACTIONS (Bureaucratic Complexity)
; {'=' * 80}

{interactions_code}

; {'=' * 80}
; UTILITY PROCEDURES
; {'=' * 80}

{utils_code}
{interface_code}
"""

# Add standard NetLogo footer
netlogo_model += """## WHAT IS IT?

This agent-based model was automatically generated from Vector Error Correction Model (VECM) analysis of Department of Defense bureaucratic growth (1987-2024).

The model implements Weber's "Iron Cage of Bureaucracy" theory, showing how DoD bureaucracy has grown despite reorganization efforts since the Goldwater-Nichols Act of 1986.

## HOW IT WORKS

**VECM Components:**

1. **Error Correction** (4 long-run equilibria):
   - Vector 1: Teeth-to-Tail Shift (Junior Enlisted decline vs Field Grade growth)
   - Vector 2: Bureaucratic Delay (FOIA processing times)
   - Vector 3: Political Appointee Trade-off (PAS positions)
   - Vector 4: Civilianization (civilian workforce substitution)

2. **Short-run VAR Dynamics** (2 lags):
   - Captures immediate year-over-year changes
   - Models feedback loops between personnel and policy variables

3. **Polynomial Interactions**:
   - Second-order: Officer growth × Policy growth → Bureaucratic expansion
   - Third-order: Officers × Civilians × Policy → Bureaucratic inertia

**Key Finding**: O-4s (Majors/LT Commanders) show highest growth - the "bureaucratic bloat" layer between company command and flag officers.

## HOW TO USE IT

1. **Setup**: Click Setup to initialize all variables to normalized baseline (0)
2. **Run**: Click Go to simulate bureaucratic evolution
3. **Adjust Parameters**:
   - `error-correction-strength`: How strongly system returns to equilibrium (0-2)
   - `var-strength`: Intensity of short-run dynamics (0-2)
   - `noise-level`: Random shocks/external events (0-5)
   - `interaction-strength`: Polynomial complexity effects (0-1)
   - `use-interactions?`: Enable/disable bureaucratic complexity terms

## THINGS TO NOTICE

- **Junior Enlisted** tends to decline (negative growth) - "teeth to tail" shift
- **Field Grade Officers** show strong positive growth - bureaucratic expansion
- **FOIA Simple Days** increases - bureaucratic delay mechanism
- **Policy Count** grows steadily - regulatory accumulation
- System exhibits path dependency and institutional inertia

## THINGS TO TRY

- Set `error-correction-strength` to 0 to see pure VAR dynamics without long-run equilibrium
- Increase `interaction-strength` to amplify bureaucratic complexity effects
- Compare behavior with and without polynomial interactions

## EXTENDING THE MODEL

This model can be extended by:
- Adding exogenous shocks (GDP growth, major conflicts)
- Implementing agent heterogeneity (different DoD components)
- Adding network effects (inter-agency coordination costs)
- Modeling policy interventions (reorganization attempts)

## NETLOGO FEATURES

This model uses:
- Time series variables with lag structures
- Cointegration and error correction mechanisms
- Polynomial interaction terms
- History tracking for all variables

## RELATED MODELS

- VAR/VECM econometric models
- Organizational growth models
- Bureaucratic theory simulations

## CREDITS AND REFERENCES

**Theoretical Framework**:
- Max Weber: "Iron Cage of Bureaucracy" (rationalization theory)
- Robert Michels: "Iron Law of Oligarchy"
- Goldwater-Nichols Act (1986) - DoD reorganization

**Data Sources**:
- OPM Federal Employment Statistics (1987-2024)
- DoD DMDC Workforce Reports
- DoD Directives and Policy Database
- FOIA Annual Reports

**Methodology**:
- Johansen Cointegration Test (4 vectors at 5% significance)
- Vector Error Correction Model (VECM) at lag 2
- Robustness tests: Ljung-Box, Jarque-Bera, ARCH

**Author**: Auto-generated from CAS593 Capstone Research
**Date**: 2024
**NetLogo Version**: 6.4.0

## COPYRIGHT AND LICENSE

Research data: Public domain (U.S. Government sources)
Model code: Open source for academic use

For questions or collaboration: See CAS593_git repository
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
"""

# =============================================================================
# WRITE NETLOGO FILE
# =============================================================================
print("\n[5/5] Writing NetLogo file...")

output_file = 'data/analysis/VECM_LAG2/DoD_Bureaucratic_Growth_VECM_Lag2.nlogo'

# Critical: Use Unix (LF) line endings for NetLogo 6.4
with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
    f.write(netlogo_model)

print(f"\n{'=' * 100}")
print("SUCCESS! NetLogo model generated")
print("=" * 100)
print(f"\nFile: {output_file}")
print(f"Size: {len(netlogo_model):,} characters")
print(f"\nModel specifications:")
print(f"  - {n_vars} endogenous variables")
print(f"  - 4 cointegrating equilibria")
print(f"  - Lag order: 2")
print(f"  - {alpha.size} error correction coefficients")
print(f"  - {gamma_full.size} short-run VAR coefficients")
print(f"\nCRITICAL: File uses Unix (LF) line endings - required for NetLogo 6.4!")
print(f"You can now open this file in NetLogo 6.4.0!")
print("=" * 100)
