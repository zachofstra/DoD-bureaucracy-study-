"""
Check if the sign correction actually fixed the GOFOs-Junior Enlisted relationship
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis")
DATA_FILE = BASE_DIR / "complete_normalized_dataset_v12.3.xlsx"
ORIGINAL_DIR = BASE_DIR / "VECM_v12.3_Final" / "VECM_Rank2_Final_Executive_Summary"
CORRECTED_DIR = BASE_DIR / "VECM_v12.3_Final" / "VECM_Rank2_CORRECTED"

# Load data to get empirical correlation
df = pd.read_excel(DATA_FILE)
df_clean = df.dropna(subset=['GOFOs_Z', 'Junior_Enlisted_Z'])
gofos = df_clean['GOFOs_Z'].values
je = df_clean['Junior_Enlisted_Z'].values
empirical_corr = np.corrcoef(gofos, je)[0, 1]

print("=" * 80)
print("VERIFYING GOFOs => JUNIOR ENLISTED CORRECTION")
print("=" * 80)

print(f"\nEMPIRICAL RELATIONSHIP:")
print(f"  Correlation: {empirical_corr:+.3f}")
print(f"  Interpretation: {'MOVE TOGETHER' if empirical_corr > 0 else 'MOVE OPPOSITE'}")

# Load ORIGINAL matrices
alpha_orig = pd.read_excel(ORIGINAL_DIR / "alpha_matrix_rank2.xlsx", index_col=0)
beta_orig = pd.read_excel(ORIGINAL_DIR / "beta_matrix_rank2.xlsx", index_col=0)

# Calculate ORIGINAL long-run influence
orig_influence = 0
for r in range(2):
    alpha_je = alpha_orig.loc['Junior_Enlisted_Z', f'EC{r+1}']
    beta_gofo = beta_orig.loc['GOFOs_Z', f'EC{r+1}']
    orig_influence += alpha_je * beta_gofo

print(f"\nORIGINAL MODEL (wrong signs overall):")
print(f"  GOFOs => Junior Enlisted influence: {orig_influence:+.4f}")
print(f"  Direction: {'AMPLIFYING' if orig_influence > 0 else 'DAMPENING'}")
if (orig_influence > 0 and empirical_corr > 0) or (orig_influence < 0 and empirical_corr < 0):
    print(f"  Match with empirical: CORRECT")
else:
    print(f"  Match with empirical: WRONG")

# Load CORRECTED matrices
alpha_corr = pd.read_excel(CORRECTED_DIR / "alpha_matrix_rank2_CORRECTED.xlsx", index_col=0)
beta_corr = pd.read_excel(CORRECTED_DIR / "beta_matrix_rank2_CORRECTED.xlsx", index_col=0)

# Calculate CORRECTED long-run influence
corr_influence = 0
for r in range(2):
    alpha_je = alpha_corr.loc['Junior_Enlisted_Z', f'EC{r+1}']
    beta_gofo = beta_corr.loc['GOFOs_Z', f'EC{r+1}']
    corr_influence += alpha_je * beta_gofo
    print(f"\n  EC{r+1}: alpha_JE={alpha_je:+.4f} × beta_GOFO={beta_gofo:+.4f} = {alpha_je * beta_gofo:+.4f}")

print(f"\nCORRECTED MODEL (flipped vectors):")
print(f"  GOFOs => Junior Enlisted influence: {corr_influence:+.4f}")
print(f"  Direction: {'AMPLIFYING' if corr_influence > 0 else 'DAMPENING'}")
if (corr_influence > 0 and empirical_corr > 0) or (corr_influence < 0 and empirical_corr < 0):
    print(f"  Match with empirical: CORRECT")
else:
    print(f"  Match with empirical: WRONG")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if abs(orig_influence - corr_influence) < 0.001:
    print("\nThe 'correction' did NOT change the GOFOs-Junior Enlisted relationship!")
    print("Both models show the same influence direction.")
elif (orig_influence > 0) == (corr_influence > 0):
    print("\nThe 'correction' kept the same direction for GOFOs-Junior Enlisted.")
    print("The sign is still WRONG relative to empirical correlation.")
else:
    print("\nThe 'correction' FLIPPED the GOFOs-Junior Enlisted relationship.")
    if (corr_influence > 0 and empirical_corr < 0) or (corr_influence < 0 and empirical_corr > 0):
        print("But it made it WORSE - now it's even more wrong!")
    else:
        print("And it's now correct!")

print("\n" + "=" * 80)
print("ROOT CAUSE")
print("=" * 80)
print("""
The sign correction flips ENTIRE cointegration vectors based on MAJORITY voting.
This means:
  - If 4 out of 7 variables have wrong signs, we flip the whole vector
  - But the other 3 variables get flipped too (making them wrong)
  - Individual relationships like GOFOs-Junior Enlisted can still be incorrect

The RANK=2 VECM specification forces these 8 variables into just 2 cointegration
relationships. This oversimplification cannot accurately capture all the complex
directional relationships in your data.

RECOMMENDATION:
  - The VECM may not be the right model for variables with such different
    directional relationships
  - Consider separating variables into groups that actually cointegrate
  - Or accept that some relationships will be incorrectly specified
""")

print("=" * 80)
