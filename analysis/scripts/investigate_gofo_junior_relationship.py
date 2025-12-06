"""
Investigate GOFOs vs Junior Enlisted relationship
User's observation: GOFOs went UP, Junior Enlisted went DOWN
But VECM shows "amplifying" - this seems wrong!
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
INPUT_DIR = BASE_DIR / "VECM_Rank2_Final_Executive_Summary"

# Load matrices
alpha_df = pd.read_excel(INPUT_DIR / "alpha_matrix_rank2.xlsx", index_col=0)
beta_df = pd.read_excel(INPUT_DIR / "beta_matrix_rank2.xlsx", index_col=0)

print("=" * 80)
print("INVESTIGATING GOFOs vs JUNIOR ENLISTED RELATIONSHIP")
print("=" * 80)

print("\n[1] BETA COEFFICIENTS (Define equilibrium relationships)")
print("-" * 80)
print("\nJunior Enlisted:")
for r in range(2):
    print(f"  EC{r+1}: beta = {beta_df.loc['Junior_Enlisted_Z', f'EC{r+1}']:.4f}")

print("\nGOFOs:")
for r in range(2):
    print(f"  EC{r+1}: beta = {beta_df.loc['GOFOs_Z', f'EC{r+1}']:.4f}")

print("\n[2] WHAT THIS MEANS:")
print("-" * 80)
print("\nCointegration vector EC1:")
beta_je_ec1 = beta_df.loc['Junior_Enlisted_Z', 'EC1']
beta_gofo_ec1 = beta_df.loc['GOFOs_Z', 'EC1']
print(f"  Junior Enlisted: {beta_je_ec1:.4f}")
print(f"  GOFOs: {beta_gofo_ec1:.4f}")

if np.sign(beta_je_ec1) == np.sign(beta_gofo_ec1):
    print("  => SAME SIGN: In equilibrium, they move TOGETHER")
else:
    print("  => OPPOSITE SIGNS: In equilibrium, they move OPPOSITE")

print("\nCointegration vector EC2:")
beta_je_ec2 = beta_df.loc['Junior_Enlisted_Z', 'EC2']
beta_gofo_ec2 = beta_df.loc['GOFOs_Z', 'EC2']
print(f"  Junior Enlisted: {beta_je_ec2:.4f}")
print(f"  GOFOs: {beta_gofo_ec2:.4f}")

if np.sign(beta_je_ec2) == np.sign(beta_gofo_ec2):
    print("  => SAME SIGN: In equilibrium, they move TOGETHER")
else:
    print("  => OPPOSITE SIGNS: In equilibrium, they move OPPOSITE")

print("\n[3] ALPHA COEFFICIENTS (Error correction speeds)")
print("-" * 80)
print("\nJunior Enlisted:")
for r in range(2):
    print(f"  EC{r+1}: alpha = {alpha_df.loc['Junior_Enlisted_Z', f'EC{r+1}']:.4f}")

print("\nGOFOs:")
for r in range(2):
    print(f"  EC{r+1}: alpha = {alpha_df.loc['GOFOs_Z', f'EC{r+1}']:.4f}")

print("\n[4] INFLUENCE CALCULATION: GOFOs -> Junior Enlisted")
print("-" * 80)

influence_sum = 0
for r in range(2):
    alpha_je = alpha_df.loc['Junior_Enlisted_Z', f'EC{r+1}']
    beta_gofo = beta_df.loc['GOFOs_Z', f'EC{r+1}']
    influence = alpha_je * beta_gofo
    influence_sum += influence
    print(f"\nEC{r+1}:")
    print(f"  alpha[Junior_Enlisted] = {alpha_je:.4f}")
    print(f"  beta[GOFOs] = {beta_gofo:.4f}")
    print(f"  Influence = {alpha_je:.4f} x {beta_gofo:.4f} = {influence:.4f}")

print(f"\nTotal influence (signed): {influence_sum:.4f}")
print(f"Direction: {'AMPLIFYING' if influence_sum > 0 else 'DAMPENING'}")

print("\n[5] WHAT 'AMPLIFYING' ACTUALLY MEANS IN VECM")
print("-" * 80)
print("\nCRITICAL: alpha[i,r] * beta[j,r] tells us ERROR CORRECTION, not trend!")
print("\nIf GOFOs -> Junior Enlisted is 'amplifying' (+):")
print("  - When GOFOs deviate HIGH from equilibrium")
print("  - Junior Enlisted INCREASES to restore equilibrium")
print("\nIf GOFOs -> Junior Enlisted is 'dampening' (-):")
print("  - When GOFOs deviate HIGH from equilibrium")
print("  - Junior Enlisted DECREASES to restore equilibrium")

print("\n[6] USER'S EMPIRICAL OBSERVATION")
print("-" * 80)
print("  GOFOs: INCREASED over 1987-2024")
print("  Junior Enlisted: DECREASED over 1987-2024")
print("  => They moved in OPPOSITE directions")
print("\nThis suggests:")
print("  1. Beta coefficients should have OPPOSITE signs (inverse equilibrium)")
print("  2. OR they're not truly cointegrated")
print("  3. OR the 'amplifying' label is confusing error correction with trends")

print("\n" + "=" * 80)
print("RECOMMENDATION: Check if beta signs match empirical direction")
print("=" * 80)
