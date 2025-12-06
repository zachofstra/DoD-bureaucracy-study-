"""
Examine PAS relationships with Field Grades and GOFOs
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
INPUT_DIR = BASE_DIR / "VECM_Rank2_Final_Executive_Summary"

# Load matrices
alpha_df = pd.read_excel(INPUT_DIR / "alpha_matrix_rank2.xlsx", index_col=0)
beta_df = pd.read_excel(INPUT_DIR / "beta_matrix_rank2.xlsx", index_col=0)
gamma_df = pd.read_excel(INPUT_DIR / "gamma_matrix_rank2.xlsx", index_col=0)
longrun_df = pd.read_excel(INPUT_DIR / "longrun_influence_rank2.xlsx", index_col=0)

print("=" * 80)
print("EXAMINING PAS RELATIONSHIPS WITH FIELD GRADES AND GOFOS")
print("=" * 80)

# Variables of interest
vars_interest = ['Field_Grade_Officers_Z', 'GOFOs_Z', 'Total_PAS_Z']

print("\n[1] LONG-RUN INFLUENCE (Alpha * Beta)")
print("-" * 80)
print("\nPAS to Field Grades:")
print(f"  Magnitude: {longrun_df.loc['Field_Grade_Officers_Z', 'Total_PAS_Z']:.4f}")

# Calculate signed direction
signed_sum_pas_to_field = 0
for r in range(2):
    alpha_field = alpha_df.loc['Field_Grade_Officers_Z', f'EC{r+1}']
    beta_pas = beta_df.loc['Total_PAS_Z', f'EC{r+1}']
    signed_sum_pas_to_field += alpha_field * beta_pas
    print(f"  EC{r+1}: alpha_Field({alpha_field:.4f}) x beta_PAS({beta_pas:.4f}) = {alpha_field * beta_pas:.4f}")

print(f"  Total signed effect: {signed_sum_pas_to_field:.4f}")
print(f"  Direction: {'DAMPENING' if signed_sum_pas_to_field < 0 else 'AMPLIFYING'}")

print("\nPAS to GOFOs:")
print(f"  Magnitude: {longrun_df.loc['GOFOs_Z', 'Total_PAS_Z']:.4f}")

signed_sum_pas_to_gofo = 0
for r in range(2):
    alpha_gofo = alpha_df.loc['GOFOs_Z', f'EC{r+1}']
    beta_pas = beta_df.loc['Total_PAS_Z', f'EC{r+1}']
    signed_sum_pas_to_gofo += alpha_gofo * beta_pas
    print(f"  EC{r+1}: alpha_GOFO({alpha_gofo:.4f}) x beta_PAS({beta_pas:.4f}) = {alpha_gofo * beta_pas:.4f}")

print(f"  Total signed effect: {signed_sum_pas_to_gofo:.4f}")
print(f"  Direction: {'DAMPENING' if signed_sum_pas_to_gofo < 0 else 'AMPLIFYING'}")

print("\n[2] SHORT-RUN DYNAMICS (Gamma coefficients)")
print("-" * 80)
print("\nPAS(t-1) to Field Grades(t):")
gamma_pas_to_field = gamma_df.loc['Field_Grade_Officers_Z', 'Total_PAS_Z']
print(f"  gamma = {gamma_pas_to_field:.4f}")
print(f"  Direction: {'DAMPENING' if gamma_pas_to_field < 0 else 'AMPLIFYING'}")

print("\nPAS(t-1) to GOFOs(t):")
gamma_pas_to_gofo = gamma_df.loc['GOFOs_Z', 'Total_PAS_Z']
print(f"  gamma = {gamma_pas_to_gofo:.4f}")
print(f"  Direction: {'DAMPENING' if gamma_pas_to_gofo < 0 else 'AMPLIFYING'}")

print("\n[3] BETA COEFFICIENTS (Equilibrium loadings)")
print("-" * 80)
for var in vars_interest:
    print(f"\n{var}:")
    for r in range(2):
        beta_val = beta_df.loc[var, f'EC{r+1}']
        print(f"  EC{r+1}: beta = {beta_val:.4f}")

print("\n[4] ALPHA COEFFICIENTS (Error correction speeds)")
print("-" * 80)
for var in vars_interest:
    print(f"\n{var}:")
    for r in range(2):
        alpha_val = alpha_df.loc[var, f'EC{r+1}']
        print(f"  EC{r+1}: alpha = {alpha_val:.4f}")

print("\n[5] INTERPRETATION")
print("-" * 80)
print("\nKEY INSIGHT: You're looking at TWO DIFFERENT relationships!")
print("\nLONG-RUN (error correction):")
print("  PAS to Field Grades: AMPLIFYING (+0.3287)")
print("  PAS to GOFOs: AMPLIFYING (+0.0165, weak)")
print("  => In equilibrium, they grow TOGETHER")
print("\nSHORT-RUN (year-to-year dynamics):")
print("  PAS to Field Grades: DAMPENING (-0.3375)")
print("  PAS to GOFOs: DAMPENING (-0.1750)")
print("  => Year-to-year, PAS growth CONSTRAINS military leadership growth")
print("\nThis makes theoretical sense:")
print("  1. SHORT-TERM: Competition/substitution - more civilian appointees")
print("     means fewer military leadership positions (budget constraint)")
print("  2. LONG-TERM: Overall bureaucratic expansion - all leadership")
print("     roles grow together as the organization expands")
print("\nThe NETWORK DIAGRAMS show:")
print("  - LEFT (Long-run): PAS amplifies Field/GOFO (equilibrium relationship)")
print("  - RIGHT (Short-run): PAS dampens Field/GOFO (year-to-year dynamics)")

print("\n" + "=" * 80)
