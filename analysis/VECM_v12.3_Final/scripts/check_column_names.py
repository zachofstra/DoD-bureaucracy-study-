"""Check column names in the matrices"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\zachh\Desktop\CAS593_git\analysis\VECM_v12.3_Final")
INPUT_DIR = BASE_DIR / "VECM_Rank2_Final_Executive_Summary"

alpha_df = pd.read_excel(INPUT_DIR / "alpha_matrix_rank2.xlsx", index_col=0)
beta_df = pd.read_excel(INPUT_DIR / "beta_matrix_rank2.xlsx", index_col=0)

print("Alpha columns:", alpha_df.columns.tolist())
print("Beta columns:", beta_df.columns.tolist())
print("\nAlpha shape:", alpha_df.shape)
print("Beta shape:", beta_df.shape)
