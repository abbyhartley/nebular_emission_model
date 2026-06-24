from pathlib import Path
import pandas as pd
import numpy as np

inp = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/UM_comp/um_a0.911185_z0p1_conditions.parquet")

df = pd.read_parquet(inp)

# impose mass cut to match DESI range
LOGM_MIN = 8.3
m = np.isfinite(df["LOGM_COLOR"]) & np.isfinite(df["LOG_LHA"]) & (df["LOGM_COLOR"] >= LOGM_MIN)

df_cut = df.loc[m].copy()

out_pq = inp.with_name(f"um_a0.911185_z0p1_conditions_logMge{LOGM_MIN:.1f}.parquet")
out_csv = inp.with_name(f"um_a0.911185_z0p1_conditions_logMge{LOGM_MIN:.1f}.csv")

df_cut.to_parquet(out_pq, index=False)
df_cut.to_csv(out_csv, index=False)

print("Wrote:", out_pq)
print("Wrote:", out_csv)
print("N kept:", len(df_cut), "of", len(df))
print("LOGM median:", df_cut["LOGM_COLOR"].median())
print("LOG_LHA median:", df_cut["LOG_LHA"].median())
