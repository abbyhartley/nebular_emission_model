import pandas as pd
import numpy as np

df = pd.read_parquet("um_a0.911185_z0p1_conditions.parquet")

sm = df["SM"].to_numpy(float)
print("min SM:", np.min(sm))
print("p50 SM:", np.median(sm))
print("p90 SM:", np.percentile(sm, 90))
print("p99 SM:", np.percentile(sm, 99))
print("max SM:", np.max(sm))

print("N with SM>1e9:", np.sum(sm > 1e9), " / ", len(sm))
print("N with SM>1e10:", np.sum(sm > 1e10), " / ", len(sm))
