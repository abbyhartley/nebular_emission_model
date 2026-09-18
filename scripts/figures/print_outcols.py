# print_outcols.py
from pathlib import Path
import pickle

meta_path = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main_meta.pkl")

with open(meta_path, "rb") as f:
    meta = pickle.load(f)

print("Resolved out_cols:")
for i, name in enumerate(meta["resolved"]["out_cols"]):
    print(i, name)
