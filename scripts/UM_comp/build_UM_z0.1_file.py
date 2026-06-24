# build_um_a0911185_conditions_by_index.py
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------
# Inputs
# -----------------------
IN_DIR = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/UM_comp").resolve()
A_STR = "0.911185"
CHUNKS = range(0, 8)

OUT_PARQUET = IN_DIR / f"um_a{A_STR}_z0p1_conditions.parquet"
OUT_CSV     = IN_DIR / f"um_a{A_STR}_z0p1_conditions.csv"

# Column indices in the numeric rows (0-based), verified by your diagnostic
I_ID  = 0
I_SM  = 16
I_SFR = 18

# Kennicutt (1998) Salpeter: SFR [Msun/yr] = 7.9e-42 L_Ha [erg/s]
# => L_Ha = (1/7.9e-42) * SFR
K_HA = 1.0 / 7.9e-42  # 1.2658227848101266e41 erg/s per (Msun/yr)

SM_FLOOR = 1e-6
SFR_FLOOR = 1e-12


def read_chunk(fn: Path) -> pd.DataFrame:
    # Read only the needed numeric columns by position; skip all comment lines
    df = pd.read_csv(
        fn,
        delim_whitespace=True,
        comment="#",
        header=None,
        usecols=[I_ID, I_SM, I_SFR],
        names=["ID", "SM", "SFR"],
        engine="c",
    )
    return df


def main():
    dfs = []
    for i in CHUNKS:
        fn = IN_DIR / f"sfh_catalog_{A_STR}.{i}.txt"
        if not fn.exists():
            raise FileNotFoundError(fn)
        print("Reading:", fn)
        dfs.append(read_chunk(fn))

    um = pd.concat(dfs, ignore_index=True)
    print("\nConcatenated rows:", len(um))

    # Basic validity cuts
    sm = um["SM"].to_numpy(float)
    sfr = um["SFR"].to_numpy(float)
    m = np.isfinite(sm) & np.isfinite(sfr) & (sm > 0) & (sfr >= 0)
    um = um.loc[m].copy()
    print("Rows after finite/positive cuts:", len(um))

    # Build conditioning columns (match NF naming)
    um["LOGM_COLOR"] = np.log10(np.clip(um["SM"].to_numpy(float), SM_FLOOR, None))

    Lha = K_HA * np.clip(um["SFR"].to_numpy(float), SFR_FLOOR, None)
    um["LOG_LHA"] = np.log10(Lha)

    out = um[["ID", "LOGM_COLOR", "LOG_LHA", "SM", "SFR"]].copy()

    # Sanity checks (should now be galaxy-like)
    print("\nSanity checks:")
    print("SM median [Msun]:", float(np.median(out["SM"])))
    print("log10(SM) median:", float(np.median(out["LOGM_COLOR"])))
    print("SFR median [Msun/yr]:", float(np.median(out["SFR"])))
    print("log10(LHa) median:", float(np.median(out["LOG_LHA"])))

    # Write
    out.to_parquet(OUT_PARQUET, index=False)
    out.to_csv(OUT_CSV, index=False)
    print("\nWrote:", OUT_PARQUET)
    print("Wrote:", OUT_CSV)


if __name__ == "__main__":
    main()
