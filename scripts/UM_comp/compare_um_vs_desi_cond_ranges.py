# compare_um_vs_desi_condition_ranges.py
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table

UM_PARQUET = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/UM_comp/um_a0.911185_z0p1_conditions.parquet")
DESI_FITS  = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")


def summarize(arr, name):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    return {
        "name": name,
        "N": len(arr),
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p16": float(np.percentile(arr, 16)),
        "med": float(np.percentile(arr, 50)),
        "p84": float(np.percentile(arr, 84)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }

def print_row(d):
    print(f"{d['name']:<18s} N={d['N']:<8d} "
          f"min={d['min']:.3f} p01={d['p01']:.3f} p16={d['p16']:.3f} "
          f"med={d['med']:.3f} p84={d['p84']:.3f} p99={d['p99']:.3f} max={d['max']:.3f}")

def overlap_fraction(x, lo, hi):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    x = x[m]
    return float(np.mean((x >= lo) & (x <= hi)))


def main():
    # ---- UM ----
    um = pd.read_parquet(UM_PARQUET)
    if "LOGM_COLOR" not in um.columns or "LOG_LHA" not in um.columns:
        raise KeyError("UM parquet missing LOGM_COLOR and/or LOG_LHA")

    um_logm = um["LOGM_COLOR"].to_numpy(float)
    um_loglha = um["LOG_LHA"].to_numpy(float)

    # ---- DESI training ----
    t = Table.read(DESI_FITS, hdu=1)
    if "LOGM_COLOR" not in t.colnames:
        raise KeyError("DESI training FITS missing LOGM_COLOR")
    # LOG_LHA may or may not exist; compute from Z + HALPHA_FLUX if needed
    if "LOG_LHA" in t.colnames:
        desi_loglha = np.asarray(t["LOG_LHA"], float)
    else:
        # compute observed logLHa from Z and HALPHA_FLUX (units 1e-17 erg/s/cm^2)
        from astropy.cosmology import Planck15 as cosmo
        z = np.asarray(t["Z"], float)
        ha = np.asarray(t["HALPHA_FLUX"], float)
        m = np.isfinite(z) & (z > 0) & np.isfinite(ha) & (ha > 0)
        dl_cm = cosmo.luminosity_distance(z[m]).to("cm").value
        loglha = np.full(len(t), np.nan, dtype=float)
        loglha[m] = np.log10(ha[m] * 1e-17) + np.log10(4*np.pi) + 2*np.log10(dl_cm)
        desi_loglha = loglha

    desi_logm = np.asarray(t["LOGM_COLOR"], float)

    # ---- Summaries ----
    print("\n=== LOGM_COLOR (log10 M*/Msun) ===")
    s_um_m = summarize(um_logm, "UM")
    s_de_m = summarize(desi_logm, "DESI_BGS_train")
    print_row(s_um_m)
    print_row(s_de_m)

    print("\n=== LOG_LHA (log10 L_Ha [erg/s]) ===")
    s_um_l = summarize(um_loglha, "UM")
    s_de_l = summarize(desi_loglha, "DESI_BGS_train")
    print_row(s_um_l)
    print_row(s_de_l)

    # ---- Simple extrapolation checks ----
    # fraction of UM inside DESI 1–99% box, and vice versa
    desi_m_lo, desi_m_hi = s_de_m["p01"], s_de_m["p99"]
    desi_l_lo, desi_l_hi = s_de_l["p01"], s_de_l["p99"]

    f_um_in_desi_m = overlap_fraction(um_logm, desi_m_lo, desi_m_hi)
    f_um_in_desi_l = overlap_fraction(um_loglha, desi_l_lo, desi_l_hi)

    print("\n=== Overlap fractions ===")
    print(f"UM within DESI 1–99% LOGM_COLOR range:  {f_um_in_desi_m:.3%}")
    print(f"UM within DESI 1–99% LOG_LHA range:     {f_um_in_desi_l:.3%}")

    # joint box overlap (approx)
    um_finite = np.isfinite(um_logm) & np.isfinite(um_loglha)
    f_um_in_desi_box = float(np.mean(
        (um_logm[um_finite] >= desi_m_lo) & (um_logm[um_finite] <= desi_m_hi) &
        (um_loglha[um_finite] >= desi_l_lo) & (um_loglha[um_finite] <= desi_l_hi)
    ))
    print(f"UM within DESI 1–99% joint (M*,LHa) box: {f_um_in_desi_box:.3%}")


if __name__ == "__main__":
    main()
