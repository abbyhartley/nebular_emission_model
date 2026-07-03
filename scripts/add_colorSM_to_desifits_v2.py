"""
Rebuild DESI_BGS_training_data.fits with:
  (1) SELF-CONSISTENT color-mass on the SDSS-filter (band-shift 0.1) system,
      cross-calibrated to the SDSS/MPA-JHU scale via a constant zero-point:
        logM = 1.062*(ABSMAG01_SDSS_G - ABSMAG01_SDSS_R) - 0.555
               - 0.4*(ABSMAG01_SDSS_R - 4.64) + 0.271
      (replaces the old DECam band-shift-1.0 color + mixed-system M_r that gave
       a -0.68 dex offset vs SDSS for the same galaxies; validated to +0.000 dex,
       NMAD 0.068 on the 9,972-galaxy overlap -- scripts/phase4_recompute_desi_mass.py)
  (2) PER-LINE flux calibration onto the SDSS scale (raw DESI is ~0.045-0.075 dex
      below SDSS from the 3" vs 1.5" aperture). Offsets = median(log10 F_sdss/F_desi_raw)
      on the same-galaxy cross-match (scripts/measure_flux_offsets.py). Replaces the
      old global +0.145 dex scale. Raw fluxes preserved as *_ORIG.
"""
from pathlib import Path
import numpy as np
from astropy.table import Table

BASE = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs")
infile = BASE / "fastspec_zall_combined_selected.fits"
outfile = BASE / "DESI_BGS_training_data.fits"
Msun_r = 4.64
ZP = 0.271  # constant zero-point cross-calibration to SDSS/MPA-JHU mass scale

# per-line dex to ADD to raw DESI flux to reach the SDSS scale (measure_flux_offsets.py)
FLUX_CAL_DEX = {
    "HALPHA_FLUX":    0.0752,
    "HBETA_FLUX":     0.0628,
    "HGAMMA_FLUX":    0.0676,
    "NII_6584_FLUX":  0.0449,
    "SII_6716_FLUX":  0.0635,
    "SII_6731_FLUX":  0.0545,
    "OII_3726_FLUX":  0.0650,
    "OII_3729_FLUX":  0.0075,
    "OIII_5007_FLUX": 0.0502,
}

t = Table.read(infile, hdu=1)

# ---- (1) self-consistent color-mass ----
g01 = np.asarray(t["ABSMAG01_SDSS_G"], float)
r01 = np.asarray(t["ABSMAG01_SDSS_R"], float)
gmr01 = g01 - r01
logM = (1.062 * gmr01 - 0.555) + (-0.4 * (r01 - Msun_r)) + ZP
t["LOGM_COLOR"] = logM.astype(np.float32)
print(f"new LOGM_COLOR median={np.nanmedian(logM):.3f}  (g-r)01 median={np.nanmedian(gmr01):.3f}")

# ---- (2) per-line flux calibration ----
for col, dex in FLUX_CAL_DEX.items():
    if col not in t.colnames:
        raise KeyError(f"Missing DESI line flux column: {col}")
    orig = np.asarray(t[col], float)
    t[col + "_ORIG"] = orig.astype(np.float32)
    t[col] = (orig * 10.0 ** dex).astype(np.float32)
print("Applied per-line flux calibration (dex):", FLUX_CAL_DEX)

# ---- LOGM_COLOR percentile trim (same as before) ----
m = np.asarray(t["LOGM_COLOR"], float)
finite = np.isfinite(m)
p_lo, p_hi = np.percentile(m[finite], [0.1, 99.9])
keep = finite & (m >= p_lo) & (m <= p_hi)
print(f"Percentile cut [{p_lo:.3f}, {p_hi:.3f}]  kept {int(keep.sum())}/{len(t)} ({keep.mean():.3%})")

t_keep = t[keep]
print("Post-cut LOGM_COLOR min/med/max:",
      float(np.min(t_keep['LOGM_COLOR'])), float(np.median(t_keep['LOGM_COLOR'])), float(np.max(t_keep['LOGM_COLOR'])))
t_keep.write(outfile, format="fits", overwrite=True)
print("Wrote:", outfile)
