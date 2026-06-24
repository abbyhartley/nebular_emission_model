from pathlib import Path
import sys
import numpy as np
from astropy.table import Table

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
sys.path.insert(0, str(REPO / "src"))

from normflow.stellar_mass import desi_to_sdss_gmr, log10_ml_r_from_gmr_sdss

infile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/fastspec_zall_combined_selected.fits")
outfile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

t = Table.read(infile, hdu=1)

# ----------------------------
# Compute color-derived logM*
# ----------------------------
gmr_decam_rest = np.asarray(t["ABSMAG10_DECAM_G"], dtype=float) - np.asarray(t["ABSMAG10_DECAM_R"], dtype=float)
gmr_sdss_equiv = np.asarray(desi_to_sdss_gmr(gmr_decam_rest), dtype=float)

log10_ML_r = np.asarray(log10_ml_r_from_gmr_sdss(gmr_sdss_equiv), dtype=float)

M_r_sdss = np.asarray(t["ABSMAG01_SDSS_R"], dtype=float)
Msun_r = 4.64
log10_L_r = -0.4 * (M_r_sdss - Msun_r)

logM_color = log10_ML_r + log10_L_r
t["LOGM_COLOR"] = logM_color.astype(np.float32)


# ----------------------------
# Calibrate Hα flux to SDSS-like scale
# Derived on a matched SDSS–DESI overlap sample with training-like quality cuts:
#   Δ = log10(F_Ha_SDSS) - log10(F_Ha_DESI)
#   median(Δ) = 0.14521536  ->  scale = 10**Δ = 1.39706097
# ----------------------------
# ---- Global flux-scale calibration (apply to ALL emission lines, including Hα) ----
SCALE_DEX = 0.14521535987543344
SCALE = 10**SCALE_DEX  # 1.397060970109854

LINE_FLUX_COLS = [
    "HALPHA_FLUX",
    "HBETA_FLUX",
    "HGAMMA_FLUX",
    "NII_6584_FLUX",
    "OIII_5007_FLUX",
    "OII_3726_FLUX",
    "OII_3729_FLUX",
    "SII_6716_FLUX",
    "SII_6731_FLUX",
]

for col in LINE_FLUX_COLS:
    if col not in t.colnames:
        raise KeyError(f"Missing required DESI line flux column: {col}")

    orig = np.asarray(t[col], dtype=float)
    t[col + "_ORIG"] = orig.astype(np.float32)
    t[col] = (orig * SCALE).astype(np.float32)

print(f"Applied global flux scaling to all lines: factor={SCALE:.6f} (dex={SCALE_DEX:.6f})")


# ----------------------------
# Keep only finite and within 0.1–99.9 percentiles of LOGM_COLOR
# ----------------------------
m = np.asarray(t["LOGM_COLOR"], dtype=float)
finite = np.isfinite(m)

m_f = m[finite]
p_lo, p_hi = np.percentile(m_f, [0.1, 99.9])

keep = finite & (m >= p_lo) & (m <= p_hi)

print("N total:", len(t))
print("N finite LOGM_COLOR:", int(finite.sum()))
print(f"Percentile cut: [{p_lo:.4f}, {p_hi:.4f}]")
print("N kept:", int(keep.sum()), f"({keep.mean():.3%})")
print("N removed:", int((~keep).sum()), f"({(~keep).mean():.3%})")

t_keep = t[keep]

# Optional: quick post-cut summary
m2 = np.asarray(t_keep["LOGM_COLOR"], dtype=float)
print("Post-cut LOGM_COLOR min/max:", float(np.min(m2)), float(np.max(m2)))
print("Post-cut LOGM_COLOR median:", float(np.median(m2)))

# Hα summary
ha2 = np.asarray(t_keep["HALPHA_FLUX"], dtype=float)
print("Post-cut HALPHA_FLUX (calibrated) median:", float(np.median(ha2)))

print("HALPHA_FLUX_ORIG median:", float(np.median(np.asarray(t_keep["HALPHA_FLUX_ORIG"], float))))

x = np.asarray(t_keep["HALPHA_FLUX_ORIG"], float)
print("ORIG p1/p50/p99:", np.percentile(x, [1,50,99]))
y = np.asarray(t_keep["HALPHA_FLUX"], float)
print("SCALED p1/p50/p99:", np.percentile(y, [1,50,99]))

# ----------------------------
# Write new FITS
# ----------------------------
t_keep.write(outfile, format="fits", overwrite=True)
print("Wrote:", outfile)
