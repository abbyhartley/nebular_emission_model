from pathlib import Path
import sys
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
sys.path.insert(0, str(REPO / "src"))

from normflow.stellar_mass import log10_ml_r_from_gmr_sdss

infile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo_selected_test.fits")
outfile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")

t = Table.read(infile, hdu=1)

# ----------------------------
# Compute color-derived logM*
# ----------------------------
z  = np.asarray(t["Z_1"], dtype=float)
mg = np.asarray(t["corrmag_g"], dtype=float)
mr = np.asarray(t["corrmag_r"], dtype=float)
kg = np.asarray(t["kcorr_g"], dtype=float)
kr = np.asarray(t["kcorr_r"], dtype=float)

DM = np.asarray(cosmo.distmod(z).value, dtype=float)

Mg = mg - DM - kg
Mr = mr - DM - kr
gmr_sdss_rest = Mg - Mr

log10_ML_r = np.asarray(log10_ml_r_from_gmr_sdss(gmr_sdss_rest), dtype=float)

Msun_r = 4.64
log10_L_r = -0.4 * (Mr - Msun_r)

logM_color = log10_ML_r + log10_L_r

# add as a new column (will be saved)
t["LOGM_COLOR"] = logM_color.astype(np.float32)

# ----------------------------
# Keep only finite and within 0.1–99.9 percentiles
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

# ----------------------------
# Write new FITS (preserve all old columns + new LOGM_COLOR)
# ----------------------------
# Save as a standard single-table FITS; overwrite if desired.
t_keep.write(outfile, format="fits", overwrite=True)
print("Wrote:", outfile)
