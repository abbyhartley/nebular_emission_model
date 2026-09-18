from pathlib import Path
import sys
import numpy as np
from astropy.table import Table

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
sys.path.insert(0, str(REPO / "src"))  # important: package lives in src/

from normflow.stellar_mass import desi_to_sdss_gmr, log10_ml_r_from_gmr_sdss

infile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/fastspec_zall_combined_selected_test.fits")
t = Table.read(infile, hdu=1)

# 1) DESI/DECam rest-frame color (g-r)
gmr_decam_rest = t["ABSMAG10_DECAM_G"] - t["ABSMAG10_DECAM_R"]

# 2) Convert to SDSS-equivalent rest color for M/L
gmr_sdss_equiv = desi_to_sdss_gmr(gmr_decam_rest)

# 3) Your calibrated relation
log10_ML_r = log10_ml_r_from_gmr_sdss(gmr_sdss_equiv)

# 4) SDSS r-band luminosity (rest-frame)
M_r_sdss = t["ABSMAG01_SDSS_R"]
Msun_r = 4.64
log10_L_r = -0.4 * (M_r_sdss - Msun_r)

# 5) Final color-based stellar mass
logM_color = log10_ML_r + log10_L_r

# Summary stats (finite only)
m = np.asarray(logM_color, dtype=float)
good = np.isfinite(m)
m = m[good]

print("N total:", len(logM_color))
print("N finite:", len(m))
print("logM* (color) min/max:", float(np.min(m)), float(np.max(m)))
print("logM* (color) median:", float(np.median(m)))
print("logM* (color) 16/84 percentiles:", tuple(map(float, np.percentile(m, [16, 84]))))

# Optional: also summarize the colors and M/L
c = np.asarray(gmr_sdss_equiv, dtype=float)
c = c[np.isfinite(c)]
print("g-r (SDSS-equiv) min/max:", float(np.min(c)), float(np.max(c)))
print("g-r (SDSS-equiv) median:", float(np.median(c)))
