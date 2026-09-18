# add_logmcolor_to_desi_sdsslike.py
from pathlib import Path
import sys
import numpy as np
from astropy.table import Table

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
sys.path.insert(0, str(REPO / "src"))

from normflow.stellar_mass import desi_to_sdss_gmr, log10_ml_r_from_gmr_sdss

infile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/desi_bgs_sdss_like_rlt17p77.fits")
outfile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/desi_bgs_sdss_like_rlt17p77_with_LOGM_COLOR.fits")

t = Table.read(infile, hdu=1)

# Compute DESI g-r (DECam, bandshifted) and convert to SDSS-equiv g-r
gmr_decam = np.asarray(t["ABSMAG10_DECAM_G"], dtype=float) - np.asarray(t["ABSMAG10_DECAM_R"], dtype=float)
gmr_sdss = np.asarray(desi_to_sdss_gmr(gmr_decam), dtype=float)

# M/L from SDSS-equivalent color
logMLr = np.asarray(log10_ml_r_from_gmr_sdss(gmr_sdss), dtype=float)

# r-band luminosity from SDSS absolute magnitude (as used before)
Msun_r = 4.64
Mr_sdss = np.asarray(t["ABSMAG01_SDSS_R"], dtype=float)
logLr = -0.4 * (Mr_sdss - Msun_r)

logM_color = logMLr + logLr
t["LOGM_COLOR"] = logM_color.astype(np.float32)

# Optional: clip extreme outliers like you did for the training set
m = np.asarray(t["LOGM_COLOR"], dtype=float)
finite = np.isfinite(m)
p_lo, p_hi = np.percentile(m[finite], [0.1, 99.9])
keep = finite & (m >= p_lo) & (m <= p_hi)
t = t[keep]

t.write(outfile, overwrite=True)
print("Wrote:", outfile)
print("N kept:", len(t))
print("LOGM_COLOR p0.1/p50/p99.9:", np.percentile(np.asarray(t["LOGM_COLOR"], float), [0.1, 50, 99.9]))
