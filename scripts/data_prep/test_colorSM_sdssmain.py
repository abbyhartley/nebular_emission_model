from pathlib import Path
import sys
import numpy as np
from astropy.table import Table

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
sys.path.insert(0, str(REPO / "src"))

# SDSS-color -> M/L calibration we wrote
from normflow.stellar_mass import log10_ml_r_from_gmr_sdss


infile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo_selected_test.fits")
t = Table.read(infile, hdu=1)

# ------------------------------------------------------------
# SDSS MAIN: use SDSS rest-frame mags directly from this file
# Your file has dust-corrected magnitudes corrmag_g/corrmag_r and k-corrections kcorr_g/kcorr_r.
# A common convention is: M = m_corr - DM(z) - K(z)  (K in magnitudes)
# ------------------------------------------------------------

# Choose redshift column
z = np.asarray(t["Z_1"], dtype=float)

# Apparent mags (already extinction-corrected in this catalog)
mg = np.asarray(t["corrmag_g"], dtype=float)
mr = np.asarray(t["corrmag_r"], dtype=float)

# K-corrections (magnitudes)
kg = np.asarray(t["kcorr_g"], dtype=float)
kr = np.asarray(t["kcorr_r"], dtype=float)

# Distance modulus from redshift (Planck15)
from astropy.cosmology import Planck15 as cosmo
DM = np.asarray(cosmo.distmod(z).value, dtype=float)

# Rest-frame absolute magnitudes in SDSS g,r
# (If your kcorr_* are defined with the opposite sign in this catalog, swap the sign here.
#  But for most k-corr conventions: M = m - DM - K.)
Mg = mg - DM - kg
Mr = mr - DM - kr

# 1) Rest-frame SDSS g-r color
gmr_sdss_rest = Mg - Mr

# 2) M/L relation (expects SDSS g-r)
log10_ML_r = np.asarray(log10_ml_r_from_gmr_sdss(gmr_sdss_rest), dtype=float)

# 3) r-band luminosity in solar units
Msun_r = 4.64
log10_L_r = -0.4 * (Mr - Msun_r)

# 4) Color-based stellar mass
logM_color = log10_ML_r + log10_L_r

# ------------------------------------------------------------
# Summary stats
# ------------------------------------------------------------
m = np.asarray(logM_color, dtype=float)
good = np.isfinite(m)
m = m[good]

print("N total:", len(logM_color))
print("N finite:", len(m))
print("logM* (color) min/max:", float(np.min(m)), float(np.max(m)))
print("logM* (color) median:", float(np.median(m)))
print("logM* (color) 16/84 percentiles:", tuple(map(float, np.percentile(m, [16, 84]))))

c = np.asarray(gmr_sdss_rest, dtype=float)
c = c[np.isfinite(c)]
print("g-r (SDSS rest) min/max:", float(np.min(c)), float(np.max(c)))
print("g-r (SDSS rest) median:", float(np.median(c)))
