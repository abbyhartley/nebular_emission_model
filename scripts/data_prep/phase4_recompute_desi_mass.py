"""
Phase 4: recompute DESI color-mass on a SELF-CONSISTENT SDSS-filter basis and
calibrate to the SDSS/MPA-JHU scale using the cross-matched overlap.

Old DESI mass: color from ABSMAG10_DECAM (band-shift 1.0, too blue) + M_r from
ABSMAG01_SDSS_R (h/band-shift offset). New: use ABSMAG01_SDSS_{G,R} for BOTH
(same band-shift 0.1, same SDSS filters, no DECam transform), then a zero-point
Delta from the overlap puts M_r on the SDSS distance/h scale.
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
Msun_r = 4.64


def load(p):
    t = Table.read(p, hdu=1)
    return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()


s = load(BASE + "SDSS_main_training_data.fits")
d = load(BASE + "DESI_BGS_training_data.fits")
cs = SkyCoord(s["RA_1"].to_numpy(float) * u.deg, s["DEC_1"].to_numpy(float) * u.deg)
cd = SkyCoord(d["TARGET_RA"].to_numpy(float) * u.deg, d["TARGET_DEC"].to_numpy(float) * u.deg)
idx, sep, _ = cs.match_to_catalog_sky(cd)
z_s = s["Z_1"].to_numpy(float); z_d = d["Z"].to_numpy(float)
m = (sep.arcsec < 1.5) & (np.abs(z_s - z_d[idx]) < 0.001)
si = np.where(m)[0]; di = idx[m]

logM_sdss = s["LOGM_COLOR"].to_numpy(float)[si]
logM_desi_old = d["LOGM_COLOR"].to_numpy(float)[di]

g01 = d["ABSMAG01_SDSS_G"].to_numpy(float)[di]
r01 = d["ABSMAG01_SDSS_R"].to_numpy(float)[di]
gmr01 = g01 - r01
logM_v2 = (1.062 * gmr01 - 0.555) + (-0.4 * (r01 - Msun_r))   # self-consistent SDSS-filter, uncalibrated

good = np.isfinite(logM_sdss) & np.isfinite(logM_v2) & np.isfinite(logM_desi_old)
def med(x): return float(np.median(x[good]))
def nmad(x): x=x[good]; return float(1.4826*np.median(np.abs(x-np.median(x))))

print(f"matched N={good.sum()}")
print(f"\n(g-r) SDSS-band01 median = {med(gmr01):.3f}   (old DECam-based equiv was +0.157; SDSS rest-frame +0.449)")
print("\n=== same-galaxy offset (SDSS - DESI) ===")
print(f"  OLD  logM_sdss - logM_desi_old = {med(logM_sdss - logM_desi_old):+.3f} dex  (NMAD {nmad(logM_sdss-logM_desi_old):.3f})")
print(f"  v2   logM_sdss - logM_v2       = {med(logM_sdss - logM_v2):+.3f} dex  (NMAD {nmad(logM_sdss-logM_v2):.3f})")

# is the v2 residual a constant zero-point or does it slope with mass?
res = (logM_sdss - logM_v2)[good]
xm = logM_sdss[good]
slope, intcpt = np.polyfit(xm, res, 1)
Delta = np.median(res)
print(f"\n  v2 residual vs logM_sdss: slope={slope:+.3f}, intercept={intcpt:+.3f}; median Delta={Delta:+.3f}")

# apply constant Delta -> final calibrated DESI mass on the overlap
logM_final = logM_v2 + Delta
print(f"\n=== after constant zero-point calibration (Delta={Delta:+.3f}) ===")
print(f"  logM_sdss - logM_final = {med(logM_sdss - logM_final):+.3f} dex  (NMAD {nmad(logM_sdss-logM_final):.3f})")
print(f"  per-galaxy agreement scatter (NMAD) = {nmad(logM_final - logM_sdss):.3f} dex")
