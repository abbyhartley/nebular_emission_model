"""
Phase 3: localize the DESI-SDSS color-mass offset (~-0.68 dex) by decomposing
logM = [log M/L term] + [log L_r term] for the SAME (cross-matched) galaxies.

SDSS: Mr = corrmag_r - DM(z) - kcorr_r ; gmr = (g-r)corr - (kg-kr)
DESI: color from ABSMAG10_DECAM_{G,R} -> SDSS-equiv ; Mr = ABSMAG01_SDSS_R
Both: logML = 1.062*gmr - 0.555 ; logL = -0.4*(Mr - 4.64)
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
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
z_s = s["Z_1"].to_numpy(float)
z_d = d["Z"].to_numpy(float)
m = (sep.arcsec < 1.5) & (np.abs(z_s - z_d[idx]) < 0.001)
si = np.where(m)[0]; di = idx[m]
print("matched:", len(si))

# --- SDSS terms ---
zs = z_s[si]
mg, mr = s["corrmag_g"].to_numpy(float)[si], s["corrmag_r"].to_numpy(float)[si]
kg, kr = s["kcorr_g"].to_numpy(float)[si], s["kcorr_r"].to_numpy(float)[si]
DM = cosmo.distmod(zs).value
Mr_s = mr - DM - kr
gmr_s = (mg - kg) - (mr - kr)
ML_s = 1.062 * gmr_s - 0.555
L_s = -0.4 * (Mr_s - Msun_r)
logM_s = ML_s + L_s

# --- DESI terms ---
gA, rA = d["ABSMAG10_DECAM_G"].to_numpy(float)[di], d["ABSMAG10_DECAM_R"].to_numpy(float)[di]
gmr_decam = gA - rA
gmr_d = 0.989 * gmr_decam - 0.104
Mr_d = d["ABSMAG01_SDSS_R"].to_numpy(float)[di]
ML_d = 1.062 * gmr_d - 0.555
L_d = -0.4 * (Mr_d - Msun_r)
logM_d = ML_d + L_d

# stored
logM_s_stored = s["LOGM_COLOR"].to_numpy(float)[si]
logM_d_stored = d["LOGM_COLOR"].to_numpy(float)[di]

g = np.isfinite(logM_s) & np.isfinite(logM_d) & np.isfinite(gmr_s) & np.isfinite(gmr_d)


def med(x):
    return float(np.median(x[g]))


print("\n=== sanity: recomputed vs stored LOGM_COLOR (median |diff|) ===")
print(f"  SDSS: {np.median(np.abs(logM_s - logM_s_stored)[g]):.3f}   DESI: {np.median(np.abs(logM_d - logM_d_stored)[g]):.3f}")

print("\n=== median values (matched) ===")
print(f"  M_r   : SDSS {med(Mr_s):+.3f}   DESI {med(Mr_d):+.3f}   (DESI-SDSS = {med(Mr_d-Mr_s):+.3f} mag)")
print(f"  (g-r) : SDSS {med(gmr_s):+.3f}   DESI {med(gmr_d):+.3f}   (DESI-SDSS = {med(gmr_d-gmr_s):+.3f} mag)")
print("\n=== term decomposition of logM (DESI - SDSS) ===")
print(f"  log(M/L) term : {med(ML_d - ML_s):+.3f} dex")
print(f"  log(L_r) term : {med(L_d - L_s):+.3f} dex   <-- from M_r")
print(f"  TOTAL logM    : {med(logM_d - logM_s):+.3f} dex")
print("\n(reminder: DESI M_r uses ABSMAG01_SDSS_R [band-shift 0.1];")
print(" SDSS M_r uses corrmag_r - Planck15 DM - kcorr_r [rest-frame].)")
