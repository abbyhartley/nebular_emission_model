"""
Per-line DESI->SDSS flux calibration offsets on the same-galaxy cross-match.
c_line = median(log10(F_sdss / F_desi_ORIG))  [dex to ADD to raw DESI to reach SDSS]
Uses DESI *_ORIG (raw FastSpecFit) fluxes. Reports differential vs Halpha
(= how much each line RATIO to Ha changes), which the cross-match said is <=0.03 dex.
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
# (name, sdss_col, desi_ORIG_col, sdss_err, desi_ivar)
LINES = [
    ("Halpha",   "H_ALPHA_FLUX",  "HALPHA_FLUX_ORIG",   "H_ALPHA_FLUX_ERR",  "HALPHA_FLUX_IVAR"),
    ("Hbeta",    "H_BETA_FLUX",   "HBETA_FLUX_ORIG",    "H_BETA_FLUX_ERR",   "HBETA_FLUX_IVAR"),
    ("Hgamma",   "H_GAMMA_FLUX",  "HGAMMA_FLUX_ORIG",   "H_GAMMA_FLUX_ERR",  "HGAMMA_FLUX_IVAR"),
    ("NII6584",  "NII_6584_FLUX", "NII_6584_FLUX_ORIG", "NII_6584_FLUX_ERR", "NII_6584_FLUX_IVAR"),
    ("SII6717",  "SII_6717_FLUX", "SII_6716_FLUX_ORIG", "SII_6717_FLUX_ERR", "SII_6716_FLUX_IVAR"),
    ("SII6731",  "SII_6731_FLUX", "SII_6731_FLUX_ORIG", "SII_6731_FLUX_ERR", "SII_6731_FLUX_IVAR"),
    ("OII3726",  "OII_3726_FLUX", "OII_3726_FLUX_ORIG", "OII_3726_FLUX_ERR", "OII_3726_FLUX_IVAR"),
    ("OII3729",  "OII_3729_FLUX", "OII_3729_FLUX_ORIG", "OII_3729_FLUX_ERR", "OII_3729_FLUX_IVAR"),
    ("OIII5007", "OIII_5007_FLUX","OIII_5007_FLUX_ORIG","OIII_5007_FLUX_ERR","OIII_5007_FLUX_IVAR"),
]

def load(p):
    t = Table.read(p, hdu=1)
    return t[[n for n in t.colnames if len(t[n].shape) <= 1]]

s = load(BASE + "SDSS_main_training_data.fits")
d = load(BASE + "DESI_BGS_training_data.fits")
cs = SkyCoord(np.asarray(s["RA_1"], float) * u.deg, np.asarray(s["DEC_1"], float) * u.deg)
cd = SkyCoord(np.asarray(d["TARGET_RA"], float) * u.deg, np.asarray(d["TARGET_DEC"], float) * u.deg)
idx, sep, _ = cs.match_to_catalog_sky(cd)
zs = np.asarray(s["Z_1"], float); zd = np.asarray(d["Z"], float)
m = (sep.arcsec < 1.5) & (np.abs(zs - zd[idx]) < 0.001)
si = np.where(m)[0]; di = idx[m]
print(f"matched N = {len(si)}\n")
print(f"{'line':9s} {'c_line (SDSS-DESIraw)':>22s} {'diff vs Ha':>12s}   N")
offs = {}
c_ha = None
for name, sc, dc, se, div in LINES:
    fs = np.asarray(s[sc], float)[si]
    fd = np.asarray(d[dc], float)[di]
    iv = np.asarray(d[div], float)[di]
    ee = np.asarray(s[se], float)[si]
    g = np.isfinite(fs) & (fs > 0) & np.isfinite(fd) & (fd > 0) & (iv > 0) & (ee > 0) & (fs/ee > 5) & (fd*np.sqrt(iv) > 5)
    c = float(np.median(np.log10(fs[g] / fd[g])))
    offs[name] = c
    if name == "Halpha":
        c_ha = c
    print(f"{name:9s} {c:>+22.4f} {c-c_ha if c_ha is not None else 0:>+12.4f}   {int(g.sum())}")
print("\ndict form:", {k: round(v,4) for k,v in offs.items()})
