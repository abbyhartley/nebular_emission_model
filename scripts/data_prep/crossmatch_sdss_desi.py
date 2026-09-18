"""
Phase 0: cross-match the SDSS and DESI SELECTED training samples by sky position
+ redshift -> the "same galaxy, two spectra" sample. Save both surveys' 9 line
fluxes (8 targets + Ha) for the matched galaxies for the flux-comparison analysis.
"""
from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = Path(BASE + "nebular_emission_model")
SEP_MAX = 1.5     # arcsec
DZ_MAX = 0.001

# 9 lines in a common order; (sdss_col, desi_col)
LINES = [
    ("Hbeta",    "H_BETA_FLUX",   "HBETA_FLUX"),
    ("Hgamma",   "H_GAMMA_FLUX",  "HGAMMA_FLUX"),
    ("NII6584",  "NII_6584_FLUX", "NII_6584_FLUX"),
    ("SII6717",  "SII_6717_FLUX", "SII_6716_FLUX"),
    ("SII6731",  "SII_6731_FLUX", "SII_6731_FLUX"),
    ("OII3726",  "OII_3726_FLUX", "OII_3726_FLUX"),
    ("OII3729",  "OII_3729_FLUX", "OII_3729_FLUX"),
    ("OIII5007", "OIII_5007_FLUX", "OIII_5007_FLUX"),
    ("Halpha",   "H_ALPHA_FLUX",  "HALPHA_FLUX"),
]


def load(p):
    t = Table.read(p, hdu=1)
    return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()


s = load(BASE + "SDSS_main_training_data.fits")
d = load(BASE + "DESI_BGS_training_data.fits")
print("SDSS", len(s), " DESI", len(d))

ra_s, dec_s, z_s = s["RA_1"].to_numpy(float), s["DEC_1"].to_numpy(float), s["Z_1"].to_numpy(float)
ra_d, dec_d, z_d = d["TARGET_RA"].to_numpy(float), d["TARGET_DEC"].to_numpy(float), d["Z"].to_numpy(float)

cs = SkyCoord(ra_s * u.deg, dec_s * u.deg)
cd = SkyCoord(ra_d * u.deg, dec_d * u.deg)
idx, sep, _ = cs.match_to_catalog_sky(cd)         # nearest DESI for each SDSS
sep_as = sep.arcsec
dz = np.abs(z_s - z_d[idx])
m = (sep_as < SEP_MAX) & (dz < DZ_MAX) & np.isfinite(z_s) & np.isfinite(z_d[idx])
print(f"\nmatched (sep<{SEP_MAX}\" & |dz|<{DZ_MAX}): {int(m.sum())}")
print(f"  median sep = {np.median(sep_as[m]):.3f}\"  ;  median |dz| = {np.median(dz[m]):.2e}")
print(f"  z_sdss median {np.median(z_s[m]):.3f} ; z_desi median {np.median(z_d[idx][m]):.3f}")

si = np.where(m)[0]
di = idx[m]
out = {"sep_arcsec": sep_as[m], "z_sdss": z_s[m], "z_desi": z_d[di]}
for name, sc, dc in LINES:
    out[f"sdss_{name}"] = s[sc].to_numpy(float)[si]
    out[f"desi_{name}"] = d[dc].to_numpy(float)[di]
# carry stellar masses too
out["logm_sdss"] = s["LOGM_COLOR"].to_numpy(float)[si]
out["logm_desi"] = d["LOGM_COLOR"].to_numpy(float)[di]
df = pd.DataFrame(out)
outp = REPO / "docs" / "crossmatch_sdss_desi_fluxes.csv"
df.to_csv(outp, index=False)
print("Wrote", outp, " rows:", len(df))

# quick per-line median flux ratio DESI/SDSS (both fluxes>0)
print("\n=== median log10(DESI/SDSS) flux per line (matched, both>0) ===")
for name, _, _ in LINES:
    a = df[f"desi_{name}"].to_numpy(); b = df[f"sdss_{name}"].to_numpy()
    g = (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)
    r = np.log10(a[g] / b[g])
    print(f"  {name:9s} n={g.sum():5d}  median={np.median(r):+.3f}  NMAD={1.4826*np.median(np.abs(r-np.median(r))):.3f}")
