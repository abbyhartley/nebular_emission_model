"""
Measure the intrinsic scatter of the (g-r) -> log10(M/L_r) color relation used in
src/normflow/stellar_mass.py:  log10(M/L_r) = 1.062*(g-r)_SDSS - 0.555

Calibration sample: SDSS galaxies in mpa_rcsed2_combo.fits, which carry BOTH
  - MPA-JHU SED-fit total stellar mass LGM_TOT_P50  ("truth")
  - rest-frame SDSS g,r photometry (corrmag_*, kcorr_*)

Method: the color relation predicts log10(M/L_r); the MPA-JHU "truth" log10(M/L_r)
is LGM_TOT_P50 - log10(L_r).  Their residual isolates the M/L scatter because the
common log10(L_r) term cancels.  A constant zero-point offset (h, IMF, aperture)
shifts the median but not the scatter, so std/NMAD of the residual is the relation
dispersion.  We report it raw and after removing the median offset, and also
re-fit the relation freshly to confirm.
"""
from pathlib import Path
import sys
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
sys.path.insert(0, str(REPO / "src"))
from normflow.stellar_mass import log10_ml_r_from_gmr_sdss

MSUN_R = 4.64
INFILE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo.fits"

t = Table.read(INFILE, hdu=1)
N0 = len(t)

z   = np.asarray(t["Z_1"], float)
mg  = np.asarray(t["corrmag_g"], float)
mr  = np.asarray(t["corrmag_r"], float)
kg  = np.asarray(t["kcorr_g"], float)
kr  = np.asarray(t["kcorr_r"], float)
lgm = np.asarray(t["LGM_TOT_P50"], float)          # MPA-JHU SED-fit total mass
rel = np.asarray(t["RELIABLE"], int)
typ = np.asarray(t["SPECTROTYPE"]).astype(str)

DM  = cosmo.distmod(np.clip(z, 1e-4, None)).value
Mg  = mg - DM - kg
Mr  = mr - DM - kr
gmr = Mg - Mr                                       # rest-frame SDSS g-r

log10_Lr     = -0.4 * (Mr - MSUN_R)
ml_pred      = np.asarray(log10_ml_r_from_gmr_sdss(gmr), float)   # relation
logM_color   = ml_pred + log10_Lr                   # color-based stellar mass
ml_true      = lgm - log10_Lr                        # MPA-JHU implied log(M/L_r)

good = (
    np.isfinite(z) & (z > 0.0)
    & np.isfinite(mg) & np.isfinite(mr) & np.isfinite(kg) & np.isfinite(kr)
    & np.isfinite(lgm) & (lgm > 6) & (lgm < 13)
    & (rel == 1)
    & (typ == "GALAXY")
    & np.isfinite(gmr) & (gmr > -0.5) & (gmr < 2.0)
)

def stats(x):
    x = x[np.isfinite(x)]
    med = np.median(x)
    nmad = 1.4826 * np.median(np.abs(x - med))
    return dict(n=len(x), median=float(med), mean=float(np.mean(x)),
                std=float(np.std(x)), nmad=float(nmad),
                p16=float(np.percentile(x, 16)), p84=float(np.percentile(x, 84)))

# Residual in log(M/L_r) == residual in logM* (L_r cancels)
res = (ml_true - ml_pred)[good]              # = LGM_TOT_P50 - logM_color
s_raw = stats(res)
s_dezp = stats(res - s_raw["median"])        # after removing constant zero-point offset

# Fresh re-fit of log(M/L_r) vs (g-r) on the same sample (sanity check)
c = gmr[good]
y = ml_true[good]
A = np.vstack([c, np.ones_like(c)]).T
slope, intcpt = np.linalg.lstsq(A, y, rcond=None)[0]
resid_fit = y - (slope * c + intcpt)
s_fit = stats(resid_fit)

print(f"Input rows                 : {N0:,}")
print(f"Calibration sample (good)  : {good.sum():,}")
print(f"median rest-frame g-r      : {np.median(c):.3f}  (16/84: {np.percentile(c,16):.3f}/{np.percentile(c,84):.3f})")
print()
print("=== Residual log(M/L_r): MPA-JHU truth  -  color relation (1.062*(g-r)-0.555) ===")
print(f"  median offset (zero-pt)  : {s_raw['median']:+.3f} dex   [h/IMF/aperture; not scatter]")
print(f"  scatter  std             : {s_raw['std']:.3f} dex")
print(f"  scatter  NMAD            : {s_raw['nmad']:.3f} dex   <-- robust dispersion of the relation")
print(f"  16/84 about median       : [{s_dezp['p16']:+.3f}, {s_dezp['p84']:+.3f}] dex")
print()
print("=== Fresh least-squares re-fit of log(M/L_r) vs (g-r) on this sample ===")
print(f"  this-sample best fit     : log(M/L_r) = {slope:.3f}*(g-r) {intcpt:+.3f}")
print(f"  in-code relation         : log(M/L_r) = 1.062*(g-r) -0.555")
print(f"  scatter about fresh fit  : std {s_fit['std']:.3f} dex,  NMAD {s_fit['nmad']:.3f} dex")
print()
# Scatter vs color (is it homoscedastic?)
print("=== NMAD of residual in (g-r) color bins ===")
edges = np.array([0.2, 0.5, 0.65, 0.8, 0.95, 1.1, 1.4])
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (c >= lo) & (c < hi)
    if sel.sum() > 50:
        r = res[sel]; r = r[np.isfinite(r)]
        nm = 1.4826 * np.median(np.abs(r - np.median(r)))
        print(f"  {lo:.2f}-{hi:.2f}  n={sel.sum():7d}  NMAD={nm:.3f} dex")
