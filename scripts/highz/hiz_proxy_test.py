#!/usr/bin/env python3
"""
Which line best predicts OBSERVED L_Ha at z>0.5 (where Ha leaves the DESI window)?
Test on DESI-COSMOS z<0.49 galaxies where Ha, Hb, [OII], [OIII] are ALL observed, so we
know the true L_Ha and can measure how well each proxy recovers it.

Proxies compared (all on observed/fiber flux scale, calibrated):
  - Hb + fixed decrement R=3.9
  - Hb + mass-dependent decrement R(M*)
  - Hb + M* (multilinear fit)
  - [OII]3727 (linear fit)         + [OII] + M*
  - [OIII]5007 (linear fit)        + [OIII] + M*
Report residual scatter (NMAD & 16-84 half-width), bias, Spearman rho.
Also the raw Balmer-decrement scatter (the floor for the Hb proxy).
"""
import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr

COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
FLUX_SCALE = 1e-17; MASS_ZP = 0.13
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def decode(a):
    a = np.asarray(a)
    if a.dtype.kind == "S":
        a = np.char.decode(a)
    return np.char.strip(a.astype(str))


def scat(res):
    res = res[np.isfinite(res)]
    return (1.4826 * np.median(np.abs(res - np.median(res))),
            0.5 * (np.percentile(res, 84) - np.percentile(res, 16)))


h = fits.open(COSMOS); d1, d6 = h[1].data, h[6].data
z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
stype = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float)


def cf(name):
    raw = np.asarray(d1[name + "_FLUX"], float); iv = np.asarray(d1[name + "_FLUX_IVAR"], float)
    f = raw * 10 ** CAL[name]; sn = raw * np.sqrt(np.clip(iv, 0, None)); return f, sn


Fha, Sha = cf("HALPHA"); Fhb, Shb = cf("HBETA"); Foa, Soa = cf("OII_3726"); Fob, Sob = cf("OII_3729"); Fo3, So3 = cf("OIII_5007")
Foii = Foa + Fob
snoii = (Foa + Fob) / np.sqrt((Foa / np.clip(Soa, 1e-9, None)) ** -2 * 0 + 1) if False else None
# common sample: all lines detected
m = ((zwarn == 0) & (stype == "GALAXY") & (z > 0.05) & (z < 0.49) & np.isfinite(lpm) & (lpm > 6) & (lpm < 13)
     & (Sha > 3) & (Shb > 3) & (Soa > 3) & (Sob > 3) & (So3 > 3) & (Fha > 0) & (Fhb > 0) & (Foii > 0) & (Fo3 > 0))
z, lpm = z[m], lpm[m]
logm = lpm + MASS_ZP
Lha = log10_lum(z, Fha[m]); Lhb = log10_lum(z, Fhb[m]); Loii = log10_lum(z, Foii[m]); Lo3 = log10_lum(z, Fo3[m])
print("common all-line sample N=%d  (z 0.05-0.49)" % len(z))

# raw Balmer decrement (log Ha - log Hb)  = log R
dec = Lha - Lhb
print("\nBalmer decrement log10(Ha/Hb): median=%.3f (R=%.2f)  scatter NMAD=%.3f  16-84/2=%.3f"
      % (np.median(dec), 10 ** np.median(dec), *scat(dec)))
print("  => this is the irreducible floor of the Hb->L_Ha proxy\n")


def report(name, pred):
    res = pred - Lha; nmad, hw = scat(res)
    print("  %-26s bias=%+.3f  NMAD=%.3f  16-84/2=%.3f  rho=%.3f"
          % (name, np.median(res), nmad, hw, spearmanr(pred, Lha).correlation))


def linfit(X, y):
    A = np.column_stack([X, np.ones(len(y))]) if X.ndim == 1 else np.column_stack([X, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef, coef


print("PROXY PERFORMANCE (predicting observed log L_Ha):")
# Hb fixed decrement
report("Hb + fixed R=3.9", Lhb + np.log10(3.9))
# Hb + mass-dependent decrement R(M*)
mb = np.linspace(logm.min(), logm.max(), 9)
Rz = np.full(len(z), np.nan)
for a, b in zip(mb[:-1], mb[1:]):
    s = (logm >= a) & (logm < b)
    if s.sum() > 20:
        Rz[s] = np.median(dec[s])
# fill ends
Rz[~np.isfinite(Rz)] = np.median(dec)
report("Hb + mass-dep decrement", Lhb + Rz)
# Hb + M* multilinear
p, _ = linfit(np.column_stack([Lhb, logm]), Lha); report("Hb + M* (fit)", p)
# Hb alone (fit slope)
p, _ = linfit(Lhb, Lha); report("Hb alone (fit)", p)
# [OII]
p, _ = linfit(Loii, Lha); report("[OII]3727 alone (fit)", p)
p, _ = linfit(np.column_stack([Loii, logm]), Lha); report("[OII]3727 + M* (fit)", p)
# [OIII]
p, _ = linfit(Lo3, Lha); report("[OIII]5007 alone (fit)", p)
p, _ = linfit(np.column_stack([Lo3, logm]), Lha); report("[OIII]5007 + M* (fit)", p)
# [OII]+[OIII]+Hb combined (upper bound)
p, _ = linfit(np.column_stack([Lhb, Loii, Lo3, logm]), Lha); report("Hb+[OII]+[OIII]+M* (fit)", p)
print("=== DONE ===")
