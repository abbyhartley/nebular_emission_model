#!/usr/bin/env python3
"""
Selection-robustness check for the high-z validation.
Re-derives the SAME Tier-1/Tier-2 samples as hiz_compute.py (deterministic idx), aligns them to the
cached predictions, and recomputes metrics under progressively stricter, training-matched selection:

  V0  compared-lines only        : per-line S/N>3 on Ha + the target line (current headline)
  V1  all observable lines >3    : Hb,[OII]3726,[OII]3729,[OIII]5007 (+Ha) all S/N>3 simultaneously
                                   (strict per-line analog on the lines DESI-COSMOS actually has)
  V2  V1 + continuum proxy        : + TSNR2_BGS above a threshold (emulates continuum-S/N>3)

No continuum-S/N column exists in DESI-COSMOS-v2.0, so TSNR2_BGS is the closest proxy; we test a
range of thresholds. If bias/scatter/rho are stable across V0->V2 the result is selection-robust.
"""
import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
FLUX_SCALE = 1e-17; MASS_ZP = 0.13; R = 3.90
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
NAMES = ["Hbeta", "OII3726", "OII3729", "OIII5007"]


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def decode(a):
    a = np.asarray(a)
    if a.dtype.kind == "S":
        a = np.char.decode(a)
    return np.char.strip(a.astype(str))


def st(pred, obs):
    r = pred - obs; g = np.isfinite(r); rr = r[g]
    return (int(g.sum()), float(np.median(rr)),
            float(0.5 * (np.percentile(rr, 84) - np.percentile(rr, 16))),
            float(np.sqrt(np.mean(rr**2))), float(spearmanr(pred[g], obs[g]).correlation))


h = fits.open(COSMOS); d1, d6 = h[1].data, h[6].data
z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
stype = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float)
tsnr = np.asarray(d1["TSNR2_BGS"], float)


def flux(n):
    f = np.asarray(d1[n + "_FLUX"], float) * 10.0 ** CAL[n]
    sn = np.asarray(d1[n + "_FLUX"], float) * np.sqrt(np.clip(np.asarray(d1[n + "_FLUX_IVAR"], float), 0, None))
    return f, sn


Fha, SNha = flux("HALPHA"); Fhb, SNhb = flux("HBETA")
Foa, SNoa = flux("OII_3726"); Fob, SNob = flux("OII_3729"); Fo3, SNo3 = flux("OIII_5007")
base = (zwarn == 0) & (stype == "GALAXY") & np.isfinite(lpm) & (lpm > 6) & (lpm < 13) & (z > 0.05)

A = np.load(REPO + "figs_ALTB/hiz_arrays.npz", allow_pickle=True)

# ---------- Tier 1 ----------
t1 = base & (z < 0.49) & (SNha > 3) & (Fha > 0)
idx = np.where(t1)[0]
assert np.allclose(A["t1_z"], z[idx]), "Tier-1 idx mismatch vs cached arrays!"
obs = A["t1_obs"]; sn = A["t1_sn"]; pm = A["t1_predmean"]; inbox = A["t1_inbox"]
tsnr1 = tsnr[idx]
all4 = np.all(sn > 3, axis=1)
print("=== TIER 1 selection robustness (in-box throughout) ===")
print("N in-box: %d | all-4-lines>3 & in-box: %d" % (inbox.sum(), (all4 & inbox).sum()))
thr_med = np.median(tsnr1[all4 & inbox])
variants = [("V0 compared-lines", lambda c: (sn[:, c] > 3) & inbox),
            ("V1 all-lines>3", lambda c: all4 & inbox),
            ("V2 +TSNR2>1000", lambda c: all4 & inbox & (tsnr1 > 1000)),
            ("V2 +TSNR2>median", lambda c: all4 & inbox & (tsnr1 > thr_med))]
for vname, msk in variants:
    row = []
    for c in range(4):
        m = msk(c) & np.isfinite(obs[:, c])
        N, b, s, rm, rho = st(pm[m, c], obs[m, c])
        row.append((NAMES[c], N, b, s, rm, rho))
    print("\n[%s]  (TSNR2 median=%.0f)" % (vname, thr_med))
    for name, N, b, s, rm, rho in row:
        print("   %-9s N=%5d bias=%+.3f scat=%.3f rmse=%.3f rho=%.3f" % (name, N, b, s, rm, rho))

# ---------- Tier 2 ----------
t2 = base & (z >= 0.49) & (z < 1.00) & (SNhb > 3) & (SNo3 > 3) & (Fhb > 0) & (Fo3 > 0)
idx2 = np.where(t2)[0]
assert np.allclose(A["t2_z"], z[idx2]), "Tier-2 idx mismatch!"
o = A["t2_obs_o3hb"]; pmm = A["t2_pred_o3hb_mean"]; ib2 = A["t2_inbox"]
tsnr2 = tsnr[idx2]
print("\n=== TIER 2 [OIII]/Hb selection robustness (in-box) ===")
for vname, m in [("V0 (Hb,OIII>3)", ib2),
                 ("V2 +TSNR2>1000", ib2 & (tsnr2 > 1000)),
                 ("V2 +TSNR2>median", ib2 & (tsnr2 > np.median(tsnr2[ib2])))]:
    mm = m & np.isfinite(o)
    N, b, s, rm, rho = st(pmm[mm], o[mm])
    print("   %-18s N=%5d bias=%+.3f scat=%.3f rmse=%.3f rho=%.3f" % (vname, N, b, s, rm, rho))
print("=== DONE ===")
