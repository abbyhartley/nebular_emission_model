# plot_delta_logFha_sdss_minus_desi_vs_logFha_desi.py
#
# Makes a diagnostic plot on the SDSS–DESI matched sample:
#   Δ = log10(F_Ha_SDSS) - log10(F_Ha_DESI)
# vs
#   log10(F_Ha_DESI)
#
# Also overlays binned medians + 16/84th percentiles to reveal trends.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u


# -----------------------
# Inputs / parameters
# -----------------------
SDSS_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo.fits")
DESI_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/fastspec-iron-main-bright.fits")

MATCH_RADIUS = 1.0 * u.arcsec
DZ_MAX = 5e-4

N_DESI = 1_000_000   # increase to get more matches
SEED = 2

SDSS_RA_CANDS = ["RA_1", "RA"]
SDSS_DEC_CANDS = ["DEC_1", "DEC"]
SDSS_Z_CANDS = ["Z_1", "Z"]
SDSS_HA = "H_ALPHA_FLUX"

DESI_RA = "RA"
DESI_DEC = "DEC"
DESI_Z = "Z"
DESI_HA = "HALPHA_FLUX"
DESI_HA_IVAR = "HALPHA_FLUX_IVAR"  # optional


# -----------------------
# Helpers
# -----------------------
def pick_col(tab, candidates):
    for c in candidates:
        if c in tab.colnames:
            return c
    raise KeyError(f"None of these columns found: {candidates}")

def thin_indices(n, n_keep, seed=0):
    rng = np.random.default_rng(seed)
    n_keep = min(n_keep, n)
    return rng.choice(n, size=n_keep, replace=False)

def safe_log10(x):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x) & (x > 0)
    out[m] = np.log10(x[m])
    return out


# -----------------------
# Load SDSS
# -----------------------
print("Loading SDSS:", SDSS_FITS)
sdss = Table.read(SDSS_FITS, hdu=1)

sdss_ra = pick_col(sdss, SDSS_RA_CANDS)
sdss_dec = pick_col(sdss, SDSS_DEC_CANDS)
sdss_z = pick_col(sdss, SDSS_Z_CANDS)

mask_sdss = np.isfinite(sdss[sdss_ra]) & np.isfinite(sdss[sdss_dec]) & np.isfinite(sdss[sdss_z])
mask_sdss &= np.isfinite(sdss[SDSS_HA]) & (sdss[SDSS_HA] > 0)

sdss = sdss[mask_sdss]
print("SDSS usable rows:", len(sdss))


# -----------------------
# Load DESI (fast + meta)
# -----------------------
print("Loading DESI:", DESI_FITS)
hdul = fits.open(DESI_FITS, memmap=True)
fast = hdul[1].data
meta = hdul[2].data

ra_all = meta[DESI_RA].astype(np.float64)
dec_all = meta[DESI_DEC].astype(np.float64)
z_all = fast[DESI_Z].astype(np.float64)
ha_all = fast[DESI_HA].astype(np.float64)

mask_desi = np.isfinite(ra_all) & np.isfinite(dec_all) & np.isfinite(z_all)
mask_desi &= np.isfinite(ha_all) & (ha_all > 0)

if DESI_HA_IVAR in fast.names:
    iv = fast[DESI_HA_IVAR].astype(np.float64)
    mask_desi &= np.isfinite(iv) & (iv > 0)

ra_all = ra_all[mask_desi]
dec_all = dec_all[mask_desi]
z_all = z_all[mask_desi]
ha_all = ha_all[mask_desi]
hdul.close()

print("DESI usable rows:", len(ra_all))

idx = thin_indices(len(ra_all), N_DESI, seed=SEED)
ra_desi = ra_all[idx]
dec_desi = dec_all[idx]
z_desi = z_all[idx]
ha_desi = ha_all[idx]
print("DESI sampled:", len(ra_desi))


# -----------------------
# Match SDSS -> DESI by sky, then z
# -----------------------
print("Matching on sky...")
sdss_coords = SkyCoord(sdss[sdss_ra] * u.deg, sdss[sdss_dec] * u.deg)
desi_coords = SkyCoord(ra_desi * u.deg, dec_desi * u.deg)

idx_match, d2d, _ = sdss_coords.match_to_catalog_sky(desi_coords)
m_sky = d2d < MATCH_RADIUS
print("Sky matches:", int(np.sum(m_sky)))

z_sdss = np.asarray(sdss[sdss_z], float)
z_match = z_desi[idx_match]
m_z = m_sky & np.isfinite(z_sdss) & np.isfinite(z_match) & (np.abs(z_sdss - z_match) < DZ_MAX)
print("Sky+z matches:", int(np.sum(m_z)))

sdss_m = sdss[m_z]
i_d = idx_match[m_z]

ha_sdss_m = np.asarray(sdss_m[SDSS_HA], float)
ha_desi_m = ha_desi[i_d]

logF_sdss = safe_log10(ha_sdss_m)
logF_desi = safe_log10(ha_desi_m)

good = np.isfinite(logF_sdss) & np.isfinite(logF_desi)
logF_sdss = logF_sdss[good]
logF_desi = logF_desi[good]

delta = logF_sdss - logF_desi

print("Matched N used:", len(delta))
print("Delta logF (SDSS-DESI) median:", float(np.median(delta)))
print("Delta logF (SDSS-DESI) p16/p84:", np.percentile(delta, [16, 84]))


# -----------------------
# Plot scatter + binned trends
# -----------------------
plt.figure(figsize=(7.2, 5.6), constrained_layout=True)

# 2D density (fast, readable)
hb = plt.hexbin(logF_desi, delta, gridsize=90, bins="log", mincnt=1, cmap="viridis")
cb = plt.colorbar(hb)
cb.set_label("log10(N)")

plt.axhline(0.0, color="white", lw=1.2, alpha=0.9)
plt.axhline(np.median(delta), color="crimson", lw=1.5, alpha=0.9, label="median Δ")

# binned medians + 16/84
nbins = 20
bins = np.linspace(np.percentile(logF_desi, 1), np.percentile(logF_desi, 99), nbins + 1)
cent = 0.5 * (bins[:-1] + bins[1:])
meds = np.full(nbins, np.nan)
p16 = np.full(nbins, np.nan)
p84 = np.full(nbins, np.nan)

for i in range(nbins):
    m = (logF_desi >= bins[i]) & (logF_desi < bins[i+1])
    if np.sum(m) < 50:
        continue
    meds[i] = np.median(delta[m])
    p16[i], p84[i] = np.percentile(delta[m], [16, 84])

plt.plot(cent, meds, color="crimson", lw=2.0, label="binned median")
plt.fill_between(cent, p16, p84, color="crimson", alpha=0.25, label="binned 16–84%")

plt.xlabel(r"$\log_{10} F_{H\alpha}^{\rm DESI}$")
plt.ylabel(r"$\Delta=\log_{10}F_{H\alpha}^{\rm SDSS}-\log_{10}F_{H\alpha}^{\rm DESI}$")
plt.title("Matched SDSS–DESI: Hα flux offset vs DESI Hα flux")
plt.legend(loc="best")

out = "delta_logFha_sdss_minus_desi_vs_logFha_desi.png"
plt.savefig(out, dpi=250)
print("Saved:", out)
plt.show()
