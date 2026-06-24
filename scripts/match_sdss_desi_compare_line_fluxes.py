# match_sdss_desi_compare_line_fluxes.py
#
# Crossmatch SDSS and DESI (sky + dz) and then directly compare
# emission-line flux measurements for the SAME galaxies:
#   x-axis: SDSS flux
#   y-axis: DESI flux
#
# Produces a 3x3 grid (8 target lines + Halpha), square panels, 1:1 line,
# and prints median offset in dex: median[ log10(F_DESI/F_SDSS) ].

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "no-latex"]) 

from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import spearmanr


# -----------------------
# Inputs / parameters
# -----------------------
SDSS_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo.fits")
DESI_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/fastspec-iron-main-bright.fits")

MATCH_RADIUS = 1.0 * u.arcsec
DZ_MAX = 5e-4

# Use all SDSS rows (your SDSS file is only ~294k)
N_DESI = 1_000_000  # downsample DESI for speed; increase for more matches

# Flux units: both are typically in 1e-17 erg/s/cm^2 for these products.
# We compare in log space so the scaling cancels, but keep consistent.
FLUX_SCALE = 1.0

# (panel label, SDSS flux col, DESI flux col)
LINES = [
    ("H$\\alpha$",  "H_ALPHA_FLUX", "HALPHA_FLUX"),
    ("H$\\beta$",   "H_BETA_FLUX",  "HBETA_FLUX"),
    ("H$\\gamma$",  "H_GAMMA_FLUX", "HGAMMA_FLUX"),
    ("[NII]6584",   "NII_6584_FLUX","NII_6584_FLUX"),
    ("[SII]671x",   "SII_6717_FLUX","SII_6716_FLUX"),
    ("[SII]6731",   "SII_6731_FLUX","SII_6731_FLUX"),
    ("[OII]3726",   "OII_3726_FLUX","OII_3726_FLUX"),
    ("[OII]3729",   "OII_3729_FLUX","OII_3729_FLUX"),
    ("[OIII]5007",  "OIII_5007_FLUX","OIII_5007_FLUX"),
]


# -----------------------
# Helpers
# -----------------------
def thin_indices(n, n_keep, seed=0):
    rng = np.random.default_rng(seed)
    n_keep = min(n_keep, n)
    return rng.choice(n, size=n_keep, replace=False)

def robust_scatter_dex(dlog):
    p16, p84 = np.percentile(dlog, [16, 84])
    return 0.5 * (p84 - p16)

def pick_col(tab, candidates):
    for c in candidates:
        if c in tab.colnames:
            return c
    raise KeyError(f"None of these columns found: {candidates}")

def make_square_hex(ax, x, y, label, *, mincnt=3):
    # robust symmetric limits in log-flux space
    xv = x[np.isfinite(x) & np.isfinite(y)]
    yv = y[np.isfinite(x) & np.isfinite(y)]
    lo = np.percentile(np.concatenate([xv, yv]), 0.5)
    hi = np.percentile(np.concatenate([xv, yv]), 99.5)
    pad = 0.15 * (hi - lo)
    lo -= pad; hi += pad

    hb = ax.hexbin(xv, yv, gridsize=70, extent=(lo, hi, lo, hi),
                   bins="log", mincnt=mincnt, cmap="viridis")
    ax.plot([lo, hi], [lo, hi], color="k", lw=1.0, alpha=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel(r"$\log F_{\rm SDSS}$")
    ax.set_ylabel(r"$\log F_{\rm DESI}$")
    return hb

def log10_safe(flux):
    flux = np.asarray(flux, float) * FLUX_SCALE
    out = np.full_like(flux, np.nan, dtype=float)
    m = np.isfinite(flux) & (flux > 0)
    out[m] = np.log10(flux[m])
    return out


# -----------------------
# Load SDSS
# -----------------------
print("Loading SDSS:", SDSS_FITS)
sdss = Table.read(SDSS_FITS, hdu=1)

sdss_ra = pick_col(sdss, ["RA_1", "RA"])
sdss_dec = pick_col(sdss, ["DEC_1", "DEC"])
sdss_z = pick_col(sdss, ["Z_1", "Z"])

# basic finite cuts
mask_sdss = np.isfinite(sdss[sdss_ra]) & np.isfinite(sdss[sdss_dec]) & np.isfinite(sdss[sdss_z])
for _, c_sdss, _ in LINES:
    if c_sdss not in sdss.colnames:
        raise KeyError(f"SDSS missing required column: {c_sdss}")
    mask_sdss &= np.isfinite(sdss[c_sdss])

sdss = sdss[mask_sdss]
print("SDSS rows after basic finite cuts:", len(sdss))


# -----------------------
# Load DESI (fast + meta)
# -----------------------
print("Loading DESI:", DESI_FITS)
hdul = fits.open(DESI_FITS, memmap=True)
fast = hdul[1].data
meta = hdul[2].data

if "RA" not in meta.names or "DEC" not in meta.names:
    raise KeyError("DESI meta HDU missing RA/DEC")
if "Z" not in fast.names:
    raise KeyError("DESI fast HDU missing Z")
for _, _, c_desi in LINES:
    if c_desi not in fast.names:
        raise KeyError(f"DESI fast HDU missing required column: {c_desi}")

ra_all = meta["RA"].astype(np.float64)
dec_all = meta["DEC"].astype(np.float64)
z_all = fast["Z"].astype(np.float64)

mask_desi = np.isfinite(ra_all) & np.isfinite(dec_all) & np.isfinite(z_all)

# recommended: require Halpha ivar > 0 to avoid sentinel zeros if present
if "HALPHA_FLUX_IVAR" in fast.names:
    mask_desi &= (fast["HALPHA_FLUX_IVAR"].astype(np.float64) > 0)

for _, _, c_desi in LINES:
    mask_desi &= np.isfinite(fast[c_desi])

ra_all = ra_all[mask_desi]
dec_all = dec_all[mask_desi]
z_all = z_all[mask_desi]

flux_desi = {lab: fast[c].astype(np.float64)[mask_desi] for (lab, _, c) in LINES}
hdul.close()

print("DESI rows after basic finite cuts:", len(ra_all))

idx = thin_indices(len(ra_all), N_DESI, seed=2)
ra_desi = ra_all[idx]
dec_desi = dec_all[idx]
z_desi = z_all[idx]
for lab in flux_desi:
    flux_desi[lab] = flux_desi[lab][idx]
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


# -----------------------
# Build log-flux arrays and plot
# -----------------------
# 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
axes = axes.ravel()

hb_last = None

print("\nPer-line median offset: median log10(F_DESI/F_SDSS) and robust scatter")
for j, (lab, c_sdss, _c_desi) in enumerate(LINES):
    # SDSS flux
    f_sdss = np.asarray(sdss_m[c_sdss], float)

    # DESI flux pulled from dict by label (same ordering as LINES)
    f_desi = flux_desi[lab][i_d]

    # Use log space; require both positive
    lx = log10_safe(f_sdss)
    ly = log10_safe(f_desi)
    good = np.isfinite(lx) & np.isfinite(ly)

    # offsets
    dlog = (ly[good] - lx[good])
    med = float(np.median(dlog))
    scat = float(robust_scatter_dex(dlog))
    rho, _ = spearmanr(lx[good], ly[good])

    print(f"{lab:10s}: N={good.sum():6d}  med={med:+.3f} dex  scat={scat:.3f} dex  rho={float(rho):.3f}")

    # plot
    ax = axes[j]
    hb_last = make_square_hex(ax, lx[good], ly[good], lab, mincnt=5)

    # inset text
    ax.text(
        0.03, 0.97,
        f"med={med:+.2f} dex\nscat={scat:.2f}\n$\\rho$={float(rho):.2f}",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round,pad=0.2")
    )

# shared colorbar
cbar = fig.colorbar(hb_last, ax=axes.tolist(), location="right", shrink=0.92, pad=0.01)
cbar.set_label("log10(N per hex), mincnt=5")

fig.suptitle("Matched SDSS–DESI galaxies: log flux comparisons", fontsize=14)
out = "matched_sdss_vs_desi_flux_panels.png"
fig.savefig(out, dpi=250)
print("\nSaved:", out)
plt.show()
