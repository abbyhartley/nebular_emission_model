# match_sdss_desi_compare_line_ratios.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

# -----------------------
# User parameters
# -----------------------
SDSS_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo.fits")
DESI_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/fastspec-iron-main-bright.fits")  # change if needed

N_SDSS = 500_000
N_DESI = 1_000_000
MATCH_RADIUS = 1.0 * u.arcsec
DZ_MAX = 5e-4

# Lines: (label, sdss_flux_col, desi_flux_col)
LINES = [
    ("Hbeta",    "H_BETA_FLUX",   "HBETA_FLUX"),
    ("Hgamma",   "H_GAMMA_FLUX",  "HGAMMA_FLUX"),
    ("NII6584",  "NII_6584_FLUX", "NII_6584_FLUX"),
    ("SII671x",  "SII_6717_FLUX", "SII_6716_FLUX"),   # SDSS=6717, DESI=6716
    ("SII6731",  "SII_6731_FLUX", "SII_6731_FLUX"),
    ("OII3726",  "OII_3726_FLUX", "OII_3726_FLUX"),
    ("OII3729",  "OII_3729_FLUX", "OII_3729_FLUX"),
    ("OIII5007", "OIII_5007_FLUX","OIII_5007_FLUX"),
]

# Halpha columns
SDSS_HA = "H_ALPHA_FLUX"
DESI_HA = "HALPHA_FLUX"

# Redshift columns
SDSS_Z = "Z_1"   # if not present, we try "Z"
DESI_Z = "Z"     # in fast HDU

# Continuum SNR (optional)
DESI_SNR_COL = "SNR_R"  # in fast HDU for many fastspec products


# -----------------------
# Helpers
# -----------------------
def pick_col(tab, candidates):
    for c in candidates:
        if c in tab.colnames:
            return c
    raise KeyError(f"None of these columns found: {candidates}")

def robust_scatter(x):
    p16, p84 = np.percentile(x, [16, 84])
    return 0.5 * (p84 - p16)

def safe_log10_ratio(num, den):
    m = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    out = np.full_like(num, np.nan, dtype=float)
    out[m] = np.log10(num[m] / den[m])
    return out

def thin_indices(n, n_keep, seed=0):
    rng = np.random.default_rng(seed)
    n_keep = min(n_keep, n)
    return rng.choice(n, size=n_keep, replace=False)


# -----------------------
# Load SDSS
# -----------------------
print("Loading SDSS:", SDSS_FITS)
sdss = Table.read(SDSS_FITS, hdu=1)

# resolve RA/DEC and z column names robustly
sdss_ra_col = pick_col(sdss, ["RA_1", "RA"])
sdss_dec_col = pick_col(sdss, ["DEC_1", "DEC"])
sdss_z_col = sdss.colnames[sdss.colnames.index(SDSS_Z)] if SDSS_Z in sdss.colnames else pick_col(sdss, ["Z", "Z_1"])

# minimal validity mask: need positions, z, Hα, and the 8 lines
need_cols = [sdss_ra_col, sdss_dec_col, sdss_z_col, SDSS_HA] + [c for _, c, _ in LINES]
for c in need_cols:
    if c not in sdss.colnames:
        raise KeyError(f"SDSS missing required column: {c}")

mask_sdss = np.isfinite(sdss[sdss_ra_col]) & np.isfinite(sdss[sdss_dec_col]) & np.isfinite(sdss[sdss_z_col])
mask_sdss &= np.isfinite(sdss[SDSS_HA])

# You can either keep negative fluxes (and later mask per-ratio), or pre-mask to positive:
# Here we keep them and let safe_log10_ratio handle it.
sdss = sdss[mask_sdss]
print("SDSS rows after basic finite cuts:", len(sdss))

# downsample for matching speed
idx = thin_indices(len(sdss), N_SDSS, seed=1)
sdss = sdss[idx]
print("SDSS sampled:", len(sdss))


# -----------------------
# Load DESI (fast + meta HDUs like your earlier script)
# -----------------------
print("Loading DESI:", DESI_FITS)
hdul = fits.open(DESI_FITS, memmap=True)
fast = hdul[1].data
meta = hdul[2].data

# require RA/DEC in meta; z and line fluxes in fast
if "RA" not in meta.names or "DEC" not in meta.names:
    raise KeyError("DESI meta HDU missing RA/DEC columns.")

if DESI_Z not in fast.names:
    raise KeyError(f"DESI fast HDU missing {DESI_Z}")

needed_fast = [DESI_Z, DESI_HA, DESI_SNR_COL] + [c for _, _, c in LINES]
for c in [DESI_Z, DESI_HA] + [c for _, _, c in LINES]:
    if c not in fast.names:
        raise KeyError(f"DESI fast HDU missing required column: {c}")

ra_all = meta["RA"].astype(np.float64)
dec_all = meta["DEC"].astype(np.float64)

z_all = fast[DESI_Z].astype(np.float64)
ha_all = fast[DESI_HA].astype(np.float64)

snr_all = fast[DESI_SNR_COL].astype(np.float64) if DESI_SNR_COL in fast.names else None

mask_desi = np.isfinite(ra_all) & np.isfinite(dec_all) & np.isfinite(z_all) & np.isfinite(ha_all)
# Keep all flux signs; ratios will be masked later.
mask_desi &= (ha_all != 0)  # optional: keep 0s out (they're usually sentinel); we still require >0 for ratios later.

# optional: enforce Halpha ivar>0 if present (recommended)
if "HALPHA_FLUX_IVAR" in fast.names:
    mask_desi &= (fast["HALPHA_FLUX_IVAR"].astype(np.float64) > 0)

ra_all = ra_all[mask_desi]
dec_all = dec_all[mask_desi]
z_all = z_all[mask_desi]
ha_all = ha_all[mask_desi]
snr_all = snr_all[mask_desi] if snr_all is not None else None

# also keep the line flux arrays for later
line_flux_desi = {}
for name, _, desi_col in LINES:
    line_flux_desi[name] = fast[desi_col].astype(np.float64)[mask_desi]

hdul.close()

print("DESI rows after basic finite cuts:", len(ra_all))

# downsample for matching speed
idx = thin_indices(len(ra_all), N_DESI, seed=2)
ra_desi = ra_all[idx]
dec_desi = dec_all[idx]
z_desi = z_all[idx]
ha_desi = ha_all[idx]
snr_desi = snr_all[idx] if snr_all is not None else None
for name in line_flux_desi:
    line_flux_desi[name] = line_flux_desi[name][idx]

print("DESI sampled:", len(ra_desi))


# -----------------------
# Sky match + redshift consistency
# -----------------------
print("Matching on sky...")
sdss_coords = SkyCoord(sdss[sdss_ra_col] * u.deg, sdss[sdss_dec_col] * u.deg)
desi_coords = SkyCoord(ra_desi * u.deg, dec_desi * u.deg)

idx_match, d2d, _ = sdss_coords.match_to_catalog_sky(desi_coords)
m_sky = d2d < MATCH_RADIUS
print("Sky matches:", int(np.sum(m_sky)))

# redshift cut
z_sdss = np.asarray(sdss[sdss_z_col], dtype=float)
z_match = z_desi[idx_match]
m_z = m_sky & np.isfinite(z_sdss) & np.isfinite(z_match) & (np.abs(z_sdss - z_match) < DZ_MAX)
print("Sky+z matches:", int(np.sum(m_z)))

# matched arrays
sdss_m = sdss[m_z]
i_d = idx_match[m_z]

z_sdss_m = np.asarray(sdss_m[sdss_z_col], float)
z_desi_m = z_desi[i_d]
snr_desi_m = snr_desi[i_d] if snr_desi is not None else None

# -----------------------
# Compute ratios and deltas
# -----------------------
print("Computing Δ log(line/Hα) for matched galaxies...")

ha_sdss_m = np.asarray(sdss_m[SDSS_HA], float)
ha_desi_m = ha_desi[i_d]

delta_by_line = {}

for label, sdss_col, desi_col in LINES:
    f_sdss = np.asarray(sdss_m[sdss_col], float)
    f_desi = line_flux_desi[label][i_d]

    r_sdss = safe_log10_ratio(f_sdss, ha_sdss_m)
    r_desi = safe_log10_ratio(f_desi, ha_desi_m)

    dlt = r_desi - r_sdss
    good = np.isfinite(dlt)

    delta_by_line[label] = dlt[good]

    med = np.median(dlt[good])
    scat = robust_scatter(dlt[good])
    print(f"{label:8s}: N={good.sum():6d}  median={med:+.3f} dex  scatter~{scat:.3f} dex")

# -----------------------
# Plot 1: median ± scatter per line
# -----------------------
labels = [l for l,_,_ in LINES]
meds = [np.median(delta_by_line[l]) for l in labels]
scats = [robust_scatter(delta_by_line[l]) for l in labels]

x = np.arange(len(labels))
plt.figure(figsize=(10, 4.2), constrained_layout=True)
plt.errorbar(x, meds, yerr=scats, fmt="o", capsize=3, color="k")
plt.axhline(0, color="0.5", lw=1)
plt.xticks(x, labels, rotation=0)
plt.ylabel(r"$\Delta \log_{10}(F_{\rm line}/F_{H\alpha})$  (DESI - SDSS)")
plt.title("Matched SDSS–DESI galaxies: median ratio offsets (±0.5[p84-p16])")
plt.savefig("matched_ratio_offsets.png", dpi=200)
plt.show()

# -----------------------
# Plot 2: Δ vs redshift (per line, small multiples)
# -----------------------
fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, constrained_layout=True)
axes = axes.ravel()

for j, (label, _, _) in enumerate(LINES):
    ax = axes[j]
    # Need same-length arrays; recompute Δ with full mask per object for this line
    f_sdss = np.asarray(sdss_m[LINES[j][1]], float)
    f_desi = line_flux_desi[label][i_d]
    r_sdss = safe_log10_ratio(f_sdss, ha_sdss_m)
    r_desi = safe_log10_ratio(f_desi, ha_desi_m)
    dlt = r_desi - r_sdss
    good = np.isfinite(dlt) & np.isfinite(z_sdss_m)

    ax.hexbin(z_sdss_m[good], dlt[good], gridsize=60, bins="log", mincnt=1, cmap="viridis")
    ax.axhline(0, color="w", lw=1)
    ax.set_title(label)
    ax.set_ylabel("Δ log ratio")
    ax.set_xlabel("z (SDSS)")

fig.suptitle("Matched SDSS–DESI galaxies: Δ log(line/Hα) vs redshift", fontsize=14)
plt.savefig("matched_delta_vs_z.png", dpi=200)
plt.show()

# -----------------------
# Plot 3: Δ vs DESI continuum SNR (if available)
# -----------------------
if snr_desi_m is not None:
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for j, (label, sdss_col, _) in enumerate(LINES):
        ax = axes[j]
        f_sdss = np.asarray(sdss_m[sdss_col], float)
        f_desi = line_flux_desi[label][i_d]
        r_sdss = safe_log10_ratio(f_sdss, ha_sdss_m)
        r_desi = safe_log10_ratio(f_desi, ha_desi_m)
        dlt = r_desi - r_sdss
        good = np.isfinite(dlt) & np.isfinite(snr_desi_m)

        ax.hexbin(snr_desi_m[good], dlt[good], gridsize=60, bins="log", mincnt=1, cmap="magma")
        ax.axhline(0, color="w", lw=1)
        ax.set_title(label)
        ax.set_ylabel("Δ log ratio")
        ax.set_xlabel("DESI SNR_R")

    fig.suptitle("Matched SDSS–DESI galaxies: Δ log(line/Hα) vs DESI continuum SNR", fontsize=14)
    plt.savefig("matched_delta_vs_snr.png", dpi=200)
    plt.show()
else:
    print("DESI SNR_R not found; skipping Δ vs SNR plot.")

print("Done.")
