# fit_ha_flux_mapping_parent_with_cuts_autohdu.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u


SDSS_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo.fits")
DESI_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/fastspec-iron-main-bright.fits")

MATCH_RADIUS = 1.0 * u.arcsec
DZ_MAX = 5e-4
N_DESI = 2_000_000

# SDSS columns
SDSS_HA_COL = "H_ALPHA_FLUX"
SDSS_HA_ERR_COL = "H_ALPHA_FLUX_ERR"
SDSS_SNR_COL = "SN_MEDIAN"
SDSS_SNR_MIN = 5.0
SDSS_HA_SNR_MIN = 5.0

# DESI columns (we will search HDUs for these)
DESI_RA = "RA"
DESI_DEC = "DEC"
DESI_Z = "Z"
DESI_ZWARN = "ZWARN"
DESI_SNR = "SNR_R"
DESI_HA = "HALPHA_FLUX"
DESI_HA_IVAR = "HALPHA_FLUX_IVAR"
DESI_SNR_MIN = 5.0
DESI_HA_SNR_MIN = 5.0


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

def binned_stats(x, y, nbins=20, min_per_bin=50):
    bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), nbins + 1)
    cent = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(nbins, np.nan)
    p16 = np.full(nbins, np.nan)
    p84 = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (x >= bins[i]) & (x < bins[i+1]) & np.isfinite(y)
        if np.sum(m) < min_per_bin:
            continue
        med[i] = np.median(y[m])
        p16[i], p84[i] = np.percentile(y[m], [16, 84])
    return cent, med, p16, p84

def find_table_hdu_with_cols(hdul, cols_any, cols_all=()):
    """
    Find the first table HDU that contains all cols_all and at least one of cols_any.
    Set cols_any=[] to ignore that requirement.
    Returns: (hdu_index, recarray)
    """
    for i, h in enumerate(hdul):
        if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
            continue
        if h.data is None:
            continue
        names = list(h.columns.names)
        if any(c in names for c in cols_any) and all(c in names for c in cols_all):
            return i, h.data
    return None, None

def find_table_hdu_with_all(hdul, cols_all):
    return find_table_hdu_with_cols(hdul, cols_any=cols_all, cols_all=cols_all)


# -----------------------
# Load SDSS + cuts
# -----------------------
print("Loading SDSS:", SDSS_FITS)
sdss = Table.read(SDSS_FITS, hdu=1)
sdss_ra = "RA_1" if "RA_1" in sdss.colnames else "RA"
sdss_dec = "DEC_1" if "DEC_1" in sdss.colnames else "DEC"
sdss_z = "Z_1" if "Z_1" in sdss.colnames else "Z"

for c in [sdss_ra, sdss_dec, sdss_z, SDSS_HA_COL, SDSS_HA_ERR_COL]:
    if c not in sdss.colnames:
        raise KeyError(f"SDSS missing required column: {c}")

m = np.isfinite(sdss[sdss_ra]) & np.isfinite(sdss[sdss_dec]) & np.isfinite(sdss[sdss_z])
m &= np.isfinite(sdss[SDSS_HA_COL]) & (sdss[SDSS_HA_COL] > 0)
m &= np.isfinite(sdss[SDSS_HA_ERR_COL]) & (sdss[SDSS_HA_ERR_COL] > 0)
m &= (sdss[SDSS_HA_COL] / sdss[SDSS_HA_ERR_COL] > SDSS_HA_SNR_MIN)

if SDSS_SNR_COL in sdss.colnames:
    m &= np.isfinite(sdss[SDSS_SNR_COL]) & (sdss[SDSS_SNR_COL] > SDSS_SNR_MIN)
else:
    print(f"WARNING: SDSS {SDSS_SNR_COL} not found; skipping continuum SNR cut.")

sdss = sdss[m]
print("SDSS rows after cuts:", len(sdss))


# -----------------------
# Load DESI from arbitrary HDU layout
# -----------------------
print("Loading DESI:", DESI_FITS)
with fits.open(DESI_FITS, memmap=True) as hdul:
    # RA/DEC often live in a "meta" HDU, but search all tables
    hdu_pos, pos = find_table_hdu_with_all(hdul, [DESI_RA, DESI_DEC])
    if pos is None:
        raise KeyError("Could not find RA/DEC in any DESI table HDU.")
    ra_all = pos[DESI_RA].astype(np.float64)
    dec_all = pos[DESI_DEC].astype(np.float64)
    print(f"Found RA/DEC in HDU {hdu_pos}")

    # Halpha flux (+ optional ivar)
    # Prefer a table that has both HALPHA_FLUX and Z (common), but fall back if needed.
    hdu_ha, ha_tab = find_table_hdu_with_cols(hdul, cols_any=[DESI_HA], cols_all=[DESI_HA])
    if ha_tab is None:
        raise KeyError("Could not find HALPHA_FLUX in any DESI table HDU.")
    print(f"Found HALPHA_FLUX in HDU {hdu_ha}")
    ha_all = ha_tab[DESI_HA].astype(np.float64)

    # Z: try same HDU as Halpha first, else search anywhere
    if DESI_Z in ha_tab.dtype.names:
        z_all = ha_tab[DESI_Z].astype(np.float64)
    else:
        hdu_z, z_tab = find_table_hdu_with_all(hdul, [DESI_Z])
        if z_tab is None:
            raise KeyError("Could not find Z in any DESI table HDU.")
        print(f"Found Z in HDU {hdu_z} (different from HALPHA_FLUX HDU)")
        z_all = z_tab[DESI_Z].astype(np.float64)

    # Optional: ZWARN and SNR_R (use if found, otherwise skip)
    zwarn_all = None
    snr_all = None

    hdu_zw, zw_tab = find_table_hdu_with_all(hdul, [DESI_ZWARN])
    if zw_tab is not None:
        zwarn_all = zw_tab[DESI_ZWARN].astype(np.float64)
        print(f"Found ZWARN in HDU {hdu_zw}")
    else:
        print("WARNING: ZWARN not found in this DESI file; skipping ZWARN==0 cut.")

    hdu_snr, snr_tab = find_table_hdu_with_all(hdul, [DESI_SNR])
    if snr_tab is not None:
        snr_all = snr_tab[DESI_SNR].astype(np.float64)
        print(f"Found SNR_R in HDU {hdu_snr}")
    else:
        print("WARNING: SNR_R not found in this DESI file; skipping SNR_R cut.")

    # Optional: Halpha ivar (use if found, otherwise skip IVAR cuts)
    ha_ivar_all = None
    if DESI_HA_IVAR in ha_tab.dtype.names:
        ha_ivar_all = ha_tab[DESI_HA_IVAR].astype(np.float64)
    else:
        hdu_ivar, iv_tab = find_table_hdu_with_all(hdul, [DESI_HA_IVAR])
        if iv_tab is not None:
            ha_ivar_all = iv_tab[DESI_HA_IVAR].astype(np.float64)
            print(f"Found HALPHA_FLUX_IVAR in HDU {hdu_ivar}")
        else:
            print("WARNING: HALPHA_FLUX_IVAR not found; skipping IVAR and per-line SNR cuts.")

# Sanity: lengths must match. If they don't, file uses different row sets per HDU.
# In that case, you need a common identifier column to join (TARGETID). We check here:
n = len(ra_all)
if len(dec_all) != n or len(z_all) != n or len(ha_all) != n:
    raise RuntimeError(
        "DESI HDUs have different lengths; cannot align rows by index. "
        "Need a common key (e.g., TARGETID) and a join. "
        f"Lengths: RA={len(ra_all)} Z={len(z_all)} HA={len(ha_all)}"
    )

m = np.isfinite(ra_all) & np.isfinite(dec_all) & np.isfinite(z_all)
m &= np.isfinite(ha_all) & (ha_all > 0)

if zwarn_all is not None:
    m &= np.isfinite(zwarn_all) & (zwarn_all == 0)
if snr_all is not None:
    m &= np.isfinite(snr_all) & (snr_all > DESI_SNR_MIN)
if ha_ivar_all is not None:
    m &= np.isfinite(ha_ivar_all) & (ha_ivar_all > 0)
    m &= (ha_all * np.sqrt(ha_ivar_all) > DESI_HA_SNR_MIN)

ra_all = ra_all[m]
dec_all = dec_all[m]
z_all = z_all[m]
ha_all = ha_all[m]

print("DESI rows after cuts:", len(ra_all))

idx = thin_indices(len(ra_all), N_DESI, seed=2)
ra_desi = ra_all[idx]
dec_desi = dec_all[idx]
z_desi = z_all[idx]
ha_desi = ha_all[idx]
print("DESI sampled:", len(ra_desi))


# -----------------------
# Match + compute Δ
# -----------------------
print("Matching...")
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

logF_sdss = safe_log10(np.asarray(sdss_m[SDSS_HA_COL], float))
logF_desi = safe_log10(ha_desi[i_d])

good = np.isfinite(logF_sdss) & np.isfinite(logF_desi)
logF_sdss = logF_sdss[good]
logF_desi = logF_desi[good]
delta = logF_sdss - logF_desi

med = float(np.median(delta))
p16, p84 = np.percentile(delta, [16, 84])
print("\nMatched N used:", len(delta))
print("Δ median:", med)
print("Δ p16/p84:", p16, p84)
print("Scale factor 10^medianΔ:", float(10**med))


# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
hb = plt.hexbin(logF_desi, delta, gridsize=90, bins="log", mincnt=1, cmap="viridis")
cb = plt.colorbar(hb); cb.set_label("log10(N)")

plt.axhline(med, color="crimson", lw=1.8, label="median Δ")
cent, dmed, d16, d84 = binned_stats(logF_desi, delta, nbins=20, min_per_bin=50)
plt.plot(cent, dmed, color="crimson", lw=2.0, label="binned median")
plt.fill_between(cent, d16, d84, color="crimson", alpha=0.25, label="binned 16–84%")

plt.xlabel(r"$\log_{10} F_{H\alpha}^{\rm DESI}$")
plt.ylabel(r"$\Delta=\log_{10}F_{H\alpha}^{\rm SDSS}-\log_{10}F_{H\alpha}^{\rm DESI}$")
plt.title("Matched SDSS–DESI (auto-HDU): Hα flux offset vs DESI Hα flux")
plt.legend(loc="best")

out = "delta_logFha_sdss_minus_desi_vs_logFha_desi_parent_with_cuts_autohdu.png"
plt.savefig(out, dpi=250)
print("\nSaved:", out)
plt.show()
