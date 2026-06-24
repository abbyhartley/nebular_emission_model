# fit_flux_mappings_all_lines_parent_with_cuts_autohdu.py
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

# SDSS basics
SDSS_SNR_COL = "SN_MEDIAN"
SDSS_SNR_MIN = 5.0

# DESI basics
DESI_RA = "RA"
DESI_DEC = "DEC"
DESI_Z = "Z"
DESI_ZWARN = "ZWARN"
DESI_SNR = "SNR_R"
DESI_SNR_MIN = 5.0

# Lines: (label, SDSS flux col, SDSS err col, DESI flux col, DESI ivar col)
LINES = [
    ("Halpha",   "H_ALPHA_FLUX",   "H_ALPHA_FLUX_ERR",   "HALPHA_FLUX",    "HALPHA_FLUX_IVAR"),
    ("Hbeta",    "H_BETA_FLUX",    "H_BETA_FLUX_ERR",    "HBETA_FLUX",     "HBETA_FLUX_IVAR"),
    ("Hgamma",   "H_GAMMA_FLUX",   "H_GAMMA_FLUX_ERR",   "HGAMMA_FLUX",    "HGAMMA_FLUX_IVAR"),
    ("NII6584",  "NII_6584_FLUX",  "NII_6584_FLUX_ERR",  "NII_6584_FLUX",  "NII_6584_FLUX_IVAR"),
    ("SII671x",  "SII_6717_FLUX",  "SII_6717_FLUX_ERR",  "SII_6716_FLUX",  "SII_6716_FLUX_IVAR"),
    ("SII6731",  "SII_6731_FLUX",  "SII_6731_FLUX_ERR",  "SII_6731_FLUX",  "SII_6731_FLUX_IVAR"),
    ("OII3726",  "OII_3726_FLUX",  "OII_3726_FLUX_ERR",  "OII_3726_FLUX",  "OII_3726_FLUX_IVAR"),
    ("OII3729",  "OII_3729_FLUX",  "OII_3729_FLUX_ERR",  "OII_3729_FLUX",  "OII_3729_FLUX_IVAR"),
    ("OIII5007", "OIII_5007_FLUX", "OIII_5007_FLUX_ERR", "OIII_5007_FLUX", "OIII_5007_FLUX_IVAR"),
]

LINE_SNR_MIN = 5.0  # per-line S/N cut applied in both surveys


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

need_sdss = [sdss_ra, sdss_dec, sdss_z]
for _, fcol, ecol, _, _ in LINES:
    need_sdss += [fcol, ecol]
missing = [c for c in need_sdss if c not in sdss.colnames]
if missing:
    raise KeyError(f"SDSS missing required columns: {missing}")

m = np.isfinite(sdss[sdss_ra]) & np.isfinite(sdss[sdss_dec]) & np.isfinite(sdss[sdss_z])

if SDSS_SNR_COL in sdss.colnames:
    m &= np.isfinite(sdss[SDSS_SNR_COL]) & (sdss[SDSS_SNR_COL] > SDSS_SNR_MIN)
else:
    print(f"WARNING: SDSS {SDSS_SNR_COL} not found; skipping continuum SNR cut.")

# Per-line S/N cuts for all target lines
for _, fcol, ecol, _, _ in LINES:
    f = np.asarray(sdss[fcol], float)
    e = np.asarray(sdss[ecol], float)
    m &= np.isfinite(f) & np.isfinite(e) & (f > 0) & (e > 0)
    m &= (f / e > LINE_SNR_MIN)

sdss = sdss[m]
print("SDSS rows after cuts:", len(sdss))


# -----------------------
# Load DESI + cuts (auto-HDU)
# -----------------------
print("Loading DESI:", DESI_FITS)
with fits.open(DESI_FITS, memmap=True) as hdul:
    hdu_pos, pos = find_table_hdu_with_all(hdul, [DESI_RA, DESI_DEC])
    if pos is None:
        raise KeyError("Could not find RA/DEC in any DESI table HDU.")
    ra_all = pos[DESI_RA].astype(np.float64)
    dec_all = pos[DESI_DEC].astype(np.float64)
    print(f"Found RA/DEC in HDU {hdu_pos}")

    # Find a table that contains Z and SNR_R if possible
    hdu_z, ztab = find_table_hdu_with_all(hdul, [DESI_Z])
    if ztab is None:
        raise KeyError("Could not find Z in any DESI table HDU.")
    z_all = ztab[DESI_Z].astype(np.float64)

    hdu_zw, zwtab = find_table_hdu_with_all(hdul, [DESI_ZWARN])
    zwarn_all = zwtab[DESI_ZWARN].astype(np.float64) if zwtab is not None else None
    if zwarn_all is None:
        print("WARNING: ZWARN not found; skipping ZWARN==0 cut.")

    hdu_snr, snrtab = find_table_hdu_with_all(hdul, [DESI_SNR])
    snr_all = snrtab[DESI_SNR].astype(np.float64) if snrtab is not None else None
    if snr_all is None:
        print("WARNING: SNR_R not found; skipping SNR_R cut.")

    # For each line, find a table containing its flux (prefer same HDU, but any is fine if aligned)
    flux_all = {}
    ivar_all = {}

    for label, _, _, dcol, icol in LINES:
        hdu_f, ftab = find_table_hdu_with_all(hdul, [dcol])
        if ftab is None:
            raise KeyError(f"Could not find DESI column {dcol} in any table HDU.")
        flux_all[label] = ftab[dcol].astype(np.float64)

        hdu_i, itab = find_table_hdu_with_all(hdul, [icol])
        if itab is None:
            print(f"WARNING: Could not find DESI IVAR column {icol}; will skip IVAR+SNR cut for {label}.")
            ivar_all[label] = None
        else:
            ivar_all[label] = itab[icol].astype(np.float64)

# sanity: all arrays must be same length to align by row index
n = len(ra_all)
for arr_name, arr in [("DEC", dec_all), ("Z", z_all)]:
    if len(arr) != n:
        raise RuntimeError(f"DESI {arr_name} length mismatch with RA; need join by TARGETID.")

for label in flux_all:
    if len(flux_all[label]) != n:
        raise RuntimeError(f"DESI flux length mismatch for {label}; need join by TARGETID.")
    if ivar_all[label] is not None and len(ivar_all[label]) != n:
        raise RuntimeError(f"DESI ivar length mismatch for {label}; need join by TARGETID.")

m = np.isfinite(ra_all) & np.isfinite(dec_all) & np.isfinite(z_all)
if zwarn_all is not None:
    m &= np.isfinite(zwarn_all) & (zwarn_all == 0)
if snr_all is not None:
    m &= np.isfinite(snr_all) & (snr_all > DESI_SNR_MIN)

# per-line S/N cuts (DESI: flux*sqrt(ivar))
for label in flux_all:
    f = flux_all[label]
    m &= np.isfinite(f) & (f > 0)
    iv = ivar_all[label]
    if iv is not None:
        m &= np.isfinite(iv) & (iv > 0)
        m &= (f * np.sqrt(iv) > LINE_SNR_MIN)

ra_all = ra_all[m]
dec_all = dec_all[m]
z_all = z_all[m]
for label in flux_all:
    flux_all[label] = flux_all[label][m]

print("DESI rows after cuts:", len(ra_all))

idx = thin_indices(len(ra_all), N_DESI, seed=2)
ra_desi = ra_all[idx]
dec_desi = dec_all[idx]
z_desi = z_all[idx]
for label in flux_all:
    flux_all[label] = flux_all[label][idx]
print("DESI sampled:", len(ra_desi))


# -----------------------
# Match
# -----------------------
print("Matching...")
sdss_coords = SkyCoord(np.asarray(sdss[sdss_ra], float) * u.deg, np.asarray(sdss[sdss_dec], float) * u.deg)
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
# Per-line Δ plots + summary
# -----------------------
nlines = len(LINES)
ncols = 3
nrows = int(np.ceil(nlines / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 3.5*nrows), constrained_layout=True)
axes = np.atleast_1d(axes).ravel()

print("\nPer-line flux offsets on matched sample (Δ = logF_SDSS - logF_DESI):")
for k, (label, sd_fcol, _sd_ecol, _de_fcol, _de_icol) in enumerate(LINES):
    f_sdss = np.asarray(sdss_m[sd_fcol], float)
    f_desi = flux_all[label][i_d]

    logF_sdss = safe_log10(f_sdss)
    logF_desi = safe_log10(f_desi)
    good = np.isfinite(logF_sdss) & np.isfinite(logF_desi)

    x = logF_desi[good]
    delta = logF_sdss[good] - logF_desi[good]

    med = float(np.median(delta))
    p16, p84 = np.percentile(delta, [16, 84])
    print(f"{label:8s}: N={good.sum():6d}  median={med:+.4f}  p16/p84={p16:+.4f}/{p84:+.4f}  scale=10^med={10**med:.3f}")

    ax = axes[k]
    hb = ax.hexbin(x, delta, gridsize=75, bins="log", mincnt=1, cmap="viridis")
    ax.axhline(med, color="crimson", lw=1.8)
    cent, dmed, d16, d84 = binned_stats(x, delta, nbins=18, min_per_bin=50)
    ax.plot(cent, dmed, color="crimson", lw=2.0)
    ax.fill_between(cent, d16, d84, color="crimson", alpha=0.25)

    ax.set_title(label)
    ax.set_xlabel(r"$\log_{10} F_{\rm DESI}$")
    ax.set_ylabel(r"$\Delta=\log F_{\rm SDSS}-\log F_{\rm DESI}$")

# remove unused axes
for ax in axes[nlines:]:
    ax.axis("off")

out = "delta_logF_sdss_minus_desi_all_lines.png"
plt.savefig(out, dpi=250)
print("\nSaved:", out)
plt.show()
