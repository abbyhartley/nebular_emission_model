# plot_lha_hist_full_vs_selected.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo


# -----------------------
# Shared helpers
# -----------------------
FLUX_SCALE = 1e-17  # both SDSS and DESI in your files
BINS = 80

def log10_lha(z, flux_linear_times1e17):
    """z: array, flux in catalog units (1e-17). Returns log10 L(erg/s)."""
    f_cgs = np.asarray(flux_linear_times1e17, dtype=float) * FLUX_SCALE
    z = np.asarray(z, dtype=float)
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    return np.log10(f_cgs) + np.log10(4*np.pi) + 2*np.log10(dl_cm)

def robust_range(x, p_lo=0.5, p_hi=99.5):
    x = x[np.isfinite(x)]
    return np.percentile(x, [p_lo, p_hi])


# -----------------------
# SDSS: load full (single HDU) and selected (single HDU)
# -----------------------
def load_sdss_loglha_from_fits(fits_path, *, z_col="Z_1", ha_col="H_ALPHA_FLUX"):
    t = Table.read(fits_path, hdu=1)
    # drop multi-d cols if present
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    t = t[names]
    z = np.asarray(t[z_col], dtype=float)
    ha = np.asarray(t[ha_col], dtype=float)

    m = np.isfinite(z) & np.isfinite(ha) & (ha > 0)
    return log10_lha(z[m], ha[m])


# -----------------------
# DESI: stream full multi-HDU file to avoid OOM
# -----------------------
def stream_desi_loglha_from_full(fits_path, *,
                                z_col="Z", ha_col="HALPHA_FLUX",
                                ha_ivar_col="HALPHA_FLUX_IVAR",
                                snr_col=None, snr_cut=None,
                                zwarn_col=None, zwarn_cut=None,
                                spectype_col=None, spectype_val=None):
    """
    Streams over all HDUs in a DESI fastspec-style file and returns concatenated logLHa.
    Optional additional cuts can be supplied (snr, zwarn, spectype), but defaults are None.
    """
    out = []

    with fits.open(fits_path, memmap=True) as hdul:
        for hdu in hdul[1:]:
            if hdu.data is None:
                continue
            d = hdu.data
            if z_col not in d.names or ha_col not in d.names:
                continue

            z = d[z_col].astype(np.float64)
            ha = d[ha_col].astype(np.float64)

            m = np.isfinite(z) & np.isfinite(ha) & (ha > 0)

            # require IVAR>0 if available (recommended for DESI)
            if ha_ivar_col in d.names:
                ha_ivar = d[ha_ivar_col].astype(np.float64)
                m &= np.isfinite(ha_ivar) & (ha_ivar > 0)

            if np.any(m):
                out.append(log10_lha(z[m], ha[m]))

    return np.concatenate(out) if len(out) else np.array([])


def load_desi_loglha_from_selected(fits_path, *,
                                  z_col="Z", ha_col="HALPHA_FLUX",
                                  ha_ivar_col="HALPHA_FLUX_IVAR"):
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    t = t[names]
    z = np.asarray(t[z_col], dtype=float)
    ha = np.asarray(t[ha_col], dtype=float)

    m = np.isfinite(z) & np.isfinite(ha) & (ha > 0)
    if ha_ivar_col in t.colnames:
        iv = np.asarray(t[ha_ivar_col], dtype=float)
        m &= np.isfinite(iv) & (iv > 0)

    return log10_lha(z[m], ha[m])


# -----------------------
# Plotting
# -----------------------
def plot_overlay(full_loglha, sel_loglha, *, title, out_png):
    # choose a common plotting range from the full sample
    lo, hi = robust_range(full_loglha, 0.5, 99.5)

    plt.figure(figsize=(7.0, 4.6), constrained_layout=True)
    plt.hist(full_loglha, bins=BINS, range=(lo, hi), histtype="step",
             lw=1.8, color="k", label="Full sample")
    plt.hist(sel_loglha, bins=BINS, range=(lo, hi), histtype="step",
             lw=1.8, color="crimson", label="Selected training sample")

    plt.xlabel(r"$\log_{10}\,L_{\mathrm{H}\alpha}\;[\mathrm{erg\ s^{-1}}]$")
    plt.ylabel("N")
    plt.title(title)
    plt.legend()

    # optional: log-y can help show tail differences
    # plt.yscale("log")

    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)
    plt.show()

    # print summary stats for paper text
    for name, arr in [("FULL", full_loglha), ("SEL", sel_loglha)]:
        p1, p50, p99 = np.percentile(arr[np.isfinite(arr)], [1, 50, 99])
        print(f"{title} | {name} LOG_LHA p1/p50/p99 = {p1:.3f}, {p50:.3f}, {p99:.3f} | N={len(arr):,}")


def main():
    # -------- SDSS --------
    sdss_full = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo.fits")
    sdss_sel  = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")

    print("Loading SDSS full:", sdss_full)
    lha_sdss_full = load_sdss_loglha_from_fits(sdss_full, z_col="Z_1", ha_col="H_ALPHA_FLUX")
    print("Loading SDSS selected:", sdss_sel)
    lha_sdss_sel = load_sdss_loglha_from_fits(sdss_sel, z_col="Z_1", ha_col="H_ALPHA_FLUX")

    plot_overlay(
        lha_sdss_full, lha_sdss_sel,
        title="SDSS Main: H$\\alpha$ luminosity (full vs selected)",
        out_png="sdss_logLHa_full_vs_selected.png",
    )

    # -------- DESI --------
    desi_full = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/fastspec_zall_combined.fits")
    desi_sel  = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

    print("Streaming DESI full (no OOM):", desi_full)
    lha_desi_full = stream_desi_loglha_from_full(desi_full)

    print("Loading DESI selected:", desi_sel)
    lha_desi_sel = load_desi_loglha_from_selected(desi_sel)

    plot_overlay(
        lha_desi_full, lha_desi_sel,
        title="DESI BGS: H$\\alpha$ luminosity (full vs selected)",
        out_png="desi_logLHa_full_vs_selected.png",
    )


if __name__ == "__main__":
    main()
