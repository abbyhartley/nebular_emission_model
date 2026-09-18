# compare_parent_vs_selected_sdss_desi_streaming_v3.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from scipy.ndimage import gaussian_filter


# -----------------------
# Style
# -----------------------
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 250,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
})

# Okabe–Ito (colorblind friendly)
C_PARENT = "#0072B2"   # blue
C_SEL    = "#CC79A7"   # purple

# Distinct, colorblind-friendly demarcation colors
C_KEWLEY = "#009E73"   # green
C_KAUFF  = "#E69F00"   # orange

# -----------------------
# Paths
# -----------------------
SDSS_FULL = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo.fits")
SDSS_SEL  = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")

DESI_FULL = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/fastspec_zall_combined.fits")
DESI_SEL  = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

FLUX_SCALE = 1e-17

# BPT plot range
BPT_XR = (-2.0, 1.0)
BPT_YR = (-2.0, 1.5)   # updated per your request
BPT_BINS = 260
BPT_SMOOTH_SIGMA = 1.0


# -----------------------
# BPT demarcation curves
# -----------------------
def kewley01(x):
    return 0.61 / (x - 0.47) + 1.19

def kauffmann03(x):
    return 0.61 / (x - 0.05) + 1.3

x_kew = np.linspace(-2.0, 0.45, 500)  # avoid 0.47
y_kew = kewley01(x_kew)
x_kau = np.linspace(-2.0, 0.04, 500)  # avoid 0.05
y_kau = kauffmann03(x_kau)


# -----------------------
# Helpers
# -----------------------
def read_table_scalar(path: Path, hdu=1) -> Table:
    t = Table.read(path, hdu=hdu)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names]

def safe_ratio_log10(num, den):
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    m = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    out = np.full(num.shape, np.nan, dtype=float)
    out[m] = np.log10(num[m] / den[m])
    return out

def log10_lha(z, ha_flux_1e17):
    """Return log10 L(Ha) [erg/s]. Requires z>0 and ha>0."""
    z = np.asarray(z, float)
    f = np.asarray(ha_flux_1e17, float) * FLUX_SCALE
    m = np.isfinite(z) & (z > 0) & np.isfinite(f) & (f > 0)
    if not np.any(m):
        return np.array([], dtype=float)
    dl_cm = cosmo.luminosity_distance(z[m]).to("cm").value
    out = np.log10(f[m]) + np.log10(4*np.pi) + 2*np.log10(dl_cm)
    return out[np.isfinite(out)]

def levels_from_hist(H):
    """Return contour levels for 95% and 68% HDR, given a 2D histogram H."""
    Hflat = H.ravel()
    Hflat = Hflat[Hflat > 0]
    if len(Hflat) == 0:
        return [1, 2]
    Hs = np.sort(Hflat)[::-1]
    cdf = np.cumsum(Hs) / np.sum(Hs)
    lev95 = Hs[np.searchsorted(cdf, 0.95)]
    lev68 = Hs[np.searchsorted(cdf, 0.68)]
    return [lev95, lev68]


# -----------------------
# SDSS BPT
# -----------------------
def bpt_sdss(t: Table):
    ha = np.asarray(t["H_ALPHA_FLUX"], float)
    hb = np.asarray(t["H_BETA_FLUX"], float)
    o3 = np.asarray(t["OIII_5007_FLUX"], float)
    n2 = np.asarray(t["NII_6584_FLUX"], float)
    x = safe_ratio_log10(n2, ha)
    y = safe_ratio_log10(o3, hb)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


# -----------------------
# DESI streaming BPT (parent)
# -----------------------
def stream_desi_bpt_hist2d(fits_path: Path, *, bins=BPT_BINS, xr=BPT_XR, yr=BPT_YR, smooth_sigma=BPT_SMOOTH_SIGMA):
    xedges = np.linspace(xr[0], xr[1], bins + 1)
    yedges = np.linspace(yr[0], yr[1], bins + 1)
    Htot = np.zeros((bins, bins), dtype=np.float64)

    with fits.open(fits_path, memmap=True) as hdul:
        for hdu in hdul[1:]:
            if hdu.data is None or not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                continue
            d = hdu.data
            need = ["HALPHA_FLUX", "HBETA_FLUX", "OIII_5007_FLUX", "NII_6584_FLUX"]
            if any(c not in d.names for c in need):
                continue

            ha = d["HALPHA_FLUX"].astype(np.float64)
            hb = d["HBETA_FLUX"].astype(np.float64)
            o3 = d["OIII_5007_FLUX"].astype(np.float64)
            n2 = d["NII_6584_FLUX"].astype(np.float64)

            m = np.isfinite(ha) & np.isfinite(hb) & np.isfinite(o3) & np.isfinite(n2)
            m &= (ha > 0) & (hb > 0) & (o3 > 0) & (n2 > 0)

            for iv in ["HALPHA_FLUX_IVAR", "HBETA_FLUX_IVAR", "OIII_5007_FLUX_IVAR", "NII_6584_FLUX_IVAR"]:
                if iv in d.names:
                    m &= (d[iv].astype(np.float64) > 0)

            if not np.any(m):
                continue

            x = np.log10(n2[m] / ha[m])
            y = np.log10(o3[m] / hb[m])
            good = np.isfinite(x) & np.isfinite(y)
            if not np.any(good):
                continue

            H, _, _ = np.histogram2d(x[good], y[good], bins=[xedges, yedges])
            Htot += H

    if smooth_sigma and smooth_sigma > 0:
        Htot = gaussian_filter(Htot, smooth_sigma)

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    return Htot.T, xc, yc


def bpt_desi_selected(t: Table):
    ha = np.asarray(t["HALPHA_FLUX"], float)
    hb = np.asarray(t["HBETA_FLUX"], float)
    o3 = np.asarray(t["OIII_5007_FLUX"], float)
    n2 = np.asarray(t["NII_6584_FLUX"], float)

    x = safe_ratio_log10(n2, ha)
    y = safe_ratio_log10(o3, hb)
    m = np.isfinite(x) & np.isfinite(y)

    if "HALPHA_FLUX_IVAR" in t.colnames:
        m &= (np.asarray(t["HALPHA_FLUX_IVAR"], float) > 0)
    if "HBETA_FLUX_IVAR" in t.colnames:
        m &= (np.asarray(t["HBETA_FLUX_IVAR"], float) > 0)

    return x[m], y[m]


# -----------------------
# DESI streaming logLHa (parent)
# -----------------------
def stream_desi_loglha(fits_path: Path):
    out = []
    with fits.open(fits_path, memmap=True) as hdul:
        for hdu in hdul[1:]:
            if hdu.data is None or not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                continue
            d = hdu.data
            if "Z" not in d.names or "HALPHA_FLUX" not in d.names:
                continue

            z = d["Z"].astype(np.float64)
            ha = d["HALPHA_FLUX"].astype(np.float64)

            m = np.isfinite(z) & (z > 0) & np.isfinite(ha) & (ha > 0)
            if "HALPHA_FLUX_IVAR" in d.names:
                iv = d["HALPHA_FLUX_IVAR"].astype(np.float64)
                m &= np.isfinite(iv) & (iv > 0)

            if np.any(m):
                out.append(log10_lha(z[m], ha[m]))
    return np.concatenate(out) if len(out) else np.array([])


# -----------------------
# Plot 1: BPT contours + demarcation
# -----------------------
def plot_bpt_contours(sdss_full, sdss_sel, desi_sel, out="bpt_parent_vs_selected.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharex=True, sharey=True, constrained_layout=True)

    # SDSS histograms -> smooth -> contour
    xF, yF = bpt_sdss(sdss_full)
    xS, yS = bpt_sdss(sdss_sel)

    Hf, xe, ye = np.histogram2d(xF, yF, bins=BPT_BINS, range=[BPT_XR, BPT_YR])
    Hs, _, _  = np.histogram2d(xS, yS, bins=BPT_BINS, range=[BPT_XR, BPT_YR])
    Hf = gaussian_filter(Hf, BPT_SMOOTH_SIGMA).T
    Hs = gaussian_filter(Hs, BPT_SMOOTH_SIGMA).T

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    levs_f = levels_from_hist(Hf)
    levs_s = levels_from_hist(Hs)

    axes[0].contour(xc, yc, Hf, levels=levs_f, colors=C_PARENT, linewidths=1.8, linestyles="-")
    axes[0].contour(xc, yc, Hs, levels=levs_s, colors=C_SEL,    linewidths=1.8, linestyles="-")

    axes[0].plot(x_kew, y_kew, color=C_KEWLEY, ls="--", lw=1.5)
    axes[0].plot(x_kau, y_kau, color=C_KAUFF,  ls="-.", lw=1.5)

    axes[0].set_title("SDSS")

    # DESI parent streamed histogram + selected histogram
    Hpar, xc2, yc2 = stream_desi_bpt_hist2d(DESI_FULL)
    levs_par = levels_from_hist(Hpar)

    xS, yS = bpt_desi_selected(desi_sel)
    Hsel, xe, ye = np.histogram2d(xS, yS, bins=BPT_BINS, range=[BPT_XR, BPT_YR])
    Hsel = gaussian_filter(Hsel, BPT_SMOOTH_SIGMA).T
    levs_sel = levels_from_hist(Hsel)

    axes[1].contour(xc2, yc2, Hpar, levels=levs_par, colors=C_PARENT, linewidths=1.8, linestyles="-")
    axes[1].contour(xc,  yc,  Hsel, levels=levs_sel, colors=C_SEL,    linewidths=1.8, linestyles="-")

    axes[1].plot(x_kew, y_kew, color=C_KEWLEY, ls="--", lw=1.5)
    axes[1].plot(x_kau, y_kau, color=C_KAUFF,  ls="-.", lw=1.5)

    axes[1].set_title("DESI")

    for ax in axes:
        ax.set_xlim(*BPT_XR)
        ax.set_ylim(*BPT_YR)
        ax.set_xlabel(r"$\log([\mathrm{NII}]/\mathrm{H}\alpha)$")
    axes[0].set_ylabel(r"$\log([\mathrm{OIII}]/\mathrm{H}\beta)$")

    handles = [
        plt.Line2D([0],[0], color=C_PARENT, lw=2, ls="-", label="Parent"),
        plt.Line2D([0],[0], color=C_SEL,    lw=2, ls="-", label="Selected"),
        plt.Line2D([0],[0], color=C_KEWLEY, lw=1.8, ls="--", label="Kewley (2001)"),
        plt.Line2D([0],[0], color=C_KAUFF,  lw=1.8, ls="-.", label="Kauffmann (2003)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
               frameon=True)

    fig.savefig(out)
    print("Saved:", out)


# -----------------------
# Plot 2: LHa histograms (DESI log y-scale)
# -----------------------
def plot_lha_hist(sdss_full, sdss_sel, desi_sel, out="lha_parent_vs_selected.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)

    # SDSS
    lF = log10_lha(np.asarray(sdss_full["Z_1"], float), np.asarray(sdss_full["H_ALPHA_FLUX"], float))
    lS = log10_lha(np.asarray(sdss_sel["Z_1"], float), np.asarray(sdss_sel["H_ALPHA_FLUX"], float))

    lo, hi = np.percentile(lF, [0.5, 99.5])
    axes[0].hist(lF, bins=80, range=(lo, hi), histtype="step", lw=2.2, color=C_PARENT, label="Parent")
    axes[0].hist(lS, bins=80, range=(lo, hi), histtype="step", lw=2.2, color=C_SEL,    label="Selected")
    axes[0].set_title("SDSS")
    axes[0].set_xlabel(r"$\log_{10} L_{\mathrm{H}\alpha}\;[\mathrm{erg\,s^{-1}}]$")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=True)

    # DESI parent streamed
    lF = stream_desi_loglha(DESI_FULL)
    zS = np.asarray(desi_sel["Z"], float)
    haS = np.asarray(desi_sel["HALPHA_FLUX"], float)
    lS = log10_lha(zS, haS)

    lF = lF[np.isfinite(lF)]
    lS = lS[np.isfinite(lS)]
    lo, hi = np.percentile(lF, [0.5, 99.5])

    axes[1].hist(lF, bins=80, range=(lo, hi), histtype="step", lw=2.2, color=C_PARENT, label="Parent")
    axes[1].hist(lS, bins=80, range=(lo, hi), histtype="step", lw=2.2, color=C_SEL,    label="Selected")
    axes[1].set_title("DESI")
    axes[1].set_xlabel(r"$\log_{10} L_{\mathrm{H}\alpha}\;[\mathrm{erg\,s^{-1}}]$")
    axes[1].set_yscale("log")  # <- key change
    axes[1].legend(frameon=True)

    fig.savefig(out)
    print("Saved:", out)


def main():
    print("Reading SDSS parent:", SDSS_FULL)
    sdss_full = read_table_scalar(SDSS_FULL, hdu=1)
    print("Reading SDSS selected:", SDSS_SEL)
    sdss_sel = read_table_scalar(SDSS_SEL, hdu=1)

    print("Reading DESI selected:", DESI_SEL)
    desi_sel = read_table_scalar(DESI_SEL, hdu=1)

    print("N SDSS parent/selected:", len(sdss_full), len(sdss_sel))
    print("N DESI selected:", len(desi_sel))
    print("DESI parent streamed from:", DESI_FULL)

    plot_bpt_contours(sdss_full, sdss_sel, desi_sel, out="bpt_parent_vs_selected.png")
    plot_lha_hist(sdss_full, sdss_sel, desi_sel, out="lha_parent_vs_selected.png")


if __name__ == "__main__":
    main()
