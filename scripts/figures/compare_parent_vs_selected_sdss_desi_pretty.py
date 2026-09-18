# bpt_parent_vs_selected_sdssmain_desibgs_squareboxes.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from astropy.table import Table
from astropy.io import fits
from scipy.ndimage import gaussian_filter

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 250,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 11.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Okabe–Ito (colorblind friendly)
C_PARENT = "#0072B2"   # blue
C_SEL    = "#CC79A7"   # purple
C_KEWLEY = "#009E73"   # green
C_KAUFF  = "#E69F00"   # orange

SDSS_FULL = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo.fits")
SDSS_SEL  = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")

DESI_FULL = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/fastspec_zall_combined.fits")
DESI_SEL  = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

# Requested axis limits
BPT_XR = (-2.0, 1.0)
BPT_YR = (-2.0, 1.5)
BPT_BINS = 260
BPT_SMOOTH_SIGMA = 1.0


def kewley01(x):
    return 0.61 / (x - 0.47) + 1.19

def kauffmann03(x):
    return 0.61 / (x - 0.05) + 1.3

x_kew = np.linspace(BPT_XR[0], 0.45, 500)
y_kew = kewley01(x_kew)
x_kau = np.linspace(BPT_XR[0], 0.04, 500)
y_kau = kauffmann03(x_kau)


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

def levels_from_hist(H):
    Hflat = H.ravel()
    Hflat = Hflat[Hflat > 0]
    if len(Hflat) == 0:
        return [1, 2]
    Hs = np.sort(Hflat)[::-1]
    cdf = np.cumsum(Hs) / np.sum(Hs)
    lev95 = Hs[np.searchsorted(cdf, 0.95)]
    lev68 = Hs[np.searchsorted(cdf, 0.68)]
    return [lev95, lev68]

def bpt_sdss_xy(t: Table):
    ha = np.asarray(t["H_ALPHA_FLUX"], float)
    hb = np.asarray(t["H_BETA_FLUX"], float)
    o3 = np.asarray(t["OIII_5007_FLUX"], float)
    n2 = np.asarray(t["NII_6584_FLUX"], float)
    x = safe_ratio_log10(n2, ha)
    y = safe_ratio_log10(o3, hb)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def bpt_desi_xy_selected(t: Table):
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

def stream_desi_parent_bpt_hist2d(fits_path: Path):
    xedges = np.linspace(BPT_XR[0], BPT_XR[1], BPT_BINS + 1)
    yedges = np.linspace(BPT_YR[0], BPT_YR[1], BPT_BINS + 1)
    Htot = np.zeros((BPT_BINS, BPT_BINS), dtype=np.float64)

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

    Htot = gaussian_filter(Htot, BPT_SMOOTH_SIGMA)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    return Htot.T, xc, yc


def main():
    sdss_full = read_table_scalar(SDSS_FULL, hdu=1)
    sdss_sel  = read_table_scalar(SDSS_SEL, hdu=1)
    desi_sel  = read_table_scalar(DESI_SEL, hdu=1)

    # Use box_aspect=1 to make axes *look* square without enforcing equal data scaling
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10.6, 5.3),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.05},
        constrained_layout=False
    )

    # --- SDSS Main ---
    xF, yF = bpt_sdss_xy(sdss_full)
    xS, yS = bpt_sdss_xy(sdss_sel)

    Hf, xe, ye = np.histogram2d(xF, yF, bins=BPT_BINS, range=[BPT_XR, BPT_YR])
    Hs, _, _   = np.histogram2d(xS, yS, bins=BPT_BINS, range=[BPT_XR, BPT_YR])
    Hf = gaussian_filter(Hf, BPT_SMOOTH_SIGMA).T
    Hs = gaussian_filter(Hs, BPT_SMOOTH_SIGMA).T

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    lev_f = levels_from_hist(Hf)
    lev_s = levels_from_hist(Hs)

    axes[0].contour(xc, yc, Hf, levels=lev_f, colors=C_PARENT, linewidths=1.9, linestyles="-")
    axes[0].contour(xc, yc, Hs, levels=lev_s, colors=C_SEL,    linewidths=1.9, linestyles="-")
    axes[0].plot(x_kew, y_kew, color=C_KEWLEY, ls="--", lw=1.6)
    axes[0].plot(x_kau, y_kau, color=C_KAUFF,  ls="-.", lw=1.6)
    axes[0].set_title("SDSS Main")

    # --- DESI BGS ---
    Hpar, xc2, yc2 = stream_desi_parent_bpt_hist2d(DESI_FULL)
    lev_par = levels_from_hist(Hpar)

    xS, yS = bpt_desi_xy_selected(desi_sel)
    Hsel, xe, ye = np.histogram2d(xS, yS, bins=BPT_BINS, range=[BPT_XR, BPT_YR])
    Hsel = gaussian_filter(Hsel, BPT_SMOOTH_SIGMA).T
    lev_sel = levels_from_hist(Hsel)

    axes[1].contour(xc2, yc2, Hpar, levels=lev_par, colors=C_PARENT, linewidths=1.9, linestyles="-")
    axes[1].contour(xc,  yc,  Hsel, levels=lev_sel, colors=C_SEL,    linewidths=1.9, linestyles="-")
    axes[1].plot(x_kew, y_kew, color=C_KEWLEY, ls="--", lw=1.6)
    axes[1].plot(x_kau, y_kau, color=C_KAUFF,  ls="-.", lw=1.6)
    axes[1].set_title("DESI BGS")

    for ax in axes:
        ax.set_xlim(*BPT_XR)
        ax.set_ylim(*BPT_YR)
        ax.set_xlabel(r"$\log([\mathrm{NII}]/\mathrm{H}\alpha)$")
        ax.set_box_aspect(1)  # square-looking axes box, not equal scaling

    axes[0].set_ylabel(r"$\log([\mathrm{OIII}]/\mathrm{H}\beta)$")

    handles = [
        plt.Line2D([0],[0], color=C_PARENT, lw=2.2, ls="-", label="Parent"),
        plt.Line2D([0],[0], color=C_SEL,    lw=2.2, ls="-", label="Selected"),
        plt.Line2D([0],[0], color=C_KEWLEY, lw=1.8, ls="--", label="Kewley (2001)"),
        plt.Line2D([0],[0], color=C_KAUFF,  lw=1.8, ls="-.", label="Kauffmann (2003)"),
    ]
    axes[0].legend(handles=handles, loc="lower left", frameon=True)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.90, wspace=0.05)

    out = "bpt_parent_vs_selected_sdssmain_desibgs.png"
    fig.savefig(out)
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()
