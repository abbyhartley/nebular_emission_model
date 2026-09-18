# fullbgs_vs_sample_bpt_comp_streaming.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.io import fits
from astropy.table import Table

# -------------------------
# BPT demarcation curves (safe domain)
# -------------------------
def kewley_line(x):
    return 0.61 / (x - 0.47) + 1.19

def kauffmann_line(x):
    return 0.61 / (x - 0.05) + 1.3

# x ranges that avoid the vertical asymptotes
x_kewley = np.linspace(-2.0, 0.45, 500)   # avoid 0.47
x_kauff  = np.linspace(-2.0, 0.04, 500)   # avoid 0.05
y_kewley = kewley_line(x_kewley)
y_kauff  = kauffmann_line(x_kauff)

# -------------------------
# Paths
# -------------------------
full_fits = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo.fits")
sel_fits  = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")

# -------------------------
# Columns (SDSS, MPA-JHU)
# -------------------------
HA = "H_ALPHA_FLUX"
HB = "H_BETA_FLUX"
O3 = "OIII_5007_FLUX"
N2 = "NII_6584_FLUX"

FLUX_SCALE = 1e-17  # fastspecfit flux units

# -------------------------
# Histogram settings
# -------------------------
bins = 300
xrange = (-2.0, 1.0)   # log(NII/Ha)
yrange = (-1.5, 1.5)   # log(OIII/Hb)

def edges_from_range(nbins, r):
    return np.linspace(r[0], r[1], nbins + 1)

xedges = edges_from_range(bins, xrange)
yedges = edges_from_range(bins, yrange)

def compute_hist2d_from_arrays(x, y, xedges, yedges):
    H, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    return H

def valid_bpt_mask(ha, hb, o3, n2):
    m = np.isfinite(ha) & np.isfinite(hb) & np.isfinite(o3) & np.isfinite(n2)
    m &= (ha > 0) & (hb > 0) & (o3 > 0) & (n2 > 0)
    return m

def accumulate_full_hist(fits_path):
    """Stream over HDUs and accumulate a 2D histogram without storing all points."""
    Htot = np.zeros((bins, bins), dtype=np.int64)

    with fits.open(fits_path, memmap=True) as hdul:
        for i, hdu in enumerate(hdul[1:], start=1):
            if hdu.data is None:
                continue
            d = hdu.data
            # skip HDU if required columns missing
            needed = [HA, HB, O3, N2]
            if any(col not in d.names for col in needed):
                continue

            ha = d[HA].astype(np.float64) * FLUX_SCALE
            hb = d[HB].astype(np.float64) * FLUX_SCALE
            o3 = d[O3].astype(np.float64) * FLUX_SCALE
            n2 = d[N2].astype(np.float64) * FLUX_SCALE

            m = valid_bpt_mask(ha, hb, o3, n2)
            if not np.any(m):
                continue

            x = np.log10(n2[m] / ha[m])
            y = np.log10(o3[m] / hb[m])
            good = np.isfinite(x) & np.isfinite(y)
            if not np.any(good):
                continue

            H = compute_hist2d_from_arrays(x[good], y[good], xedges, yedges)
            Htot += H.astype(np.int64)

            if i % 10 == 0:
                print(f"Processed HDU {i}, cumulative points ~{Htot.sum():,}")

    return Htot

def hist_for_selected(sel_fits_path):
    """Compute histogram for selected file (small enough to load)."""
    t = Table.read(sel_fits_path, hdu=1)
    # keep scalar cols only
    names = [n for n in t.colnames if t[n].ndim == 1]
    t = t[names]

    ha = np.asarray(t[HA], dtype=np.float64) * FLUX_SCALE
    hb = np.asarray(t[HB], dtype=np.float64) * FLUX_SCALE
    o3 = np.asarray(t[O3], dtype=np.float64) * FLUX_SCALE
    n2 = np.asarray(t[N2], dtype=np.float64) * FLUX_SCALE

    m = valid_bpt_mask(ha, hb, o3, n2)
    x = np.log10(n2[m] / ha[m])
    y = np.log10(o3[m] / hb[m])
    good = np.isfinite(x) & np.isfinite(y)

    return compute_hist2d_from_arrays(x[good], y[good], xedges, yedges)

def plot_hist2d_side_by_side(H_full, H_sel, out_png="bpt_full_vs_selected_sdss.png"):
    vmax = max(H_full.max(), H_sel.max())
    norm = mcolors.LogNorm(vmin=1, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True, sharey=True, constrained_layout=True)

    # imshow wants [y,x] indexing; histogram2d gives H[xbin, ybin], so transpose.
    im0 = axes[0].imshow(H_full.T, origin="lower",
                         extent=[xrange[0], xrange[1], yrange[0], yrange[1]],
                         aspect="auto", cmap="plasma", norm=norm)
    im1 = axes[1].imshow(H_sel.T, origin="lower",
                         extent=[xrange[0], xrange[1], yrange[0], yrange[1]],
                         aspect="auto", cmap="plasma", norm=norm)

    for ax in axes:
        ax.plot(x_kewley, y_kewley, "r--", lw=1.5, label="Kewley (2001)")
        ax.plot(x_kauff,  y_kauff,  "b-.", lw=1.5, label="Kauffmann (2003)")
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$\log([\mathrm{NII}]\lambda6584/\mathrm{H}\alpha)$", fontsize=13)

    axes[0].set_ylabel(r"$\log([\mathrm{OIII}]\lambda5007/\mathrm{H}\beta)$", fontsize=13)
    axes[0].set_title("SDSS Main: Full parent sample", fontsize=13)
    axes[1].set_title("SDSS Main: Selected training sample", fontsize=13)
    axes[0].legend(loc="lower left", fontsize=10)

    cbar = fig.colorbar(im1, ax=axes, location="right", shrink=0.95, pad=0.02)
    cbar.set_label("Galaxy count per bin", fontsize=12)

    fig.savefig(out_png, dpi=200)
    print("Saved:", out_png)
    plt.show()

def main():
    print("Accumulating full-sample BPT histogram (streaming):", full_fits)
    H_full = accumulate_full_hist(full_fits)
    print("Done. Full histogram total counts:", int(H_full.sum()))

    print("Computing selected-sample BPT histogram:", sel_fits)
    H_sel = hist_for_selected(sel_fits)
    print("Done. Selected histogram total counts:", int(H_sel.sum()))

    plot_hist2d_side_by_side(H_full, H_sel, out_png="bpt_full_vs_selected_sdss.png")

if __name__ == "__main__":
    main()
