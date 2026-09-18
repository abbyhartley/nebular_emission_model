# defend_selection_figs.py
#
# Defend the v1 selection by SHOWING what it removes. The actual cuts (from
# src/normflow/selection.py training_mask) are, in order:
#   (1) z > 0.05
#   (2) ZWARN == 0  (+ SPECTYPE==GALAXY where available; DESI restricted to main/bright)
#   (3) GLOBAL continuum S/N > 5   (DESI: SNR_R ; SDSS: SN_MEDIAN)
#   (4) per-line S/N > 5 on ALL 9 lines  (DESI FLUX*sqrt(IVAR); SDSS FLUX/ERR)
# We decompose the excluded galaxies by which cut first removes them, and show the
# populations affected. Validated: recon selected ~= on-disk (DESI 130,382 / SDSS 57,935).

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import scienceplots  # noqa
import cmasher as cmr

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 15, "axes.titlesize": 15,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})

GFC = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs"
SDSS_FULL = f"{GFC}/mpa_rcsed2_combo.fits"
DESI_FULL = f"{GFC}/fastspec_zall_combined.fits"
SEL_COUNT = {"SDSS": 57935, "DESI": 130382}

LINE_LABELS = ["[OII]3726", "[OII]3729", "H$\\gamma$", "H$\\beta$", "H$\\alpha$",
               "[OIII]5007", "[NII]6584", "[SII]6716", "[SII]6731"]
IDX = dict(HB=3, HA=4, OIII=5, NII=6)
SDSS_FLUX = ["OII_3726_FLUX", "OII_3729_FLUX", "H_GAMMA_FLUX", "H_BETA_FLUX", "H_ALPHA_FLUX",
             "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6717_FLUX", "SII_6731_FLUX"]
DESI_FLUX = ["OII_3726_FLUX", "OII_3729_FLUX", "HGAMMA_FLUX", "HBETA_FLUX", "HALPHA_FLUX",
             "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX"]

BPT_XR, BPT_YR = (-2.0, 1.0), (-2.0, 1.5)
Z_MIN, SNR_MIN = 0.05, 5.0

C_INCL = "#CC79A7"   # included
C_CONT = "#0072B2"   # excluded by continuum S/N (blue)
C_LINE = "#D55E00"   # excluded by per-line S/N (vermillion)
C_Z = "#999999"      # excluded by z<0.05 (grey)
C_KEW, C_KAU = "#009E73", "#E69F00"
CMAP_SNR = getattr(cmr, "fusion", "coolwarm")
CMAP_Z = getattr(cmr, "bubblegum", "viridis")


def kewley01(x): return 0.61 / (x - 0.47) + 1.19
def kauffmann03(x): return 0.61 / (x - 0.05) + 1.3


def per_line_snr(rec, flux_cols, kind):
    n = len(rec[flux_cols[0]])
    snr = np.zeros((n, 9), np.float32)
    for i, fc in enumerate(flux_cols):
        flux = np.asarray(rec[fc], float)
        unc = np.asarray(rec[fc + ("_ERR" if kind == "err" else "_IVAR")], float)
        ok = np.isfinite(flux) & np.isfinite(unc) & (flux > 0) & (unc > 0)
        s = np.zeros(n)
        s[ok] = flux[ok] / unc[ok] if kind == "err" else flux[ok] * np.sqrt(unc[ok])
        snr[:, i] = s
    return snr


def bpt_xy(rec, flux_cols):
    n = len(rec[flux_cols[0]])
    ha = np.asarray(rec[flux_cols[IDX["HA"]]], float); hb = np.asarray(rec[flux_cols[IDX["HB"]]], float)
    o3 = np.asarray(rec[flux_cols[IDX["OIII"]]], float); n2 = np.asarray(rec[flux_cols[IDX["NII"]]], float)
    x = np.full(n, np.nan); y = np.full(n, np.nan)
    ox = np.isfinite(n2) & np.isfinite(ha) & (n2 > 0) & (ha > 0)
    oy = np.isfinite(o3) & np.isfinite(hb) & (o3 > 0) & (hb > 0)
    x[ox] = np.log10(n2[ox] / ha[ox]); y[oy] = np.log10(o3[oy] / hb[oy])
    return x.astype(np.float32), y.astype(np.float32)


def pack(rec, flux_cols, kind, z, zwarn, logm, cont, base_extra=True):
    snr9 = per_line_snr(rec, flux_cols, kind)
    x, y = bpt_xy(rec, flux_cols)
    base = np.isfinite(z) & (z > 0) & np.isfinite(zwarn) & (zwarn == 0) & base_extra
    return dict(min_snr=snr9.min(1).astype(np.float32), arg_lim=snr9.argmin(1).astype(np.int8),
                x=x, y=y, z=z.astype(np.float32), logm=logm.astype(np.float32),
                cont=cont.astype(np.float32), base=base)


def load_sdss():
    t = Table.read(SDSS_FULL, hdu=1)
    z = np.asarray(t["Z_1"], float); zwarn = np.asarray(t["Z_WARNING"], float)
    logm = np.asarray(t["LGM_TOT_P50"], float)
    cont = np.asarray(t["SN_MEDIAN"], float)
    if cont.ndim > 1:
        cont = np.nanmedian(cont, axis=1)
    spok = np.ones(len(t), bool)
    if "SPECTROTYPE" in t.colnames:
        spok = (np.char.strip(np.asarray(t["SPECTROTYPE"]).astype(str)) == "GALAXY")
    return pack(t, SDSS_FLUX, "err", z, zwarn, logm, cont, base_extra=spok)


def load_desi():
    parts = []
    with fits.open(DESI_FULL, memmap=True) as hdul:
        for hi in range(1, len(hdul)):
            h = hdul[hi]
            if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                continue
            d = h.data
            need = DESI_FLUX + ["Z", "ZWARN", "LOGMSTAR", "SNR_R", "SURVEY", "PROGRAM"]
            if d is None or any(c not in d.names for c in need):
                continue
            z = d["Z"].astype(float); zwarn = d["ZWARN"].astype(float)
            logm = d["LOGMSTAR"].astype(float); cont = d["SNR_R"].astype(float)
            sv = np.char.lower(np.char.strip(np.asarray(d["SURVEY"]).astype(str)))
            pg = np.char.lower(np.char.strip(np.asarray(d["PROGRAM"]).astype(str)))
            parts.append(pack(d, DESI_FLUX, "ivar", z, zwarn, logm, cont,
                              base_extra=(sv == "main") & (pg == "bright")))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def masks(a):
    base = a["base"]
    pz = a["z"] > Z_MIN
    pc = a["cont"] > SNR_MIN
    pl = a["min_snr"] > SNR_MIN
    sel = base & pz & pc & pl
    # disjoint exclusion reasons, priority z -> continuum -> per-line
    ex_z = base & ~pz
    ex_c = base & pz & ~pc
    ex_l = base & pz & pc & ~pl
    return dict(sel=sel, ex_z=ex_z, ex_c=ex_c, ex_l=ex_l, base=base)


def levels(H):
    f = np.sort(H[H > 0].ravel())[::-1]
    if len(f) == 0:
        return [1, 2]
    cdf = np.cumsum(f) / f.sum()
    return [f[min(np.searchsorted(cdf, q), len(f) - 1)] for q in (0.95, 0.68)]


def contour(ax, xs, ys):
    H, xe, ye = np.histogram2d(xs, ys, bins=200, range=[BPT_XR, BPT_YR])
    H = gaussian_filter(H, 1.0).T
    ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), H, levels=levels(H),
               colors="k", linewidths=1.7)


def bpt_col(ax, x, y, C, cmap, norm=None):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(C)
    hb = ax.hexbin(x[m], y[m], C=C[m], reduce_C_function=np.median, gridsize=85,
                   extent=(*BPT_XR, *BPT_YR), cmap=cmap, norm=norm, mincnt=8)
    xk = np.linspace(BPT_XR[0], 0.45, 400); xa = np.linspace(BPT_XR[0], 0.04, 400)
    ax.plot(xk, kewley01(xk), color=C_KEW, ls="--", lw=1.5)
    ax.plot(xa, kauffmann03(xa), color=C_KAU, ls="-.", lw=1.5)
    ax.set_xlim(*BPT_XR); ax.set_ylim(*BPT_YR); ax.set_box_aspect(1)
    return hb


def main():
    print("loading SDSS...", flush=True); S = load_sdss()
    print("loading DESI (streaming)...", flush=True); D = load_desi()
    DATA = {"SDSS": S, "DESI": D}
    M = {}
    for tag, A in DATA.items():
        m = masks(A); M[tag] = m
        tot_excl = m["ex_z"].sum() + m["ex_c"].sum() + m["ex_l"].sum()
        print(f"[{tag}] base={m['base'].sum():,}  selected(recon)={m['sel'].sum():,} "
              f"(target {SEL_COUNT[tag]:,})", flush=True)
        print(f"       excl z<0.05={m['ex_z'].sum():,}  excl continuum-S/N={m['ex_c'].sum():,}  "
              f"excl per-line-S/N={m['ex_l'].sum():,}", flush=True)
        lim = A["arg_lim"][m["ex_l"]]; frac = np.bincount(lim, minlength=9) / max(len(lim), 1)
        order = np.argsort(frac)[::-1]
        print(f"       per-line-excluded limited by:",
              ", ".join(f"{LINE_LABELS[i]}={frac[i]:.0%}" for i in order[:4]), flush=True)

    # ---------- FIGURE 1: BPT colored by continuum S/N, min per-line S/N, redshift ----------
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 11), constrained_layout=True)
    nrm = TwoSlopeNorm(vcenter=np.log10(SNR_MIN), vmin=np.log10(1.0), vmax=np.log10(60.0))
    for r, tag in enumerate(["SDSS", "DESI"]):
        A, m = DATA[tag], M[tag]; b = m["base"]
        h0 = bpt_col(ax[r, 0], A["x"][b], A["y"][b], np.log10(np.clip(A["cont"][b], 1e-2, None)), CMAP_SNR, nrm)
        contour(ax[r, 0], A["x"][m["sel"]], A["y"][m["sel"]])
        c0 = fig.colorbar(h0, ax=ax[r, 0], shrink=0.85); c0.set_label(r"med. $\log_{10}$(continuum S/N)")
        c0.ax.axhline(np.log10(SNR_MIN), color="k", lw=2)
        h1 = bpt_col(ax[r, 1], A["x"][b], A["y"][b], np.log10(np.clip(A["min_snr"][b], 1e-2, None)), CMAP_SNR, nrm)
        contour(ax[r, 1], A["x"][m["sel"]], A["y"][m["sel"]])
        c1 = fig.colorbar(h1, ax=ax[r, 1], shrink=0.85); c1.set_label(r"med. $\log_{10}$(min per-line S/N)")
        c1.ax.axhline(np.log10(SNR_MIN), color="k", lw=2)
        h2 = bpt_col(ax[r, 2], A["x"][b], A["y"][b], A["z"][b], CMAP_Z)
        contour(ax[r, 2], A["x"][m["sel"]], A["y"][m["sel"]])
        c2 = fig.colorbar(h2, ax=ax[r, 2], shrink=0.85); c2.set_label("median redshift")
        ax[r, 0].set_ylabel(f"{tag}\n" + r"$\log([\mathrm{OIII}]/\mathrm{H}\beta)$")
        for c in range(3):
            ax[r, c].set_xlabel(r"$\log([\mathrm{NII}]/\mathrm{H}\alpha)$")
    ax[0, 0].set_title("continuum S/N"); ax[0, 1].set_title("min per-line S/N"); ax[0, 2].set_title("redshift")
    ax[0, 0].plot([], [], "k-", lw=1.7, label="selected 68/95%"); ax[0, 0].legend(loc="lower left")
    fig.suptitle("Parent BPT (main/bright) colored by the selection quantities; selected sample overlaid",
                 fontsize=16)
    fig.savefig("bpt_selection_snr_z.png", dpi=210, bbox_inches="tight")
    print("Saved: bpt_selection_snr_z.png", flush=True)

    # ---------- FIGURE 2: exclusion decomposition ----------
    fig2, a2 = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    sb = np.logspace(np.log10(0.3), np.log10(300), 60)
    for r, tag in enumerate(["SDSS", "DESI"]):
        A, m = DATA[tag], M[tag]
        # continuum S/N
        ax0 = a2[r, 0]
        ax0.hist(np.clip(A["cont"][m["sel"]], 0.3, 300), bins=sb, density=True, histtype="step", lw=2,
                 color=C_INCL, label=f"included ({m['sel'].sum():,})")
        ax0.hist(np.clip(A["cont"][m["ex_c"]], 0.3, 300), bins=sb, density=True, histtype="step", lw=2,
                 color=C_CONT, label=f"excl: continuum ({m['ex_c'].sum():,})")
        ax0.axvline(SNR_MIN, color="k", ls=":", lw=1.5); ax0.set_xscale("log")
        ax0.set_xlabel("continuum S/N"); ax0.set_ylabel(f"{tag}\nnormalized density"); ax0.legend()
        # min per-line S/N
        ax1 = a2[r, 1]
        ax1.hist(np.clip(A["min_snr"][m["sel"]], 0.3, 300), bins=sb, density=True, histtype="step", lw=2,
                 color=C_INCL, label="included")
        ax1.hist(np.clip(A["min_snr"][m["ex_l"]], 0.3, 300), bins=sb, density=True, histtype="step", lw=2,
                 color=C_LINE, label=f"excl: per-line ({m['ex_l'].sum():,})")
        ax1.axvline(SNR_MIN, color="k", ls=":", lw=1.5); ax1.set_xscale("log")
        ax1.set_xlabel("min per-line S/N"); ax1.legend()
        # redshift
        ax2 = a2[r, 2]
        zb = np.linspace(0, np.nanpercentile(A["z"][m["base"]], 99.5), 60)
        ax2.hist(A["z"][m["sel"]], bins=zb, density=True, histtype="step", lw=2, color=C_INCL, label="included")
        ax2.hist(A["z"][m["ex_c"]], bins=zb, density=True, histtype="step", lw=2, color=C_CONT, label="excl: cont")
        ax2.hist(A["z"][m["ex_l"]], bins=zb, density=True, histtype="step", lw=2, color=C_LINE, label="excl: line")
        ax2.hist(A["z"][m["ex_z"]], bins=zb, density=True, histtype="step", lw=2, color=C_Z, label="excl: z<0.05")
        ax2.axvline(Z_MIN, color="k", ls=":", lw=1.5); ax2.set_xlabel("redshift"); ax2.legend()
        # stellar mass
        ax3 = a2[r, 3]; mb = np.linspace(7.5, 12.5, 60)
        for msk, col, lab in [(m["sel"], C_INCL, "included"), (m["ex_c"], C_CONT, "excl: cont"),
                              (m["ex_l"], C_LINE, "excl: line")]:
            v = A["logm"][msk]; v = v[np.isfinite(v) & (v > 6) & (v < 13)]
            ax3.hist(v, bins=mb, density=True, histtype="step", lw=2, color=col, label=lab)
        ax3.set_xlabel(r"$\log(M_\star/M_\odot)$"); ax3.legend()
    a2[0, 0].set_title("continuum S/N"); a2[0, 1].set_title("min per-line S/N")
    a2[0, 2].set_title("redshift"); a2[0, 3].set_title("stellar mass")
    fig2.suptitle("What each cut removes: continuum-S/N cut drops faint (low-mass/high-z); "
                  "per-line cut drops weak-lined (high-mass); z-cut drops z<0.05", fontsize=14)
    fig2.savefig("selection_excluded_populations.png", dpi=210, bbox_inches="tight")
    print("Saved: selection_excluded_populations.png", flush=True)


if __name__ == "__main__":
    main()
