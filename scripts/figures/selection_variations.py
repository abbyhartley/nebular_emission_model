# selection_variations.py
#
# Explore alternative selection functions on the BPT diagram. Parent = DESI main/bright
# (ZWARN=0, valid z) and SDSS main (GALAXY, ZWARN=0). Load once; produce:
#   A) continuum S/N>3 AND per-line S/N>3 (all 9 incl Halpha)         -> bpt_var_A_perline3.png
#   B) continuum S/N>3 AND r-band mag limit (instead of per-line S/N) -> bpt_var_B_rmag.png
#   C) continuum S/N>3 AND Halpha S/N>7 only (no per-line on others)  -> bpt_var_C_halpha.png
#      + missing-line diagnostic among the Halpha-selected sample     -> var_C_missing_lines.png
# All also require z>0.05.  DESI apparent r is APPROX (M_r + distance modulus).

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import Planck15
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

LINE_LABELS = ["[OII]3726", "[OII]3729", "H$\\gamma$", "H$\\beta$", "H$\\alpha$",
               "[OIII]5007", "[NII]6584", "[SII]6716", "[SII]6731"]
NONHA = [0, 1, 2, 3, 5, 6, 7, 8]
IDX = dict(HB=3, HA=4, OIII=5, NII=6)
SDSS_FLUX = ["OII_3726_FLUX", "OII_3729_FLUX", "H_GAMMA_FLUX", "H_BETA_FLUX", "H_ALPHA_FLUX",
             "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6717_FLUX", "SII_6731_FLUX"]
DESI_FLUX = ["OII_3726_FLUX", "OII_3729_FLUX", "HGAMMA_FLUX", "HBETA_FLUX", "HALPHA_FLUX",
             "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX"]

BPT_XR, BPT_YR = (-2.0, 1.0), (-2.0, 1.5)
Z_MIN, CONT_MIN = 0.05, 3.0
PERLINE_A = 3.0            # Variation A per-line threshold
HA_MIN_C = 7.0            # Variation C Halpha S/N threshold
R_LIM = {"SDSS": 17.77, "DESI": 19.5}

C_KEW, C_KAU = "#009E73", "#E69F00"
CMAP_SNR = getattr(cmr, "fusion", "coolwarm")
CMAP_Z = getattr(cmr, "bubblegum", "viridis")
CMAP_R = getattr(cmr, "ember", "magma")
C_ALL, C_MISS = "#CC79A7", "#D55E00"

_zg = np.linspace(1e-4, 1.5, 4000)
_dmg = Planck15.distmod(_zg).value
def distmod(z): return np.interp(np.clip(z, 1e-4, 1.5), _zg, _dmg)


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


def pack(rec, flux_cols, kind, z, zwarn, logm, cont, r_app, base_extra=True):
    snr9 = per_line_snr(rec, flux_cols, kind)
    x, y = bpt_xy(rec, flux_cols)
    base = np.isfinite(z) & (z > 0) & np.isfinite(zwarn) & (zwarn == 0) & base_extra
    return dict(snr9=snr9, min9=snr9.min(1).astype(np.float32), ha=snr9[:, IDX["HA"]].astype(np.float32),
                x=x, y=y, z=z.astype(np.float32), logm=logm.astype(np.float32),
                cont=cont.astype(np.float32), r=r_app.astype(np.float32), base=base)


def load_sdss():
    t = Table.read(SDSS_FULL, hdu=1)
    z = np.asarray(t["Z_1"], float); zwarn = np.asarray(t["Z_WARNING"], float)
    logm = np.asarray(t["LGM_TOT_P50"], float)
    cont = np.asarray(t["SN_MEDIAN"], float)
    if cont.ndim > 1:
        cont = np.nanmedian(cont, axis=1)
    r_app = np.asarray(t["corrmag_r"], float)                 # real apparent r
    spok = np.ones(len(t), bool)
    if "SPECTROTYPE" in t.colnames:
        spok = (np.char.strip(np.asarray(t["SPECTROTYPE"]).astype(str)) == "GALAXY")
    return pack(t, SDSS_FLUX, "err", z, zwarn, logm, cont, r_app, base_extra=spok)


def load_desi():
    parts = []
    with fits.open(DESI_FULL, memmap=True) as hdul:
        for hi in range(1, len(hdul)):
            h = hdul[hi]
            if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                continue
            d = h.data
            need = DESI_FLUX + ["Z", "ZWARN", "LOGMSTAR", "SNR_R", "SURVEY", "PROGRAM", "ABSMAG01_SDSS_R"]
            if d is None or any(c not in d.names for c in need):
                continue
            z = d["Z"].astype(float); zwarn = d["ZWARN"].astype(float)
            logm = d["LOGMSTAR"].astype(float); cont = d["SNR_R"].astype(float)
            r_app = d["ABSMAG01_SDSS_R"].astype(float) + distmod(z)   # APPROX apparent r
            sv = np.char.lower(np.char.strip(np.asarray(d["SURVEY"]).astype(str)))
            pg = np.char.lower(np.char.strip(np.asarray(d["PROGRAM"]).astype(str)))
            parts.append(pack(d, DESI_FLUX, "ivar", z, zwarn, logm, cont, r_app,
                              base_extra=(sv == "main") & (pg == "bright")))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def levels(H):
    f = np.sort(H[H > 0].ravel())[::-1]
    if len(f) == 0:
        return [1, 2]
    cdf = np.cumsum(f) / f.sum()
    return [f[min(np.searchsorted(cdf, q), len(f) - 1)] for q in (0.95, 0.68)]


def contour(ax, xs, ys, color="k"):
    m = np.isfinite(xs) & np.isfinite(ys)
    H, xe, ye = np.histogram2d(xs[m], ys[m], bins=200, range=[BPT_XR, BPT_YR])
    H = gaussian_filter(H, 1.0).T
    ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), H, levels=levels(H),
               colors=color, linewidths=1.7)


def bpt_col(ax, x, y, C, cmap, norm=None):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(C)
    hb = ax.hexbin(x[m], y[m], C=C[m], reduce_C_function=np.median, gridsize=85,
                   extent=(*BPT_XR, *BPT_YR), cmap=cmap, norm=norm, mincnt=8)
    xk = np.linspace(BPT_XR[0], 0.45, 400); xa = np.linspace(BPT_XR[0], 0.04, 400)
    ax.plot(xk, kewley01(xk), color=C_KEW, ls="--", lw=1.5)
    ax.plot(xa, kauffmann03(xa), color=C_KAU, ls="-.", lw=1.5)
    ax.set_xlim(*BPT_XR); ax.set_ylim(*BPT_YR); ax.set_box_aspect(1)
    return hb


def bpt_fig(DATA, SEL, cols, outname, title):
    """cols: list of (getter(A)->array, label, cmap, norm)."""
    fig, ax = plt.subplots(2, len(cols), figsize=(5.4 * len(cols), 11), constrained_layout=True)
    for r, tag in enumerate(["SDSS", "DESI"]):
        A = DATA[tag]; sel = SEL[tag]; b = A["base"]
        for c, (getter, label, cmap, norm) in enumerate(cols):
            C = getter(A)
            hb = bpt_col(ax[r, c], A["x"][b], A["y"][b], C[b], cmap, norm)
            contour(ax[r, c], A["x"][sel], A["y"][sel])
            cb = fig.colorbar(hb, ax=ax[r, c], shrink=0.85); cb.set_label(label)
            ax[r, c].set_xlabel(r"$\log([\mathrm{NII}]/\mathrm{H}\alpha)$")
            if c == 0:
                ax[r, c].set_ylabel(f"{tag}\n" + r"$\log([\mathrm{OIII}]/\mathrm{H}\beta)$")
        if r == 0:
            for c, (_, label, _, _) in enumerate(cols):
                ax[0, c].set_title(label.split("(")[0])
    ax[0, 0].plot([], [], "k-", lw=1.7, label="selected 68/95%"); ax[0, 0].legend(loc="lower left")
    fig.suptitle(title, fontsize=15)
    fig.savefig(outname, dpi=205, bbox_inches="tight")
    print("Saved:", outname, flush=True)


def main():
    print("loading SDSS...", flush=True); S = load_sdss()
    print("loading DESI (streaming)...", flush=True); D = load_desi()
    DATA = {"SDSS": S, "DESI": D}

    log10c = TwoSlopeNorm(vcenter=np.log10(CONT_MIN), vmin=np.log10(0.5), vmax=np.log10(60.0))
    log3 = TwoSlopeNorm(vcenter=np.log10(3.0), vmin=np.log10(0.5), vmax=np.log10(60.0))
    log7 = TwoSlopeNorm(vcenter=np.log10(7.0), vmin=np.log10(1.0), vmax=np.log10(200.0))

    def g_cont(A): return np.log10(np.clip(A["cont"], 1e-2, None))
    def g_min9(A): return np.log10(np.clip(A["min9"], 1e-2, None))
    def g_ha(A): return np.log10(np.clip(A["ha"], 1e-2, None))
    def g_z(A): return A["z"]
    def g_r(A): return A["r"]

    # ---- Variation A: continuum>3 AND per-line>3 ----
    SEL_A = {t: DATA[t]["base"] & (DATA[t]["z"] > Z_MIN) & (DATA[t]["cont"] > CONT_MIN)
             & (DATA[t]["min9"] > PERLINE_A) for t in DATA}
    for t in DATA:
        print(f"[A {t}] selected(cont>3 & per-line>3) = {SEL_A[t].sum():,}", flush=True)
    bpt_fig(DATA, SEL_A,
            [(g_cont, r"med. $\log_{10}$(continuum S/N)  [cut 3]", CMAP_SNR, log10c),
             (g_min9, r"med. $\log_{10}$(weakest-line S/N)  [cut 3]", CMAP_SNR, log3),
             (g_z, "median redshift", CMAP_Z, None)],
            "bpt_var_A_perline3.png",
            "Variation A: continuum S/N>3 AND per-line S/N>3 (all 9); selected overlaid")

    # ---- Variation B: continuum>3 AND r-mag limit (no per-line requirement) ----
    SEL_B = {t: DATA[t]["base"] & (DATA[t]["z"] > Z_MIN) & (DATA[t]["cont"] > CONT_MIN)
             & (DATA[t]["r"] < R_LIM[t]) for t in DATA}
    for t in DATA:
        print(f"[B {t}] selected(cont>3 & r<{R_LIM[t]}) = {SEL_B[t].sum():,}  "
              f"(r-cut removes {int((DATA[t]['base'] & (DATA[t]['r']>=R_LIM[t])).sum()):,} of base)", flush=True)
    bpt_fig(DATA, SEL_B,
            [(g_cont, r"med. $\log_{10}$(continuum S/N)  [cut 3]", CMAP_SNR, log10c),
             (g_r, "median apparent r [mag]", CMAP_R, None),
             (g_z, "median redshift", CMAP_Z, None)],
            "bpt_var_B_rmag.png",
            "Variation B: continuum S/N>3 AND r-mag limit (no per-line cut); DESI r approx")

    # ---- Variation C: continuum>3 AND Halpha S/N>7 only ----
    SEL_C = {t: DATA[t]["base"] & (DATA[t]["z"] > Z_MIN) & (DATA[t]["cont"] > CONT_MIN)
             & (DATA[t]["ha"] > HA_MIN_C) for t in DATA}
    for t in DATA:
        print(f"[C {t}] selected(cont>3 & Halpha S/N>7) = {SEL_C[t].sum():,}", flush=True)
    bpt_fig(DATA, SEL_C,
            [(g_cont, r"med. $\log_{10}$(continuum S/N)  [cut 3]", CMAP_SNR, log10c),
             (g_ha, r"med. $\log_{10}$(H$\alpha$ S/N)  [cut 7]", CMAP_SNR, log7),
             (g_z, "median redshift", CMAP_Z, None)],
            "bpt_var_C_halpha.png",
            "Variation C: continuum S/N>3 AND H-alpha S/N>7 only (no per-line on the other 8)")

    # ---- Variation C diagnostic: which lines drop out among Halpha-selected ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for r, tag in enumerate(["SDSS", "DESI"]):
        A = DATA[tag]; sel = SEL_C[tag]
        snr9 = A["snr9"][sel]
        undet = (snr9 < 5.0)          # "missing" = S/N<5 at that line
        frac = undet[:, NONHA].mean(0)
        n_missing = undet[:, NONHA].sum(1)      # how many of the 8 are missing per galaxy
        all8 = (n_missing == 0)
        miss_any = ~all8
        print(f"[C-diag {tag}] Halpha-selected N={sel.sum():,}; all 8 lines S/N>5 in {all8.mean():.0%}; "
              f"missing >=1 in {miss_any.mean():.0%}", flush=True)
        for i, li in enumerate(NONHA):
            print(f"     {LINE_LABELS[li]:11s} missing (S/N<5) in {frac[i]:.0%}", flush=True)
        # col0: bar of missing fractions
        yp = np.arange(len(NONHA))
        ax[r, 0].barh(yp, frac, color=C_MISS, alpha=0.85)
        ax[r, 0].set_yticks(yp); ax[r, 0].set_yticklabels([LINE_LABELS[i] for i in NONHA]); ax[r, 0].invert_yaxis()
        ax[r, 0].set_xlabel(r"frac. of H$\alpha$-selected with line S/N<5")
        ax[r, 0].set_ylabel(f"{tag}")
        # col1: stellar mass, all-8 vs missing-any
        mb = np.linspace(7.5, 12.5, 55)
        lm = A["logm"][sel]
        for msk, col, lab in [(all8, C_ALL, "all 8 detected"), (miss_any, C_MISS, "missing $\\geq$1 line")]:
            v = lm[msk]; v = v[np.isfinite(v) & (v > 6) & (v < 13)]
            ax[r, 1].hist(v, bins=mb, density=True, histtype="step", lw=2, color=col, label=lab)
        ax[r, 1].set_xlabel(r"$\log(M_\star/M_\odot)$"); ax[r, 1].legend()
        # col2: redshift, all-8 vs missing-any
        zz = A["z"][sel]
        zb = np.linspace(0, np.nanpercentile(zz, 99.5), 55)
        for msk, col, lab in [(all8, C_ALL, "all 8 detected"), (miss_any, C_MISS, "missing $\\geq$1 line")]:
            ax[r, 2].hist(zz[msk], bins=zb, density=True, histtype="step", lw=2, color=col, label=lab)
        ax[r, 2].set_xlabel("redshift"); ax[r, 2].legend()
    ax[0, 0].set_title(r"which line drops out (S/N<5)"); ax[0, 1].set_title("stellar mass"); ax[0, 2].set_title("redshift")
    fig.suptitle(r"Variation C diagnostic: among H$\alpha$ S/N>7 galaxies, which lines are still undetected, and in what populations",
                 fontsize=14)
    fig.savefig("var_C_missing_lines.png", dpi=205, bbox_inches="tight")
    print("Saved: var_C_missing_lines.png", flush=True)


if __name__ == "__main__":
    main()
