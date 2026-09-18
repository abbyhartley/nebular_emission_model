# altb_completeness_figs.py
# Replacement for bpt_parent_vs_selected: show how complete / biased the ALT-B training
# sample is vs the full parent galaxy sample. Three permutations:
#   FIG1 completeness_coverage_hist.png : parent vs ALT-B normalized distributions of
#         z, logM_color, logL_Ha, log(Halpha flux)   [coverage + shape bias]
#   FIG2 completeness_fraction.png      : completeness = N_selected/N_parent vs each var
#   FIG3 completeness_bpt.png           : parent BPT hexbin colored by median z / logM* /
#         log(Halpha flux); ALT-B 68/95% contours overlaid
# Parent = GALAXY(main/bright DESI; SPECTROTYPE SDSS) & ZWARN=0 & z>0 & finite LOGM_COLOR.
# ALT-B  = parent & z>0.05 & continuum S/N>3 & 8-target-line S/N>3 & Halpha detected.

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm  # noqa
import scienceplots  # noqa
import cmasher as cmr

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 14, "axes.titlesize": 14,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10})

GFC = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs"
SDSS_FULL = f"{GFC}/mpa_rcsed2_combo.fits"
DESI_FULL = f"{GFC}/fastspec_zall_combined.fits"
FLUX_SCALE = 1e-17
Z_MIN, CONT_MIN, LINE_MIN = 0.05, 3.0, 3.0
Msun_r, ZP = 4.64, 0.271
C_PAR, C_SEL = "#9a9a9a", "#CC79A7"
CMAP_Z = getattr(cmr, "bubblegum", "viridis")
BPT_XR, BPT_YR = (-2.0, 1.0), (-2.0, 1.5)

# 8 target lines (for the per-line S/N>3 cut), + Halpha/Hbeta/OIII/NII for BPT
DESI_T = ["OII_3726_FLUX", "OII_3729_FLUX", "HGAMMA_FLUX", "HBETA_FLUX", "OIII_5007_FLUX",
          "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX"]
SDSS_T = ["OII_3726_FLUX", "OII_3729_FLUX", "H_GAMMA_FLUX", "H_BETA_FLUX", "OIII_5007_FLUX",
          "NII_6584_FLUX", "SII_6717_FLUX", "SII_6731_FLUX"]


def log10_lum(z, f1e17):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f1e17, float) * FLUX_SCALE) + np.log10(4*np.pi) + 2*np.log10(dl)


def snr(flux, unc, kind):
    flux = np.asarray(flux, float); unc = np.asarray(unc, float)
    ok = np.isfinite(flux) & np.isfinite(unc) & (flux > 0) & (unc > 0)
    s = np.zeros_like(flux)
    s[ok] = flux[ok] / unc[ok] if kind == "err" else flux[ok] * np.sqrt(unc[ok])
    return s


def process(rec, targets, kind, z, zwarn, cont, logm, galmask,
            ha_f, ha_u, hb_f, o3_f, n2_f):
    n = len(z)
    snr8 = np.zeros((n, 8))
    for i, c in enumerate(targets):
        snr8[:, i] = snr(rec[c], rec[c + ("_ERR" if kind == "err" else "_IVAR")], kind)
    ha_snr = snr(ha_f, ha_u, kind)
    ha_det = ha_snr > 0
    parent = galmask & np.isfinite(z) & (z > 0) & np.isfinite(zwarn) & (zwarn == 0) & np.isfinite(logm)
    sel = parent & (z > Z_MIN) & (cont > CONT_MIN) & (snr8 > LINE_MIN).all(1) & ha_det
    lha = np.full(n, np.nan)
    ok = np.isfinite(z) & (z > 0) & (np.asarray(ha_f, float) > 0)
    lha[ok] = log10_lum(z[ok], np.asarray(ha_f, float)[ok])
    haf = np.log10(np.where(np.asarray(ha_f, float) > 0, np.asarray(ha_f, float), np.nan))
    # BPT
    x = np.full(n, np.nan); y = np.full(n, np.nan)
    ha = np.asarray(ha_f, float); hb = np.asarray(hb_f, float); o3 = np.asarray(o3_f, float); n2 = np.asarray(n2_f, float)
    mx = (n2 > 0) & (ha > 0); my = (o3 > 0) & (hb > 0)
    x[mx] = np.log10(n2[mx] / ha[mx]); y[my] = np.log10(o3[my] / hb[my])
    return dict(z=z.astype(np.float32), logm=logm.astype(np.float32), lha=lha.astype(np.float32),
                haf=haf.astype(np.float32), x=x.astype(np.float32), y=y.astype(np.float32),
                parent=parent, sel=sel, ha_det=ha_det)


def load_sdss():
    t = Table.read(SDSS_FULL, hdu=1)
    z = np.asarray(t["Z_1"], float); zwarn = np.asarray(t["Z_WARNING"], float)
    cont = np.asarray(t["SN_MEDIAN"], float)
    if cont.ndim > 1:
        cont = np.nanmedian(cont, axis=1)
    Mg = np.asarray(t["corrmag_g"], float) - cosmo.distmod(np.clip(z, 1e-4, None)).value - np.asarray(t["kcorr_g"], float)
    Mr = np.asarray(t["corrmag_r"], float) - cosmo.distmod(np.clip(z, 1e-4, None)).value - np.asarray(t["kcorr_r"], float)
    logm = (1.062 * (Mg - Mr) - 0.555) + (-0.4 * (Mr - Msun_r))
    gal = np.ones(len(t), bool)
    if "SPECTROTYPE" in t.colnames:
        gal = (np.char.strip(np.asarray(t["SPECTROTYPE"]).astype(str)) == "GALAXY")
    return process(t, SDSS_T, "err", z, zwarn, cont, logm, gal,
                   t["H_ALPHA_FLUX"], t["H_ALPHA_FLUX_ERR"], t["H_BETA_FLUX"], t["OIII_5007_FLUX"], t["NII_6584_FLUX"])


def load_desi():
    parts = []
    need = DESI_T + [c + "_IVAR" for c in DESI_T] + ["HALPHA_FLUX", "HALPHA_FLUX_IVAR", "Z", "ZWARN",
            "SNR_R", "SURVEY", "PROGRAM", "ABSMAG01_SDSS_G", "ABSMAG01_SDSS_R"]
    with fits.open(DESI_FULL, memmap=True) as hdul:
        for hi in range(1, len(hdul)):
            h = hdul[hi]
            if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                continue
            d = h.data
            if d is None or any(c not in d.names for c in need):
                continue
            z = d["Z"].astype(float); zwarn = d["ZWARN"].astype(float); cont = d["SNR_R"].astype(float)
            g01 = d["ABSMAG01_SDSS_G"].astype(float); r01 = d["ABSMAG01_SDSS_R"].astype(float)
            logm = (1.062 * (g01 - r01) - 0.555) + (-0.4 * (r01 - Msun_r)) + ZP
            sv = np.char.lower(np.char.strip(np.asarray(d["SURVEY"]).astype(str)))
            pg = np.char.lower(np.char.strip(np.asarray(d["PROGRAM"]).astype(str)))
            gal = (sv == "main") & (pg == "bright")
            parts.append(process(d, DESI_T, "ivar", z, zwarn, cont, logm, gal,
                          d["HALPHA_FLUX"], d["HALPHA_FLUX_IVAR"], d["HBETA_FLUX"], d["OIII_5007_FLUX"], d["NII_6584_FLUX"]))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


VARS = [("z", "redshift", None), ("logm", r"$\log(M_\star/M_\odot)$", None),
        ("lha", r"$\log L_{\rm H\alpha}$ [erg s$^{-1}$]", "ha"), ("haf", r"$\log F_{\rm H\alpha}$ [1e-17 cgs]", "ha")]


def main():
    print("loading SDSS...", flush=True); S = load_sdss()
    print("loading DESI (streaming)...", flush=True); D = load_desi()
    DATA = {"SDSS": S, "DESI": D}
    for tag, A in DATA.items():
        print(f"[{tag}] parent={int(A['parent'].sum()):,}  ALT-B selected={int(A['sel'].sum()):,}  "
              f"overall completeness={A['sel'].sum()/max(A['parent'].sum(),1):.1%}", flush=True)

    # ---- FIG1: coverage histograms (normalized), parent vs selected ----
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    for r, tag in enumerate(["SDSS", "DESI"]):
        A = DATA[tag]
        for c, (key, lab, req) in enumerate(VARS):
            par = A["parent"] & (A["ha_det"] if req == "ha" else True)
            v = A[key]
            pv = v[par & np.isfinite(v)]; sv = v[A["sel"] & np.isfinite(v)]
            lo, hi = np.nanpercentile(pv, [0.2, 99.8])
            b = np.linspace(lo, hi, 60)
            ax[r, c].hist(pv, bins=b, density=True, color=C_PAR, alpha=0.55, label=f"parent ({len(pv):,})")
            ax[r, c].hist(sv, bins=b, density=True, histtype="step", lw=2.2, color=C_SEL, label=f"ALT-B ({len(sv):,})")
            ax[r, c].set_xlabel(lab)
            if c == 0:
                ax[r, c].set_ylabel(f"{tag}\nnormalized density"); ax[r, c].legend(fontsize=8)
    for c, (key, lab, req) in enumerate(VARS):
        ax[0, c].set_title(lab.split("[")[0])
    fig.suptitle("Parent vs ALT-B selected training sample: coverage & shape bias (normalized)", fontsize=15)
    fig.savefig("completeness_coverage_hist.png", dpi=200, bbox_inches="tight")
    print("Saved: completeness_coverage_hist.png", flush=True)

    # ---- FIG2: completeness fraction vs each variable ----
    fig2, ax2 = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    for r, tag in enumerate(["SDSS", "DESI"]):
        A = DATA[tag]
        for c, (key, lab, req) in enumerate(VARS):
            par = A["parent"] & (A["ha_det"] if req == "ha" else np.ones(len(A["z"]), bool))
            v = A[key]
            m = par & np.isfinite(v)
            lo, hi = np.nanpercentile(v[m], [0.5, 99.5])
            edges = np.linspace(lo, hi, 31)
            npar, _ = np.histogram(v[m], bins=edges)
            nsel, _ = np.histogram(v[A["sel"] & np.isfinite(v)], bins=edges)
            frac = np.where(npar > 0, nsel / np.maximum(npar, 1), np.nan)
            ctr = 0.5 * (edges[:-1] + edges[1:])
            ax2[r, c].plot(ctr, frac, "o-", color=C_SEL, ms=3)
            ax2[r, c].set_ylim(0, 1.02); ax2[r, c].set_xlabel(lab)
            if c == 0:
                ax2[r, c].set_ylabel(f"{tag}\ncompleteness  N$_{{sel}}$/N$_{{parent}}$")
    for c, (key, lab, req) in enumerate(VARS):
        ax2[0, c].set_title(lab.split("[")[0])
    fig2.suptitle("Selection completeness (ALT-B / parent) vs redshift, mass, L(H$\\alpha$), H$\\alpha$ flux", fontsize=15)
    fig2.savefig("completeness_fraction.png", dpi=200, bbox_inches="tight")
    print("Saved: completeness_fraction.png", flush=True)

    # ---- FIG3: parent BPT colored by z / logM / log Ha flux, ALT-B contours ----
    def contour(axx, xs, ys):
        m = np.isfinite(xs) & np.isfinite(ys)
        H, xe, ye = np.histogram2d(xs[m], ys[m], bins=180, range=[BPT_XR, BPT_YR])
        H = gaussian_filter(H, 1.0).T
        f = np.sort(H[H > 0].ravel())[::-1]; cdf = np.cumsum(f) / f.sum()
        lv = [f[min(np.searchsorted(cdf, q), len(f)-1)] for q in (0.95, 0.68)]
        axx.contour(0.5*(xe[:-1]+xe[1:]), 0.5*(ye[:-1]+ye[1:]), H, levels=lv, colors="k", linewidths=1.6)

    fig3, ax3 = plt.subplots(2, 3, figsize=(16, 10.5), constrained_layout=True)
    colspec = [("z", "median redshift", CMAP_Z), ("logm", r"median $\log M_\star$", CMAP_Z),
               ("haf", r"median $\log F_{\rm H\alpha}$", CMAP_Z)]
    for r, tag in enumerate(["SDSS", "DESI"]):
        A = DATA[tag]; par = A["parent"]
        for c, (key, lab, cmap) in enumerate(colspec):
            x, y, C = A["x"][par], A["y"][par], A[key][par]
            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(C)
            hb = ax3[r, c].hexbin(x[m], y[m], C=C[m], reduce_C_function=np.median, gridsize=80,
                                  extent=(*BPT_XR, *BPT_YR), cmap=cmap, mincnt=8)
            contour(ax3[r, c], A["x"][A["sel"]], A["y"][A["sel"]])
            fig3.colorbar(hb, ax=ax3[r, c], shrink=0.85).set_label(lab)
            ax3[r, c].set_xlim(*BPT_XR); ax3[r, c].set_ylim(*BPT_YR); ax3[r, c].set_box_aspect(1)
            ax3[r, c].set_xlabel(r"$\log([\mathrm{NII}]/\mathrm{H}\alpha)$")
            if c == 0:
                ax3[r, c].set_ylabel(f"{tag}\n" + r"$\log([\mathrm{OIII}]/\mathrm{H}\beta)$")
    ax3[0, 0].plot([], [], "k-", lw=1.6, label="ALT-B 68/95%"); ax3[0, 0].legend(loc="lower left")
    fig3.suptitle("Parent BPT colored by z / mass / H$\\alpha$ flux; ALT-B selected contours overlaid", fontsize=15)
    fig3.savefig("completeness_bpt.png", dpi=200, bbox_inches="tight")
    print("Saved: completeness_bpt.png", flush=True)


if __name__ == "__main__":
    main()
