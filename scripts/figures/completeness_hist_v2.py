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
plt.rcParams.update({"axes.labelsize": 17, "axes.titlesize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 16})
OUT = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/figs_ALTB/completeness_coverage_hist.png"

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
    PARLAB = {"SDSS": "SDSS MGS", "DESI": "DESI BGS"}
    for tag, A in DATA.items():
        print(f"[{tag}] parent={int(A['parent'].sum()):,}  selected={int(A['sel'].sum()):,}  "
              f"completeness={A['sel'].sum()/max(A['parent'].sum(),1):.1%}  "
              f"median z par={np.nanmedian(A['z'][A['parent']]):.3f} sel={np.nanmedian(A['z'][A['sel']]):.3f}",
              flush=True)

    # ---- coverage histograms (normalized): redshift, stellar mass, L(Halpha) only ----
    vars3 = VARS[:3]  # (z, logM*, logL_Ha)
    fig, ax = plt.subplots(2, 3, figsize=(15.0, 8.6), constrained_layout=True)
    for r, tag in enumerate(["SDSS", "DESI"]):
        A = DATA[tag]
        for c, (key, lab, req) in enumerate(vars3):
            par = A["parent"] & (A["ha_det"] if req == "ha" else np.ones(len(A["z"]), bool))
            v = A[key]
            pv = v[par & np.isfinite(v)]; sv = v[A["sel"] & np.isfinite(v)]
            lo, hi = np.nanpercentile(pv, [0.2, 99.8])
            b = np.linspace(lo, hi, 60)
            ax[r, c].hist(pv, bins=b, density=True, color=C_PAR, alpha=0.55, label=PARLAB[tag])
            ax[r, c].hist(sv, bins=b, density=True, histtype="step", lw=2.6, color=C_SEL, label="our selection")
            ax[r, c].set_xlabel(lab)
            ax[r, c].tick_params(labelsize=15)
            if c == 0:
                ax[r, c].set_ylabel(f"{tag}\nnormalized density")
                ax[r, c].legend(loc="best")
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
    print("Saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
