# completeness_hist_v3.py
# Revision of completeness_hist_v2.py addressing collaborator comments:
#   (1) distinct colors for the SDSS vs DESI selected samples (was light pink for both);
#       adopts the paper-wide Okabe-Ito survey convention  SDSS=#0072B2, DESI=#E69F00
#   (2) shared x-axis limits per column across the two survey rows, so positions/offsets
#       between the samples can be read off directly
#   (3) an unnormalized version (absolute galaxy counts) alongside the normalized one
#
# Outputs (figs_ALTB/):
#   completeness_coverage_hist_v3.png/.pdf              normalized density   [primary]
#   completeness_coverage_hist_v3_counts.png/.pdf       dN/dx, log y         [unnormalized, cross-row comparable]
#   completeness_coverage_hist_v3_rawcounts.png/.pdf    raw N, shared bin edges, log y
#   completeness_cache.npz                              processed arrays, so re-plots skip the FITS streaming
#
# Parent = GALAXY(main/bright DESI; SPECTROTYPE SDSS) & ZWARN=0 & z>0 & finite LOGM_COLOR.
# Selection = parent & z>0.05 & continuum S/N>3 & 8-target-line S/N>3 & Halpha detected.

import os
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa
import cmasher as cmr  # noqa

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 17, "axes.titlesize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 16})

REPO = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model"
OUTDIR = f"{REPO}/figs_ALTB"
CACHE = f"{OUTDIR}/completeness_cache.npz"
FORCE = os.environ.get("FORCE_RELOAD", "0") == "1"

GFC = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs"
SDSS_FULL = f"{GFC}/mpa_rcsed2_combo.fits"
DESI_FULL = f"{GFC}/fastspec_zall_combined.fits"
FLUX_SCALE = 1e-17
Z_MIN, CONT_MIN, LINE_MIN = 0.05, 3.0, 3.0
Msun_r, ZP = 4.64, 0.271

C_PAR = "#9a9a9a"
C_SEL = {"SDSS": "#0072B2", "DESI": "#E69F00"}   # Okabe-Ito, matches figs 4/5/6 of the paper
PARLAB = {"SDSS": "SDSS MGS", "DESI": "DESI BGS"}

DESI_T = ["OII_3726_FLUX", "OII_3729_FLUX", "HGAMMA_FLUX", "HBETA_FLUX", "OIII_5007_FLUX",
          "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX"]
SDSS_T = ["OII_3726_FLUX", "OII_3729_FLUX", "H_GAMMA_FLUX", "H_BETA_FLUX", "OIII_5007_FLUX",
          "NII_6584_FLUX", "SII_6717_FLUX", "SII_6731_FLUX"]

VARS3 = [("z", "redshift"),
         ("logm", r"$\log(M_\star/M_\odot)$"),
         ("lha", r"$\log L_{\rm H\alpha}$ [erg s$^{-1}$]")]

# Shared display ranges, clipped to where both *selected* samples live. The raw parent
# percentile ranges (z to 1.68, logM to 6.4, logLHa to 31.9) are dominated by sparse
# tails that are almost certainly catastrophic-z / bad-flux objects, and letting them
# set the axis compresses the SDSS panels to a few percent of the frame.
DISPLAY = {"z": (0.0, 0.60), "logm": (7.5, 12.0), "lha": (37.5, 42.5)}


def log10_lum(z, f1e17):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f1e17, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def snr(flux, unc, kind):
    flux = np.asarray(flux, float); unc = np.asarray(unc, float)
    ok = np.isfinite(flux) & np.isfinite(unc) & (flux > 0) & (unc > 0)
    s = np.zeros_like(flux)
    s[ok] = flux[ok] / unc[ok] if kind == "err" else flux[ok] * np.sqrt(unc[ok])
    return s


def process(rec, targets, kind, z, zwarn, cont, logm, galmask, ha_f, ha_u):
    n = len(z)
    snr8 = np.zeros((n, 8))
    for i, c in enumerate(targets):
        snr8[:, i] = snr(rec[c], rec[c + ("_ERR" if kind == "err" else "_IVAR")], kind)
    ha_det = snr(ha_f, ha_u, kind) > 0
    parent = galmask & np.isfinite(z) & (z > 0) & np.isfinite(zwarn) & (zwarn == 0) & np.isfinite(logm)
    sel = parent & (z > Z_MIN) & (cont > CONT_MIN) & (snr8 > LINE_MIN).all(1) & ha_det
    lha = np.full(n, np.nan)
    ok = np.isfinite(z) & (z > 0) & (np.asarray(ha_f, float) > 0)
    lha[ok] = log10_lum(z[ok], np.asarray(ha_f, float)[ok])
    return dict(z=z.astype(np.float32), logm=logm.astype(np.float32), lha=lha.astype(np.float32),
                parent=parent, sel=sel, ha_det=ha_det)


def load_sdss():
    t = Table.read(SDSS_FULL, hdu=1)
    z = np.asarray(t["Z_1"], float); zwarn = np.asarray(t["Z_WARNING"], float)
    cont = np.asarray(t["SN_MEDIAN"], float)
    if cont.ndim > 1:
        cont = np.nanmedian(cont, axis=1)
    dm = cosmo.distmod(np.clip(z, 1e-4, None)).value
    Mg = np.asarray(t["corrmag_g"], float) - dm - np.asarray(t["kcorr_g"], float)
    Mr = np.asarray(t["corrmag_r"], float) - dm - np.asarray(t["kcorr_r"], float)
    logm = (1.062 * (Mg - Mr) - 0.555) + (-0.4 * (Mr - Msun_r))
    gal = np.ones(len(t), bool)
    if "SPECTROTYPE" in t.colnames:
        gal = (np.char.strip(np.asarray(t["SPECTROTYPE"]).astype(str)) == "GALAXY")
    return process(t, SDSS_T, "err", z, zwarn, cont, logm, gal,
                   t["H_ALPHA_FLUX"], t["H_ALPHA_FLUX_ERR"])


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
                                 d["HALPHA_FLUX"], d["HALPHA_FLUX_IVAR"]))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def get_data():
    if os.path.exists(CACHE) and not FORCE:
        print(f"loading cache {CACHE}", flush=True)
        z = np.load(CACHE)
        return {tag: {k: z[f"{tag}_{k}"] for k in ("z", "logm", "lha", "parent", "sel", "ha_det")}
                for tag in ("SDSS", "DESI")}
    print("loading SDSS...", flush=True); S = load_sdss()
    print("loading DESI (streaming 323 HDUs)...", flush=True); D = load_desi()
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **{f"SDSS_{k}": v for k, v in S.items()},
                                **{f"DESI_{k}": v for k, v in D.items()})
    print(f"cached -> {CACHE}", flush=True)
    return {"SDSS": S, "DESI": D}


def ranges(DATA):
    """Per-column parent range for each survey, plus the shared (union) range."""
    out = {}
    for c, (key, lab) in enumerate(VARS3):
        per = {}
        for tag in ("SDSS", "DESI"):
            A = DATA[tag]
            par = A["parent"] & (A["ha_det"] if key == "lha" else np.ones(len(A["z"]), bool))
            v = A[key][par & np.isfinite(A[key])]
            per[tag] = tuple(np.nanpercentile(v, [0.2, 99.8]))
        shared = DISPLAY[key]
        out[key] = dict(per=per, shared=shared)
        print(f"  [{key:5s}] SDSS={per['SDSS'][0]:.3f}..{per['SDSS'][1]:.3f}  "
              f"DESI={per['DESI'][0]:.3f}..{per['DESI'][1]:.3f}  shared={shared[0]:.3f}..{shared[1]:.3f}",
              flush=True)
    return out


def masks(A, key):
    par = A["parent"] & (A["ha_det"] if key == "lha" else np.ones(len(A["z"]), bool))
    v = A[key]
    return v[par & np.isfinite(v)], v[A["sel"] & np.isfinite(v)]


def make_fig(DATA, R, mode, fname, logy=False, shared_edges=False):
    """mode: 'density' | 'dndx' | 'raw'"""
    fig, ax = plt.subplots(2, 3, figsize=(15.0, 8.6), constrained_layout=True)
    for r, tag in enumerate(["SDSS", "DESI"]):
        A = DATA[tag]
        for c, (key, lab) in enumerate(VARS3):
            pv, sv = masks(A, key)
            lo, hi = R[key]["shared"]
            b = np.linspace(lo, hi, 60)
            bw = b[1] - b[0]
            if mode == "density":
                kw_p = dict(density=True); kw_s = dict(density=True)
            elif mode == "dndx":
                kw_p = dict(weights=np.full(pv.size, 1.0 / bw))
                kw_s = dict(weights=np.full(sv.size, 1.0 / bw))
            else:
                kw_p = {}; kw_s = {}
            ax[r, c].hist(pv, bins=b, color=C_PAR, alpha=0.55,
                          label=PARLAB[tag], **kw_p)
            ax[r, c].hist(sv, bins=b, histtype="step", lw=2.6, color=C_SEL[tag],
                          label="our selection", **kw_s)
            ax[r, c].set_xlim(*R[key]["shared"])          # <-- shared x across rows
            if logy:
                ax[r, c].set_yscale("log")
            ax[r, c].set_xlabel(lab)
            ax[r, c].tick_params(labelsize=15)
            if c == 0:
                ylab = {"density": "normalized density",
                        "dndx": "galaxies per unit $x$",
                        "raw": "galaxies per bin"}[mode]
                ax[r, c].set_ylabel(f"{tag}\n{ylab}")
            ax[r, c].legend(loc="best")
    out = f"{OUTDIR}/{fname}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out, flush=True)
    return out


def main():
    DATA = get_data()
    for tag, A in DATA.items():
        p, s = int(A["parent"].sum()), int(A["sel"].sum())
        print(f"[{tag}] parent={p:,}  selected={s:,}  completeness={s/max(p,1):.1%}  "
              f"median z par={np.nanmedian(A['z'][A['parent']]):.3f} sel={np.nanmedian(A['z'][A['sel']]):.3f}  "
              f"median logM sel={np.nanmedian(A['logm'][A['sel']]):.3f}  "
              f"median logLHa sel={np.nanmedian(A['lha'][A['sel']]):.3f}", flush=True)
    print("axis ranges:", flush=True)
    R = ranges(DATA)
    make_fig(DATA, R, "density", "completeness_coverage_hist_v4.png")
    make_fig(DATA, R, "dndx", "completeness_coverage_hist_v4_counts.png", logy=True)
    make_fig(DATA, R, "raw", "completeness_coverage_hist_v4_rawcounts.png", logy=True, shared_edges=True)
    print("=== ALL FIGURES DONE ===", flush=True)


if __name__ == "__main__":
    main()
