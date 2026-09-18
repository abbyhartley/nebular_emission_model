# phase1_selection_combos.py
#
# Phase 1: quantify alternative "quality-gate" selections that drop the per-line S/N cut.
# Gates: z>0.05, ZWARN=0, GALAXY(SDSS)/main-bright(DESI), continuum S/N>3, Halpha S/N>THR
#        (+ SDSS r<17.77; DESI apparent r not available -> continuum is the brightness gate).
# For THR in {7,10,15,20,30}: how many of the 8 lines are detected(S/N>0)/at S/N>3/>5.
# Compare mass (LOGMSTAR, consistent from parent) & redshift coverage of the loose sample
# (cont>3, Halpha>7) vs the ORIGINAL STRICT selection (cont>5 & all-9 per-line>5), both
# recomputed on the parent so the mass definition is identical.

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 14, "axes.titlesize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10})

GFC = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs"
SDSS_FULL = f"{GFC}/mpa_rcsed2_combo.fits"
DESI_FULL = f"{GFC}/fastspec_zall_combined.fits"

LINE_LABELS = ["[OII]3726", "[OII]3729", "H$\\gamma$", "H$\\beta$", "H$\\alpha$",
               "[OIII]5007", "[NII]6584", "[SII]6716", "[SII]6731"]
NONHA = [0, 1, 2, 3, 5, 6, 7, 8]
HA_I = 4
SDSS_FLUX = ["OII_3726_FLUX", "OII_3729_FLUX", "H_GAMMA_FLUX", "H_BETA_FLUX", "H_ALPHA_FLUX",
             "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6717_FLUX", "SII_6731_FLUX"]
DESI_FLUX = ["OII_3726_FLUX", "OII_3729_FLUX", "HGAMMA_FLUX", "HBETA_FLUX", "HALPHA_FLUX",
             "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX"]
Z_MIN = 0.05
HA_THRS = [7, 10, 15, 20, 30]
C_ALL, C_LOOSE, C_STRICT = "#CC79A7", "#0072B2", "#D55E00"


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


def process(rec, flux_cols, kind, z, zwarn, logm, cont, base_extra=True, rmag=None, rlim=None):
    """Return loose-pool arrays (snr9,logm,z) [cont>3 & Halpha>7 & z>0.05 (+rmag)]
    and strict-pool arrays (logm,z) [cont>5 & all-9 per-line>5 & z>0.05]."""
    snr9 = per_line_snr(rec, flux_cols, kind)
    base = np.isfinite(z) & (z > 0) & np.isfinite(zwarn) & (zwarn == 0) & base_extra & (z > Z_MIN)
    if rmag is not None and rlim is not None:
        base = base & np.isfinite(rmag) & (rmag < rlim)
    loose = base & (cont > 3) & (snr9[:, HA_I] > 7)
    strict = base & (cont > 5) & (snr9.min(1) > 5)
    return (snr9[loose].astype(np.float32), logm[loose].astype(np.float32), z[loose].astype(np.float32),
            logm[strict].astype(np.float32), z[strict].astype(np.float32))


def load_sdss():
    t = Table.read(SDSS_FULL, hdu=1)
    z = np.asarray(t["Z_1"], float); zwarn = np.asarray(t["Z_WARNING"], float)
    logm = np.asarray(t["LGM_TOT_P50"], float)
    cont = np.asarray(t["SN_MEDIAN"], float)
    if cont.ndim > 1:
        cont = np.nanmedian(cont, axis=1)
    rmag = np.asarray(t["corrmag_r"], float)
    spok = np.ones(len(t), bool)
    if "SPECTROTYPE" in t.colnames:
        spok = (np.char.strip(np.asarray(t["SPECTROTYPE"]).astype(str)) == "GALAXY")
    return process(t, SDSS_FLUX, "err", z, zwarn, logm, cont, base_extra=spok, rmag=rmag, rlim=17.77)


def load_desi():
    out = [[], [], [], [], []]
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
            res = process(d, DESI_FLUX, "ivar", z, zwarn, logm, cont, base_extra=(sv == "main") & (pg == "bright"))
            for k in range(5):
                out[k].append(res[k])
    return tuple(np.concatenate(o) for o in out)


def main():
    print("loading SDSS...", flush=True); S = load_sdss()
    print("loading DESI (streaming)...", flush=True); D = load_desi()
    DATA = {"SDSS": S, "DESI": D}

    # ---- stdout table ----
    for tag, (snrL, lmL, zL, lmS, zS) in DATA.items():
        print(f"\n===== {tag} =====", flush=True)
        print(f"  STRICT (cont>5 & all-9 per-line>5): N={len(zS):,}", flush=True)
        haL = snrL[:, HA_I]
        for thr in HA_THRS:
            sub = haL > thr
            s8 = snrL[sub][:, NONHA]
            n = int(sub.sum())
            a0 = (s8 > 0).all(1).mean() if n else 0
            a3 = (s8 > 3).all(1).mean() if n else 0
            a5 = (s8 > 5).all(1).mean() if n else 0
            print(f"  cont>3 & Halpha>{thr:>2d}: N={n:,}  all-8 det(>0)={a0:.0%}  all-8 S/N>3={a3:.0%}  all-8 S/N>5={a5:.0%}", flush=True)
        # per-line detail at the requested Halpha>7
        sub = haL > 7; s8 = snrL[sub][:, NONHA]
        print(f"  --- per-line at cont>3 & Halpha>7 (N={int(sub.sum()):,}) ---", flush=True)
        for i, li in enumerate(NONHA):
            print(f"      {LINE_LABELS[li]:11s} det(>0) {(s8[:,i]>0).mean():.0%}  >3 {(s8[:,i]>3).mean():.0%}  >5 {(s8[:,i]>5).mean():.0%}", flush=True)

    # ---- figure: detection vs Halpha threshold; mass & z coverage loose vs strict ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for r, tag in enumerate(["SDSS", "DESI"]):
        snrL, lmL, zL, lmS, zS = DATA[tag]
        haL = snrL[:, HA_I]
        thrs = np.array(HA_THRS, float)
        f0 = []; f3 = []; f5 = []; ns = []
        for thr in HA_THRS:
            sub = haL > thr; s8 = snrL[sub][:, NONHA]; ns.append(int(sub.sum()))
            f0.append((s8 > 0).all(1).mean()); f3.append((s8 > 3).all(1).mean()); f5.append((s8 > 5).all(1).mean())
        a0 = ax[r, 0]
        a0.plot(thrs, f0, "o-", color=C_ALL, label="all 8 detected (S/N>0)")
        a0.plot(thrs, f3, "s-", color=C_LOOSE, label="all 8 at S/N>3")
        a0.plot(thrs, f5, "^-", color=C_STRICT, label="all 8 at S/N>5")
        a0.set_xlabel(r"H$\alpha$ S/N threshold"); a0.set_ylabel(f"{tag}\nfraction of selected"); a0.set_ylim(0, 1)
        if r == 0:
            a0.legend(loc="lower right")
        # mass coverage
        mb = np.linspace(7.5, 12.5, 55)
        a1 = ax[r, 1]
        for v, col, lab in [(lmS, C_STRICT, f"strict (N={len(lmS):,})"), (lmL, C_LOOSE, f"loose cont>3,H$\\alpha$>7 (N={len(lmL):,})")]:
            vv = v[np.isfinite(v) & (v > 6) & (v < 13)]
            a1.hist(vv, bins=mb, density=True, histtype="step", lw=2, color=col, label=lab)
        a1.set_xlabel(r"$\log(M_\star/M_\odot)$ [spectral]"); a1.legend()
        # redshift coverage
        a2 = ax[r, 2]
        zmax = np.nanpercentile(np.concatenate([zS, zL]), 99.5)
        zb = np.linspace(0, zmax, 55)
        for v, col, lab in [(zS, C_STRICT, "strict"), (zL, C_LOOSE, r"loose cont>3,H$\alpha$>7")]:
            a2.hist(v, bins=zb, density=True, histtype="step", lw=2, color=col, label=lab)
        a2.set_xlabel("redshift"); a2.legend()
    ax[0, 0].set_title(r"all-8 detection vs H$\alpha$ cut")
    ax[0, 1].set_title("stellar-mass coverage"); ax[0, 2].set_title("redshift coverage")
    fig.suptitle("Phase 1: dropping the per-line cut for quality gates (continuum S/N>3 + H-alpha S/N) "
                 "recovers the massive/high-z tail; how many lines stay detected", fontsize=13)
    fig.savefig("phase1_selection_combos.png", dpi=205, bbox_inches="tight")
    print("\nSaved: phase1_selection_combos.png", flush=True)


if __name__ == "__main__":
    main()
