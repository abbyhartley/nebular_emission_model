# var_C_detection_diag.py
#
# Variation C diagnostic: among galaxies with Halpha S/N>7 (+ continuum S/N>3,
# z>0.05, ZWARN=0, GALAXY), for each OTHER line report the fraction detected at all
# (S/N>0), at S/N>3, and at S/N>5. 3-column figure: per-line detection bars,
# stellar mass, redshift. Caches the selected-sample arrays for fast re-plots.

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 15, "axes.titlesize": 15,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10})

GFC = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs"
SDSS_FULL = f"{GFC}/mpa_rcsed2_combo.fits"
DESI_FULL = f"{GFC}/fastspec_zall_combined.fits"
CACHE = Path("var_C_diag_cache.npz")

LINE_LABELS = ["[OII]3726", "[OII]3729", "H$\\gamma$", "H$\\beta$", "H$\\alpha$",
               "[OIII]5007", "[NII]6584", "[SII]6716", "[SII]6731"]
NONHA = [0, 1, 2, 3, 5, 6, 7, 8]
NONHA_LABELS = [LINE_LABELS[i] for i in NONHA]
HA_I = 4
SDSS_FLUX = ["OII_3726_FLUX", "OII_3729_FLUX", "H_GAMMA_FLUX", "H_BETA_FLUX", "H_ALPHA_FLUX",
             "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6717_FLUX", "SII_6731_FLUX"]
DESI_FLUX = ["OII_3726_FLUX", "OII_3729_FLUX", "HGAMMA_FLUX", "HBETA_FLUX", "HALPHA_FLUX",
             "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX"]

Z_MIN, CONT_MIN, HA_MIN = 0.05, 3.0, 7.0
C_DET, C_S3, C_S5 = "#009E73", "#0072B2", "#D55E00"
C_ALL, C_MISS = "#CC79A7", "#D55E00"


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


def selected_arrays(rec, flux_cols, kind, z, zwarn, logm, cont, base_extra=True):
    """Return (snr8, logm, z) for the Halpha-selected sample only."""
    snr9 = per_line_snr(rec, flux_cols, kind)
    base = np.isfinite(z) & (z > 0) & np.isfinite(zwarn) & (zwarn == 0) & base_extra
    sel = base & (z > Z_MIN) & (cont > CONT_MIN) & (snr9[:, HA_I] > HA_MIN)
    return snr9[sel][:, NONHA].astype(np.float32), logm[sel].astype(np.float32), z[sel].astype(np.float32)


def build_cache():
    data = {}
    # SDSS
    t = Table.read(SDSS_FULL, hdu=1)
    z = np.asarray(t["Z_1"], float); zwarn = np.asarray(t["Z_WARNING"], float)
    logm = np.asarray(t["LGM_TOT_P50"], float)
    cont = np.asarray(t["SN_MEDIAN"], float)
    if cont.ndim > 1:
        cont = np.nanmedian(cont, axis=1)
    spok = np.ones(len(t), bool)
    if "SPECTROTYPE" in t.colnames:
        spok = (np.char.strip(np.asarray(t["SPECTROTYPE"]).astype(str)) == "GALAXY")
    s8, sl, sz = selected_arrays(t, SDSS_FLUX, "err", z, zwarn, logm, cont, base_extra=spok)
    data["SDSS_snr8"], data["SDSS_logm"], data["SDSS_z"] = s8, sl, sz
    print(f"SDSS Halpha-selected: {len(sz):,}", flush=True)
    # DESI (stream)
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
            parts.append(selected_arrays(d, DESI_FLUX, "ivar", z, zwarn, logm, cont,
                                         base_extra=(sv == "main") & (pg == "bright")))
    data["DESI_snr8"] = np.concatenate([p[0] for p in parts])
    data["DESI_logm"] = np.concatenate([p[1] for p in parts])
    data["DESI_z"] = np.concatenate([p[2] for p in parts])
    print(f"DESI Halpha-selected: {len(data['DESI_z']):,}", flush=True)
    np.savez(CACHE, **data)
    return data


def main():
    if CACHE.exists():
        print("loading cached arrays", flush=True)
        d = np.load(CACHE); DATA = {k: d[k] for k in d.files}
    else:
        print("building cache (streaming DESI)...", flush=True)
        DATA = build_cache()

    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for r, tag in enumerate(["SDSS", "DESI"]):
        snr = DATA[f"{tag}_snr8"]; logm = DATA[f"{tag}_logm"]; zz = DATA[f"{tag}_z"]
        N = len(zz)
        det0, det3, det5 = snr > 0, snr > 3, snr > 5
        f0, f3, f5 = det0.mean(0), det3.mean(0), det5.mean(0)
        all8 = det0.all(1); miss = ~all8
        print(f"\n[{tag}] Halpha-selected N={N:,}  all-8 detected(S/N>0) {all8.mean():.0%}  "
              f"all-8 S/N>3 {det3.all(1).mean():.0%}  all-8 S/N>5 {det5.all(1).mean():.0%}", flush=True)
        for i, li in enumerate(NONHA):
            print(f"     {LINE_LABELS[li]:11s} det(>0) {f0[i]:.0%}  >3 {f3[i]:.0%}  >5 {f5[i]:.0%}", flush=True)

        # col0: per-line detection bars (grouped: >0, >3, >5)
        yp = np.arange(8); hh = 0.26
        a0 = ax[r, 0]
        a0.barh(yp - hh, f0, height=hh, color=C_DET, label="detected (S/N>0)")
        a0.barh(yp, f3, height=hh, color=C_S3, label="S/N>3")
        a0.barh(yp + hh, f5, height=hh, color=C_S5, label="S/N>5")
        a0.set_yticks(yp); a0.set_yticklabels(NONHA_LABELS); a0.invert_yaxis()
        a0.set_xlim(0, 1); a0.set_xlabel(r"fraction of H$\alpha$-selected")
        a0.set_ylabel(f"{tag}")
        if r == 0:
            # legend ABOVE the panel (outside the bars), horizontal
            a0.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=True,
                      handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)

        # col1: stellar mass, all-8-detected vs missing >=1
        mb = np.linspace(7.5, 12.5, 55)
        a1 = ax[r, 1]
        for msk, col, lab in [(all8, C_ALL, "all 8 detected"), (miss, C_MISS, "$\\geq$1 undetected")]:
            v = logm[msk]; v = v[np.isfinite(v) & (v > 6) & (v < 13)]
            a1.hist(v, bins=mb, density=True, histtype="step", lw=2, color=col, label=lab)
        a1.set_xlabel(r"$\log(M_\star/M_\odot)$"); a1.legend()

        # col2: redshift, same split
        zb = np.linspace(0, np.nanpercentile(zz, 99.5), 55)
        a2 = ax[r, 2]
        for msk, col, lab in [(all8, C_ALL, "all 8 detected"), (miss, C_MISS, "$\\geq$1 undetected")]:
            a2.hist(zz[msk], bins=zb, density=True, histtype="step", lw=2, color=col, label=lab)
        a2.set_xlabel("redshift"); a2.legend()

    ax[0, 1].set_title("stellar mass"); ax[0, 2].set_title("redshift")
    fig.suptitle(r"Variation C: among H$\alpha$ S/N>7 galaxies, per-line detection fraction (S/N>0, >3, >5) and the populations missing a line",
                 fontsize=13)
    fig.savefig("var_C_missing_lines.png", dpi=205, bbox_inches="tight")
    print("\nSaved: var_C_missing_lines.png", flush=True)


if __name__ == "__main__":
    main()
