# snr_by_line.py  (Part A of Balmer/Hgamma verification)
# Per-line S/N distributions, DESI vs SDSS, full ALT-B samples.
# Tests the premise that DESI's Balmer lines (esp Hgamma) are noisier than SDSS's,
# which would let the flow soften the DESI Balmer-coupling structure.
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 15, "axes.titlesize": 15, "legend.fontsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 13})

# ordered blue -> red so wavelength trend is visible
LINES = ["OII_3726", "OII_3729", "H_GAMMA", "H_BETA", "OIII_5007", "H_ALPHA", "NII_6584", "SII_6717", "SII_6731"]
LAB = ["[OII]3726", "[OII]3729", "Hγ", "Hβ", "[OIII]5007", "Hα", "[NII]6584", "[SII]6717", "[SII]6731"]
DESI_MAP = {"H_GAMMA": "HGAMMA", "H_BETA": "HBETA", "H_ALPHA": "HALPHA", "SII_6717": "SII_6716"}


def snr_cols(fits, kind):
    t = Table.read(fits, hdu=1)
    out = {}
    for ln in LINES:
        col = DESI_MAP.get(ln, ln) if kind == "ivar" else ln
        f = np.asarray(t[col + "_FLUX"], float)
        if kind == "err":
            u = np.asarray(t[col + "_FLUX_ERR"], float)
            ok = np.isfinite(f) & np.isfinite(u) & (u > 0)
            s = np.where(ok, f / u, np.nan)
        else:
            iv = np.asarray(t[col + "_FLUX_IVAR"], float)
            ok = np.isfinite(f) & np.isfinite(iv) & (iv > 0)
            s = np.where(ok, f * np.sqrt(iv), np.nan)
        out[ln] = s
    return out, len(t)


def main():
    sdss, nS = snr_cols(BASE + "SDSS_main_training_data_ALTB.fits", "err")
    desi, nD = snr_cols(BASE + "DESI_BGS_training_data_ALTB.fits", "ivar")
    print(f"SDSS N={nS:,}   DESI N={nD:,}\n")
    print(f"{'line':12s} {'SDSS med':>9s} {'SDSS 16-84':>16s} {'DESI med':>9s} {'DESI 16-84':>16s} {'DESI/SDSS med':>13s}")
    med = {"SDSS": [], "DESI": []}
    for ln, lab in zip(LINES, LAB):
        sS = sdss[ln][np.isfinite(sdss[ln])]; sD = desi[ln][np.isfinite(desi[ln])]
        mS, mD = np.median(sS), np.median(sD)
        pS = np.percentile(sS, [16, 84]); pD = np.percentile(sD, [16, 84])
        med["SDSS"].append(mS); med["DESI"].append(mD)
        print(f"{lab:12s} {mS:9.1f} [{pS[0]:6.1f},{pS[1]:7.1f}] {mD:9.1f} [{pD[0]:6.1f},{pD[1]:7.1f}] {mD/mS:13.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), constrained_layout=True)
    # (a) median S/N per line, grouped bars
    x = np.arange(len(LINES)); w = 0.38
    axes[0].bar(x - w/2, med["SDSS"], w, label="SDSS", color="#d1495b")
    axes[0].bar(x + w/2, med["DESI"], w, label="DESI", color="#2e86ab")
    axes[0].set_yscale("log"); axes[0].set_xticks(x); axes[0].set_xticklabels(LAB, rotation=45, ha="right")
    axes[0].set_ylabel("median per-line S/N"); axes[0].axhline(3, ls="--", color="0.4", lw=1)
    axes[0].set_title("Median line S/N (dashed = cut at 3)"); axes[0].legend()
    # (b) Hgamma & Hbeta S/N histograms
    bins = np.linspace(0, 40, 60)
    for ln, c, style in [("H_GAMMA", "#e67e22", "-"), ("H_BETA", "#8e44ad", "--")]:
        axes[1].hist(sdss[ln][np.isfinite(sdss[ln])], bins=bins, density=True, histtype="step",
                     lw=2, ls=style, color=c, label=f"SDSS {('Hγ' if 'GAMMA' in ln else 'Hβ')}")
        axes[1].hist(desi[ln][np.isfinite(desi[ln])], bins=bins, density=True, histtype="step",
                     lw=2, ls=style, color="black" if "GAMMA" in ln else "#2e86ab",
                     label=f"DESI {('Hγ' if 'GAMMA' in ln else 'Hβ')}")
    axes[1].axvline(3, ls="--", color="0.4", lw=1); axes[1].set_xlabel("per-line S/N")
    axes[1].set_ylabel("normalized density"); axes[1].set_title("Balmer S/N: DESI vs SDSS"); axes[1].legend()
    out = REPO + "figs_ALTB/snr_by_line.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("\nSaved:", out, flush=True)

    # save per-line median S/N in the correlation-matrix line order (LAB8, first 8 rows)
    key = {"H_BETA": "Hbeta", "H_GAMMA": "Hgamma", "NII_6584": "NII", "SII_6717": "SIIa",
           "SII_6731": "SIIb", "OII_3726": "OIIa", "OII_3729": "OIIb", "OIII_5007": "OIII"}
    order = ["Hbeta", "Hgamma", "NII", "SIIa", "SIIb", "OIIa", "OIIb", "OIII"]  # == LAB8
    lut_s = {key[ln]: med["SDSS"][i] for i, ln in enumerate(LINES) if ln in key}
    lut_d = {key[ln]: med["DESI"][i] for i, ln in enumerate(LINES) if ln in key}
    np.savez(REPO + "figs_ALTB/snr_medians.npz",
             names=np.array(order), sdss=np.array([lut_s[k] for k in order]),
             desi=np.array([lut_d[k] for k in order]))
    print("Saved: snr_medians.npz", flush=True)


if __name__ == "__main__":
    main()
