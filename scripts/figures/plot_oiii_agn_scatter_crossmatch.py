"""
Cross-matched (same galaxy, two spectra) version of the [O III] AGN-scatter test.

Uses docs/crossmatch_sdss_desi_fluxes.csv (SDSS & DESI fluxes for the SAME galaxies).
Instead of an NF prediction residual, the cross-survey [O III] "residual" is the
DIRECT per-object difference:
    delta = log10([OIII]/Ha)_DESI  -  log10([OIII]/Ha)_SDSS
i.e. how differently the two surveys measure the Ha-normalized [O III] for the very
same galaxy. We also BPT-classify each matched galaxy independently in EACH survey.

Panels:
  (0,0) delta histogram split by SDSS BPT class (SF/composite/AGN) + per-class sigma
  (0,1) SDSS->DESI class migration matrix (does DESI push galaxies toward AGN?)
  (1,0) SDSS NII-BPT plane, hexbin-colored by median |delta|
  (1,1) DESI NII-BPT plane, hexbin-colored by median |delta|
Also prints per-class scatter/bias and the SDSS vs DESI AGN/composite/SF fractions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa
try:
    import cmasher as cmr; CMAP = cmr.bubblegum
except Exception:
    CMAP = "magma"

REPO = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/"
CSV = REPO + "docs/crossmatch_sdss_desi_fluxes.csv"
CLASS_COL = {"SF": "#0072B2", "composite": "#E69F00", "AGN": "#CC79A7"}
CLASSES = ["SF", "composite", "AGN"]


def classify(nii6584, ha, oiii, hb):
    """NII-BPT class per galaxy: 0=SF, 1=composite, 2=AGN, -1=undefined.
    Kauffmann03 (SF/comp) and Kewley01 (comp/AGN) demarcations."""
    nii6584, ha, oiii, hb = map(lambda a: np.asarray(a, float), (nii6584, ha, oiii, hb))
    v = (nii6584 > 0) & (ha > 0) & (oiii > 0) & (hb > 0)
    v &= np.isfinite(nii6584) & np.isfinite(ha) & np.isfinite(oiii) & np.isfinite(hb)
    x = np.full(len(ha), np.nan); y = np.full(len(ha), np.nan)
    x[v] = np.log10(nii6584[v] / ha[v]); y[v] = np.log10(oiii[v] / hb[v])
    ka = np.where(x < 0.05, 0.61 / (x - 0.05) + 1.30, -np.inf)
    ke = np.where(x < 0.47, 0.61 / (x - 0.47) + 1.19, -np.inf)
    sf = v & (y < ka); agn = v & ((y > ke) | (x >= 0.47)); comp = v & ~sf & ~agn
    lab = np.full(len(ha), -1, int); lab[sf] = 0; lab[comp] = 1; lab[agn] = 2
    return lab, x, y


def scat(r):
    r = r[np.isfinite(r)]
    p16, p84 = np.percentile(r, [16, 84]); return 0.5 * (p84 - p16)


def main():
    df = pd.read_csv(CSV)
    print(f"matched galaxies: {len(df)}  (median sep {np.median(df['sep_arcsec']):.2f}\")")

    lab_s, xs, ys = classify(df.sdss_NII6584, df.sdss_Halpha, df.sdss_OIII5007, df.sdss_Hbeta)
    lab_d, xd, yd = classify(df.desi_NII6584, df.desi_Halpha, df.desi_OIII5007, df.desi_Hbeta)

    # direct cross-survey [OIII]/Ha residual (DESI - SDSS) for the same galaxy
    oi_s = df.sdss_OIII5007.to_numpy(float); ha_s = df.sdss_Halpha.to_numpy(float)
    oi_d = df.desi_OIII5007.to_numpy(float); ha_d = df.desi_Halpha.to_numpy(float)
    vr = (oi_s > 0) & (ha_s > 0) & (oi_d > 0) & (ha_d > 0)
    delta = np.full(len(df), np.nan)
    delta[vr] = np.log10(oi_d[vr] / ha_d[vr]) - np.log10(oi_s[vr] / ha_s[vr])

    # ---- stats: per-class residual (SDSS classification) ----
    print("\ndelta = log([OIII]/Ha)_DESI - log([OIII]/Ha)_SDSS, split by SDSS BPT class:")
    for k, cl in enumerate(CLASSES):
        mk = (lab_s == k) & vr
        print(f"  {cl:9s} N={int(mk.sum()):6d}  bias(median)={np.median(delta[mk]):+.3f}  scatter={scat(delta[mk]):.3f}")

    # ---- AGN fraction: SDSS vs DESI on the common BPT-valid set ----
    both = (lab_s >= 0) & (lab_d >= 0)
    nb = int(both.sum())
    print(f"\nBPT-valid in BOTH surveys: N={nb}")
    for name, lab in [("SDSS", lab_s), ("DESI", lab_d)]:
        fr = [np.mean(lab[both] == k) for k in range(3)]
        print(f"  {name}:  SF={fr[0]:.1%}  composite={fr[1]:.1%}  AGN={fr[2]:.1%}")
    # migration matrix (rows SDSS class, cols DESI class), counts on 'both'
    Mig = np.zeros((3, 3), int)
    for i in range(3):
        for j in range(3):
            Mig[i, j] = int(np.sum((lab_s == i) & (lab_d == j) & both))
    print("\nSDSS -> DESI migration counts (rows SDSS, cols DESI) [SF, comp, AGN]:")
    for i in range(3):
        print(f"  {CLASSES[i]:9s} -> {Mig[i]}")
    agn_only_desi = int(np.sum((lab_s != 2) & (lab_d == 2) & both))
    agn_only_sdss = int(np.sum((lab_s == 2) & (lab_d != 2) & both))
    print(f"  AGN in DESI but NOT SDSS: {agn_only_desi}   |   AGN in SDSS but NOT DESI: {agn_only_sdss}")

    # ---------------- figure ----------------
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"axes.labelsize": 14, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 11, "axes.titlesize": 13})
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10.5), constrained_layout=True)

    # (0,0) residual histogram by SDSS class
    a = ax[0, 0]
    for k, cl in enumerate(CLASSES):
        mk = (lab_s == k) & vr
        a.hist(delta[mk], bins=np.linspace(-1.2, 1.2, 61), density=True, histtype="step", lw=2,
               color=CLASS_COL[cl], label=f"{cl} ({mk.mean():.0%}, $\\sigma$={scat(delta[mk]):.2f})")
    a.axvline(0, color="k", ls=":", lw=1)
    a.set_xlabel(r"$\Delta\log([{\rm O\,III}]/{\rm H}\alpha)$  (DESI $-$ SDSS) [dex]")
    a.set_ylabel("normalized density")
    a.set_title("Cross-survey [O III] residual by (SDSS) BPT class"); a.legend()

    # (0,1) migration matrix, row-normalized
    b = ax[0, 1]
    frac = Mig / Mig.sum(axis=1, keepdims=True).clip(min=1)
    im = b.imshow(frac, cmap="Purples", vmin=0, vmax=1)
    b.set_xticks(range(3)); b.set_xticklabels(CLASSES); b.set_yticks(range(3)); b.set_yticklabels(CLASSES)
    b.set_xlabel("DESI class"); b.set_ylabel("SDSS class")
    b.set_title("Class migration SDSS $\\rightarrow$ DESI (row-normalized)")
    for i in range(3):
        for j in range(3):
            b.text(j, i, f"{frac[i,j]:.0%}\n({Mig[i,j]})", ha="center", va="center",
                   fontsize=11, color="white" if frac[i, j] > 0.55 else "black")
    fr_s = [np.mean(lab_s[both] == k) for k in range(3)]; fr_d = [np.mean(lab_d[both] == k) for k in range(3)]
    b.text(1.0, -0.75, f"AGN fraction:  SDSS {fr_s[2]:.1%}   $\\rightarrow$   DESI {fr_d[2]:.1%}",
           ha="center", va="center", fontsize=12, fontweight="bold")

    # (1,0)/(1,1) BPT planes colored by median |delta|
    xn = np.linspace(-1.6, 0.3, 200); ka_c = 0.61 / (xn - 0.05) + 1.30; ke_c = 0.61 / (xn - 0.47) + 1.19
    for col, (xx, yy, ttl) in enumerate([(xs, ys, "SDSS BPT plane"), (xd, yd, "DESI BPT plane")]):
        c = ax[1, col]
        g = vr & np.isfinite(xx) & np.isfinite(yy)
        hb = c.hexbin(xx[g], yy[g], C=np.abs(delta[g]), gridsize=55, reduce_C_function=np.median,
                      cmap=CMAP, extent=(-1.6, 0.4, -1.2, 1.3), mincnt=8)
        m1 = xn < 0.05; m2 = xn < 0.47
        c.plot(xn[m1], ka_c[m1], "k--", lw=1.5, label="Kauffmann03")
        c.plot(xn[m2], ke_c[m2], "k-.", lw=1.5, label="Kewley01")
        c.set_xlabel(r"$\log$([N II]/H$\alpha$)"); c.set_ylabel(r"$\log$([O III]/H$\beta$)")
        c.set_xlim(-1.6, 0.4); c.set_ylim(-1.2, 1.3); c.set_title(ttl); c.legend(loc="lower left")
        cb = fig.colorbar(hb, ax=c, shrink=0.9); cb.set_label(r"median $|\Delta\log([{\rm O\,III}]/{\rm H}\alpha)|$ [dex]")

    fig.suptitle("Cross-matched SDSS$\\leftrightarrow$DESI: [O III] cross-survey residual is concentrated in AGN/composite excitation",
                 fontsize=14)
    for e in ("png", "pdf"):
        fig.savefig(REPO + f"figs/oiii_agn_scatter_crossmatch.{e}", dpi=180, bbox_inches="tight")
    print("\nWrote figs/oiii_agn_scatter_crossmatch.png")


if __name__ == "__main__":
    main()
