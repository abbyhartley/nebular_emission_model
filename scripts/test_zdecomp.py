"""
Review 2.4 / 2.1 -- redshift decomposition of the cross-survey shift.

At FIXED (M*, L_Ha) (a control box common to both surveys), does the OBSERVED
[NII]/Ha and [OIII]/Hb ratio evolve with redshift within each survey? And in the
z-overlap, do SDSS and DESI still differ at matched z + conditioning?

If ratios trend with z, then averaging over the two surveys' different z-distributions
produces part of the apparent "shift at fixed (M*, L_Ha)".

Outputs: docs/../zdecomp_ratio_vs_z.csv  and  scripts/plotting/zdecomp_ratio_vs_z.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model")
sys.path.insert(0, str(REPO / "scripts"))
from eval_nf_metrics_corrected import prep_eval_dataframe, FITS

Z_EDGES = np.array([0.05, 0.07, 0.09, 0.11, 0.13, 0.16, 0.20, 0.25])
MIN_BIN = 50


def ratios_frame(survey):
    df, res = prep_eval_dataframe(FITS[survey], survey=survey)
    ha = "LOG10_HA_FLUX"
    def col(sub):
        for c in res["lines"]:
            if sub in c:
                return f"LOG10_{c}"
        raise KeyError(sub)
    nii, oiii, hb = col("NII_6584"), col("OIII_5007"), col("BETA")
    return pd.DataFrame({
        "z": df[res["z"]].astype(float).to_numpy(),
        "logM": df["LOGM_COLOR"].astype(float).to_numpy(),
        "logLHa": df["LOG_LHA"].astype(float).to_numpy(),
        "r_NII_Ha": (df[nii] - df[ha]).to_numpy(),
        "r_OIII_Hb": (df[oiii] - df[hb]).to_numpy(),
    }).replace([np.inf, -np.inf], np.nan).dropna()


def main():
    s = ratios_frame("sdss")
    d = ratios_frame("desi")

    # common (M*, L_Ha) control box = intersection of 10-90 percentile ranges
    loM = max(np.percentile(s.logM, 10), np.percentile(d.logM, 10))
    hiM = min(np.percentile(s.logM, 90), np.percentile(d.logM, 90))
    loL = max(np.percentile(s.logLHa, 10), np.percentile(d.logLHa, 10))
    hiL = min(np.percentile(s.logLHa, 90), np.percentile(d.logLHa, 90))
    print(f"Control box: logM in [{loM:.2f},{hiM:.2f}], logLHa in [{loL:.2f},{hiL:.2f}]")

    def in_box(df):
        return df[(df.logM >= loM) & (df.logM <= hiM) & (df.logLHa >= loL) & (df.logLHa <= hiL)]
    sb, db = in_box(s), in_box(d)
    print(f"In box: SDSS {len(sb):,}  DESI {len(db):,}")

    rows = []
    for name, df in [("sdss", sb), ("desi", db)]:
        for i in range(len(Z_EDGES) - 1):
            lo, hi = Z_EDGES[i], Z_EDGES[i + 1]
            m = (df.z >= lo) & (df.z < hi)
            n = int(m.sum())
            if n < MIN_BIN:
                continue
            rows.append(dict(survey=name, z_lo=lo, z_hi=hi, z_mid=0.5 * (lo + hi), n=n,
                             med_NII_Ha=float(np.median(df.r_NII_Ha[m])),
                             med_OIII_Hb=float(np.median(df.r_OIII_Hb[m]))))
    binned = pd.DataFrame(rows)
    binned.to_csv(REPO / "docs" / "zdecomp_ratio_vs_z.csv", index=False)
    print("\n", binned.to_string(index=False))

    # z-trend slope within each survey (in box), and overlap vs full comparison
    print("\n=== ratio vs z linear slope (in box) ===")
    for name, df in [("sdss", sb), ("desi", db)]:
        for rr in ["r_NII_Ha", "r_OIII_Hb"]:
            sl = np.polyfit(df.z, df[rr], 1)[0]
            print(f"  {name} d({rr})/dz = {sl:+.3f} dex/unit-z")

    print("\n=== SDSS-DESI offset at matched conditioning ===")
    zov = (0.07, 0.10)
    for rr in ["r_NII_Ha", "r_OIII_Hb"]:
        full = np.median(sb[rr]) - np.median(db[rr])
        so = sb[(sb.z >= zov[0]) & (sb.z < zov[1])]
        do = db[(db.z >= zov[0]) & (db.z < zov[1])]
        ov = np.median(so[rr]) - np.median(do[rr])
        print(f"  {rr}: full-sample(box) offset = {full:+.3f} dex ; z-overlap[{zov[0]},{zov[1]}] offset = {ov:+.3f} dex "
              f"(SDSS n={len(so)}, DESI n={len(do)})")

    # plot
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, (rr, lab) in enumerate([("med_NII_Ha", r"median $\log$([NII]/H$\alpha$)"),
                                   ("med_OIII_Hb", r"median $\log$([OIII]/H$\beta$)")]):
        for name, color in [("sdss", "C0"), ("desi", "C1")]:
            sub = binned[binned.survey == name]
            ax[j].plot(sub.z_mid, sub[rr], "o-", color=color, label=name.upper())
        ax[j].set_xlabel("redshift"); ax[j].set_ylabel(lab)
        ax[j].axvspan(0.07, 0.10, color="gray", alpha=0.15)
    ax[0].legend(title="at fixed $(M_\\star, L_{H\\alpha})$")
    fig.tight_layout()
    outpng = REPO / "scripts" / "plotting" / "zdecomp_ratio_vs_z.png"
    fig.savefig(outpng, dpi=150)
    print("\nWrote:", outpng)
    print("Wrote:", REPO / "docs" / "zdecomp_ratio_vs_z.csv")


if __name__ == "__main__":
    main()
