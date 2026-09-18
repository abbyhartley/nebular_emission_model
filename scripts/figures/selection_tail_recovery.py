# selection_tail_recovery.py — compare mass & redshift coverage of the three DESI
# selections (strict baseline, ALT-B per-line>3, ALT-A detect-only) using LOGM_COLOR & Z
# from the training FITS (identical mass definition). Shows how much of the high-mass/
# high-z tail each recovers, plus tail fractions. Also prints the full POOL sizes.
from pathlib import Path
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 15, "axes.titlesize": 15,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})

GFC = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs"
SAMPLES = [
    ("strict (per-line>5, cont>5)", f"{GFC}/DESI_BGS_training_data.fits", "#D55E00", 130382),
    ("ALT-B (per-line>3, cont>3)",  f"{GFC}/DESI_ALT_B.fits",             "#0072B2", 499944),
    ("ALT-A (detect-only, cont>3)", f"{GFC}/DESI_ALT_A.fits",             "#009E73", 1528556),
]


def load(path):
    t = Table.read(path, hdu=1)
    return np.asarray(t["LOGM_COLOR"], float), np.asarray(t["Z"], float)


def main():
    data = {}
    print(f"{'selection':32s} {'N(sub)':>8s} {'pool':>10s}  frac z>0.3  frac logM>10.5  frac logM>11", flush=True)
    for lab, path, col, pool in SAMPLES:
        lm, z = load(path)
        g = np.isfinite(lm) & np.isfinite(z)
        lm, z = lm[g], z[g]
        data[lab] = (lm, z, col, pool)
        print(f"{lab:32s} {len(z):>8,} {pool:>10,}  {np.mean(z>0.3):>8.1%}  {np.mean(lm>10.5):>12.1%}  {np.mean(lm>11):>10.1%}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    mb = np.linspace(8, 12, 60)
    zb = np.linspace(0, 0.5, 60)
    for lab, (lm, z, col, pool) in data.items():
        ax[0].hist(lm, bins=mb, density=True, histtype="step", lw=2.2, color=col, label=lab)
        ax[1].hist(z, bins=zb, density=True, histtype="step", lw=2.2, color=col, label=lab)
    ax[0].set_xlabel(r"$\log(M_\star/M_\odot)$ [LOGM_COLOR]"); ax[0].set_ylabel("normalized density")
    ax[0].axvline(10.5, color="0.6", ls=":", lw=1.3); ax[0].legend()
    ax[1].set_xlabel("redshift"); ax[1].axvline(0.3, color="0.6", ls=":", lw=1.3); ax[1].legend()
    fig.suptitle("DESI selection: mass & redshift coverage (normalized shapes; pools 130k / 500k / 1.53M)",
                 fontsize=14)
    fig.savefig("selection_tail_recovery.png", dpi=210, bbox_inches="tight")
    print("Saved: selection_tail_recovery.png", flush=True)


if __name__ == "__main__":
    main()
