# plot_um_vs_desi_condition_hists.py
#
# Overlays histograms of conditioning variables for UM vs DESI:
#   (1) LOGM_COLOR  (log10 M*)
#   (2) LOG_LHA     (log10 L_Ha [erg/s])
#
# Uses SciencePlots styling.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 250,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ---- inputs ----
UM_PARQUET = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/UM_comp/um_a0.911185_z0p1_conditions_logMge8.3.parquet")
DESI_FITS  = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

# colors (colorblind-friendly)
C_UM = "#0072B2"    # blue
C_DESI = "#E69F00"  # orange


def desi_loglha_from_table(t: Table):
    """Compute LOG_LHA if not present; assumes HALPHA_FLUX in 1e-17 cgs and Z present."""
    if "LOG_LHA" in t.colnames:
        return np.asarray(t["LOG_LHA"], float)

    z = np.asarray(t["Z"], float)
    ha = np.asarray(t["HALPHA_FLUX"], float)
    m = np.isfinite(z) & (z > 0) & np.isfinite(ha) & (ha > 0)
    out = np.full(len(t), np.nan, dtype=float)
    dl_cm = cosmo.luminosity_distance(z[m]).to("cm").value
    out[m] = np.log10(ha[m] * 1e-17) + np.log10(4*np.pi) + 2*np.log10(dl_cm)
    return out


def main():
    # UM
    um = pd.read_parquet(UM_PARQUET)
    um_m = um["LOGM_COLOR"].to_numpy(float)
    um_l = um["LOG_LHA"].to_numpy(float)

    um_m = um_m[np.isfinite(um_m)]
    um_l = um_l[np.isfinite(um_l)]

    # DESI
    t = Table.read(DESI_FITS, hdu=1)
    desi_m = np.asarray(t["LOGM_COLOR"], float)
    desi_l = desi_loglha_from_table(t)

    desi_m = desi_m[np.isfinite(desi_m)]
    desi_l = desi_l[np.isfinite(desi_l)]

    # ---- (1) Mass histogram ----
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)

    lo = min(np.percentile(um_m, 0.5), np.percentile(desi_m, 0.5))
    hi = max(np.percentile(um_m, 99.5), np.percentile(desi_m, 99.5))

    bins = np.linspace(lo, hi, 55)
    ax.hist(um_m, bins=bins, histtype="step", lw=2.2, color=C_UM, density=True, label="UM (filtered)")
    ax.hist(desi_m, bins=bins, histtype="step", lw=2.2, color=C_DESI, density=True, label="DESI BGS train")

    ax.set_xlabel(r"$\log_{10} M_\star$")
    ax.set_ylabel("Normalized density")
    ax.legend(frameon=True)

    out1 = "hist_logM_um_vs_desi.png"
    fig.savefig(out1)
    print("Saved:", out1)

    # ---- (2) LHa histogram ----
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)

    lo = min(np.percentile(um_l, 0.5), np.percentile(desi_l, 0.5))
    hi = max(np.percentile(um_l, 99.5), np.percentile(desi_l, 99.5))

    bins = np.linspace(lo, hi, 55)
    ax.hist(um_l, bins=bins, histtype="step", lw=2.2, color=C_UM, density=True, label="UM (filtered)")
    ax.hist(desi_l, bins=bins, histtype="step", lw=2.2, color=C_DESI, density=True, label="DESI BGS train")

    ax.set_xlabel(r"$\log_{10} L_{H\alpha}\;[\mathrm{erg\,s^{-1}}]$")
    ax.set_ylabel("Normalized density")
    ax.legend(frameon=True)

    out2 = "hist_logLHa_um_vs_desi.png"
    fig.savefig(out2)
    print("Saved:", out2)


if __name__ == "__main__":
    main()
