"""
Scatter (NMAD) of the Balmer decrement R=Ha/Hb in the (log M*, log SFR) plane,
companion to the median map. Shows where the dust distribution is tight vs broad.
"""
from pathlib import Path
import numpy as np
from astropy.table import Table
from scipy.stats import binned_statistic_2d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import cmasher as cmr

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = Path(BASE + "nebular_emission_model")
MBINS = np.linspace(8.5, 11.5, 26)
SBINS = np.linspace(-1.5, 1.8, 26)
MINCT = 25


def load(p):
    t = Table.read(p, hdu=1)
    return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()


def prep(df, survey):
    logm = df["LOGM_COLOR"].to_numpy(float)
    if survey == "sdss":
        logsfr = df["SFR_TOT_P50"].to_numpy(float); ok = np.isfinite(logsfr) & (logsfr > -90)
        ha, hb = df["H_ALPHA_FLUX"].to_numpy(float), df["H_BETA_FLUX"].to_numpy(float)
    else:
        sfr = df["SFR"].to_numpy(float); ok = np.isfinite(sfr) & (sfr > 0)
        logsfr = np.where(ok, np.log10(np.clip(sfr, 1e-9, None)), np.nan)
        ha, hb = df["HALPHA_FLUX"].to_numpy(float), df["HBETA_FLUX"].to_numpy(float)
    g = np.isfinite(logm) & ok & (ha > 0) & (hb > 0)
    R = np.where(g, ha / hb, np.nan)
    m = g & np.isfinite(R) & (R > 0) & (R < 15)
    return logm[m], logsfr[m], R[m]


def nmad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) else np.nan


def scatter_map(logm, logsfr, R):
    stat, _, _, _ = binned_statistic_2d(logm, logsfr, R, statistic=nmad, bins=[MBINS, SBINS])
    cnt, _, _, _ = binned_statistic_2d(logm, logsfr, R, statistic="count", bins=[MBINS, SBINS])
    stat[cnt < MINCT] = np.nan
    return np.ma.masked_invalid(stat.T)


maps = {"SDSS": scatter_map(*prep(load(BASE + "SDSS_main_training_data.fits"), "sdss")),
        "DESI": scatter_map(*prep(load(BASE + "DESI_BGS_training_data.fits"), "desi"))}
vmax = float(np.nanpercentile(np.concatenate([m.compressed() for m in maps.values()]), 98))

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 16, "axes.titlesize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True, sharey=True)
ext = [MBINS[0], MBINS[-1], SBINS[0], SBINS[-1]]
for ax, (lbl, M) in zip(axes, maps.items()):
    im = ax.imshow(M, origin="lower", extent=ext, aspect="auto", cmap=cmr.bubblegum, vmin=0, vmax=vmax)
    ax.set_title(lbl); ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$"); ax.set_box_aspect(1)
axes[0].set_ylabel(r"$\log_{10}(\mathrm{SFR}\,/\,M_\odot\,\mathrm{yr}^{-1})$")
cbar = fig.colorbar(im, ax=axes, shrink=0.82, location="right", pad=0.02)
cbar.set_label(r"Balmer-decrement scatter  (NMAD of $R$)", fontsize=14)
for e in ("png", "pdf"):
    (REPO / "figs").mkdir(exist_ok=True)
    fig.savefig(REPO / "figs" / f"balmer_scatter_mass_sfr.{e}", dpi=180, bbox_inches="tight")
    print("Wrote figs/balmer_scatter_mass_sfr." + e)
print("vmax98 =", round(vmax, 3))
