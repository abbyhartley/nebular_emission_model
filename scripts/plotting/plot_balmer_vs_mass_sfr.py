"""
Balmer decrement (R = Ha/Hb) as a function of stellar mass and SFR, for the
SDSS and DESI training samples. 2D map: median R in (log M*, log SFR) bins.
Stellar mass = LOGM_COLOR (the NF conditioning mass). SFR: SDSS uses MPA-JHU
SFR_TOT_P50 (already log10); DESI uses FastSpecFit SFR (linear -> log10).
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
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()


def prep(df, survey):
    logm = df["LOGM_COLOR"].to_numpy(float)
    if survey == "sdss":
        logsfr = df["SFR_TOT_P50"].to_numpy(float)            # already log10
        ha, hb = df["H_ALPHA_FLUX"].to_numpy(float), df["H_BETA_FLUX"].to_numpy(float)
        sfr_ok = np.isfinite(logsfr) & (logsfr > -90)         # drop -9999 sentinels
    else:
        sfr = df["SFR"].to_numpy(float)                        # linear Msun/yr
        sfr_ok = np.isfinite(sfr) & (sfr > 0)
        logsfr = np.full(len(df), np.nan)
        logsfr[sfr_ok] = np.log10(sfr[sfr_ok])
        ha, hb = df["HALPHA_FLUX"].to_numpy(float), df["HBETA_FLUX"].to_numpy(float)
    R = np.full(len(df), np.nan)
    good_hb = np.isfinite(ha) & np.isfinite(hb) & (ha > 0) & (hb > 0)
    R[good_hb] = ha[good_hb] / hb[good_hb]
    m = np.isfinite(logm) & sfr_ok & good_hb & np.isfinite(R) & (R > 0) & (R < 15)
    return logm[m], logsfr[m], R[m]


def med_map(logm, logsfr, R):
    stat, _, _, _ = binned_statistic_2d(logm, logsfr, R, statistic="median", bins=[MBINS, SBINS])
    cnt, _, _, _ = binned_statistic_2d(logm, logsfr, R, statistic="count", bins=[MBINS, SBINS])
    stat[cnt < MINCT] = np.nan
    return np.ma.masked_invalid(stat.T)            # transpose: rows=SFR, cols=M*


sdss = prep(load(BASE + "SDSS_main_training_data.fits"), "sdss")
desi = prep(load(BASE + "DESI_BGS_training_data.fits"), "desi")
maps = {"SDSS": med_map(*sdss), "DESI": med_map(*desi)}

vmin, vmax = 2.86, float(np.nanpercentile(np.concatenate(
    [m.compressed() for m in maps.values()]), 98))

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 16, "axes.titlesize": 15,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True, sharey=True)
ext = [MBINS[0], MBINS[-1], SBINS[0], SBINS[-1]]
for ax, (lbl, M) in zip(axes, maps.items()):
    im = ax.imshow(M, origin="lower", extent=ext, aspect="auto", cmap=cmr.bubblegum,
                   vmin=vmin, vmax=vmax)
    ax.set_title(lbl)
    ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    ax.set_box_aspect(1)
axes[0].set_ylabel(r"$\log_{10}(\mathrm{SFR}\,/\,M_\odot\,\mathrm{yr}^{-1})$")
cbar = fig.colorbar(im, ax=axes, shrink=0.82, location="right", pad=0.02)
cbar.set_label(r"median Balmer decrement  $R = F_{\mathrm{H}\alpha}/F_{\mathrm{H}\beta}$", fontsize=14)
for e in ("png", "pdf"):
    (REPO / "figs").mkdir(exist_ok=True)
    fig.savefig(REPO / "figs" / f"balmer_vs_mass_sfr.{e}", dpi=180, bbox_inches="tight")
    print("Wrote figs/balmer_vs_mass_sfr." + e)
print("vmax(98pct) =", round(vmax, 2))
