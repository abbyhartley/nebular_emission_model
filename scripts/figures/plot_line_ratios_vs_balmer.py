"""
The 8 target line ratios log10(L_line/L_Ha) as a function of Balmer decrement
R=Ha/Hb, for SDSS and DESI. Two figures:
  (B) full sample
  (C) within a fixed (M*, L_Ha) control box -> does dust carry info beyond the
      conditioning variables? (key test for conditioning the flow on R)
Lines colored by rest wavelength: dust attenuates bluer lines more, so relative
to Ha, bluer-line ratios should decline with R while red lines (~Ha) stay flat.
"""
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import binned_statistic

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots  # noqa: F401
import cmasher as cmr

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = Path(BASE + "nebular_emission_model")
FLUX_SCALE = 1e-17
META = {"sdss": REPO / "nf_sdss_main_meta.pkl", "desi": REPO / "nf_desi_bgs_meta.pkl"}
FITS = {"sdss": BASE + "SDSS_main_training_data.fits", "desi": BASE + "DESI_BGS_training_data.fits"}
# target order = out_cols order
LABELS = [r"H$\beta$", r"H$\gamma$", r"[N II]6584", r"[S II]6716", r"[S II]6731",
          r"[O II]3726", r"[O II]3729", r"[O III]5007"]
WL = np.array([4861.3, 4340.5, 6583.5, 6716.4, 6730.8, 3726.0, 3728.8, 5006.8])
RBINS = np.linspace(2.86, 7.0, 16)
BOX = dict(logm=(9.6, 10.0), loglha=(40.56, 41.06))


def load(p):
    t = Table.read(p, hdu=1)
    return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()


def prep(survey):
    meta = pickle.load(open(META[survey], "rb"))
    df = load(FITS[survey])
    tcols = [c[6:] if c.startswith("LOG10_") else c for c in meta["resolved"]["target_cols"]]
    hacol = next(c for c in ["H_ALPHA_FLUX", "HALPHA_FLUX"] if c in df.columns)
    F = np.column_stack([df[c].to_numpy(float) for c in tcols])      # 8 line fluxes
    fha = df[hacol].to_numpy(float)
    logm = df["LOGM_COLOR"].to_numpy(float)
    z = df["Z_1" if survey == "sdss" else "Z"].to_numpy(float)
    okz = np.isfinite(z) & (z > 0) & np.isfinite(fha) & (fha > 0)
    loglha = np.full(len(df), np.nan)
    loglha[okz] = np.log10(fha[okz] * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(
        cosmo.luminosity_distance(z[okz]).to("cm").value)
    good = np.all(F > 0, axis=1) & (fha > 0) & np.isfinite(fha) & np.all(np.isfinite(F), axis=1) & np.isfinite(logm)
    ratios = np.log10(F) - np.log10(fha)[:, None]                    # (N,8) = log10(line/Ha)
    R = fha / F[:, 0]                                                # Ha / Hbeta  (Hbeta = col 0)
    return dict(ratios=ratios, R=R, logm=logm, loglha=loglha, good=good & np.isfinite(R) & (R > 0))


data = {s: prep(s) for s in ("sdss", "desi")}
norm = mpl.colors.Normalize(vmin=WL.min(), vmax=WL.max())
cmap = cmr.bubblegum


def panel(ax, d, mask, title):
    Rv = d["R"][mask]
    for i in range(8):
        y = d["ratios"][mask, i]
        med, edges, _ = binned_statistic(Rv, y, statistic="median", bins=RBINS)
        cnt, _, _ = binned_statistic(Rv, y, statistic="count", bins=RBINS)
        med[cnt < 30] = np.nan
        xc = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(xc, med, "-o", ms=3.5, lw=2.0, color=cmap(norm(WL[i])), label=LABELS[i])
    ax.axvline(2.86, color="0.5", ls=":", lw=1.1)
    ax.set_title(title); ax.set_xlabel(r"Balmer decrement $R=F_{\mathrm{H}\alpha}/F_{\mathrm{H}\beta}$")


def make(figkey, boxcut, suptitle, fname):
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"axes.labelsize": 15, "axes.titlesize": 14, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 9.5})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True, sharey=True)
    for ax, s in zip(axes, ("sdss", "desi")):
        d = data[s]; m = d["good"].copy()
        if boxcut:
            m &= (d["logm"] >= BOX["logm"][0]) & (d["logm"] < BOX["logm"][1])
            m &= (d["loglha"] >= BOX["loglha"][0]) & (d["loglha"] < BOX["loglha"][1])
        panel(ax, d, m, f"{s.upper()}  (N={int(m.sum()):,})")
    axes[0].set_ylabel(r"median $\log_{10}(L_{\rm line}/L_{\mathrm{H}\alpha})$")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.85, location="right", pad=0.02)
    cbar.set_label(r"rest wavelength (\AA)", fontsize=13)
    axes[0].legend(frameon=True, framealpha=0.9, loc="lower left", ncol=2)
    fig.suptitle(suptitle, fontsize=14)
    for e in ("png", "pdf"):
        (REPO / "figs").mkdir(exist_ok=True)
        fig.savefig(REPO / "figs" / f"{fname}.{e}", dpi=170, bbox_inches="tight")
    print("Wrote figs/" + fname + ".png")


make("full", False, r"Line ratios vs Balmer decrement (full sample)", "line_ratios_vs_balmer_full")
make("box", True, r"Line ratios vs Balmer decrement at fixed $(M_\star, L_{\mathrm{H}\alpha})$",
     "line_ratios_vs_balmer_box")
