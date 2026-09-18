"""
Data & sample slide figure: compare the SDSS vs DESI training-sample distributions
in the two conditioning variables (log M*, log L_Ha) and redshift. Normalized
histograms, house style (SciencePlots + Okabe-Ito, SDSS blue / DESI orange).
Motivates the cross-survey story: the two samples occupy different regions of
(M*, L_Ha, z) space.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = Path(BASE + "nebular_emission_model")
FLUX_SCALE = 1e-17
C = {"sdss": "#0072B2", "desi": "#E69F00"}


def load(p):
    t = Table.read(p, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()


def loglha(df, survey):
    if survey == "sdss":
        z = df["Z_1"].to_numpy(float); ha = df["H_ALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    else:
        z = df["Z"].to_numpy(float); ha = df["HALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    m = np.isfinite(z) & (z > 0) & np.isfinite(ha) & (ha > 0)
    out = np.full(len(df), np.nan)
    out[m] = np.log10(ha[m]) + np.log10(4 * np.pi) + 2 * np.log10(cosmo.luminosity_distance(z[m]).to("cm").value)
    return out, z


sdss = load(BASE + "SDSS_main_training_data.fits")
desi = load(BASE + "DESI_BGS_training_data.fits")
lha_s, z_s = loglha(sdss, "sdss"); lha_d, z_d = loglha(desi, "desi")
dat = {
    "sdss": {"logM": sdss["LOGM_COLOR"].to_numpy(float), "logLHa": lha_s, "z": z_s},
    "desi": {"logM": desi["LOGM_COLOR"].to_numpy(float), "logLHa": lha_d, "z": z_d},
}

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 13})
panels = [("logM", r"$\log_{10}(M_\star/M_\odot)$", np.linspace(8.5, 11.8, 60)),
          ("logLHa", r"$\log_{10}(L_{\mathrm{H}\alpha}\,/\,\mathrm{erg\,s^{-1}})$", np.linspace(39.5, 43.0, 60)),
          ("z", r"redshift", np.linspace(0.0, 0.30, 60))]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
for ax, (key, xlab, bins) in zip(axes, panels):
    for s in ("sdss", "desi"):
        v = dat[s][key]; v = v[np.isfinite(v)]
        ax.hist(v, bins=bins, density=True, histtype="step", lw=2.4, color=C[s],
                label=s.upper() if key == "logM" else None)
        ax.hist(v, bins=bins, density=True, histtype="stepfilled", alpha=0.18, color=C[s])
    ax.set_xlabel(xlab)
    ax.set_yticklabels([])
axes[0].set_ylabel("normalized density")
axes[0].legend(frameon=True, framealpha=0.9, loc="upper left")
for ext in ("png", "pdf"):
    (REPO / "figs").mkdir(exist_ok=True)
    fig.savefig(REPO / "figs" / f"data_distributions.{ext}", dpi=180, bbox_inches="tight")
    print("Wrote figs/data_distributions." + ext)
