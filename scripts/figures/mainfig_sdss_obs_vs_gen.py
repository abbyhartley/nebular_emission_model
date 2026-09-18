# make_main_figure_sdss_obs_vs_gen.py
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow


# -------------------------
# Configuration
# -------------------------
FLOW_PATH = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main.eqx")
META_PATH = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main_meta.pkl")
SDSS_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")

FLUX_SCALE = 1e-17
N_MC = 50
SEED = 0

# SDSS column names
Z_COL = "Z_1"
HA_COL = "H_ALPHA_FLUX"
LOGM_COL = "LOGM_COLOR"

# Lines in the same order you trained/evaluated (8)
LINE_SPECS = [
    ("H$\\beta$",    "H_BETA_FLUX"),
    ("H$\\gamma$",   "H_GAMMA_FLUX"),
    ("[NII]6584",    "NII_6584_FLUX"),
    ("[SII]6717",    "SII_6717_FLUX"),   # SDSS has 6717
    ("[SII]6731",    "SII_6731_FLUX"),
    ("[OII]3726",    "OII_3726_FLUX"),
    ("[OII]3729",    "OII_3729_FLUX"),
    ("[OIII]5007",   "OIII_5007_FLUX"),
]


# -------------------------
# Helpers
# -------------------------
def log10_lha_from_flux(z, ha_flux_1e17):
    f = np.asarray(ha_flux_1e17, float) * FLUX_SCALE
    dl_cm = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(f) + np.log10(4*np.pi) + 2*np.log10(dl_cm)

def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    cond_dim = 2
    key = jr.key(int(meta.get("seed", 0)))
    template = block_neural_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=cond_dim,
    )
    return eqx.tree_deserialise_leaves(flow_path, template)

def prepare_sdss_df(fits_path):
    """Return df with LOG_LHA and LOG10 flux columns for Hα and the 8 lines, with strict mask."""
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    t = t[names]
    df = t.to_pandas()

    # base finite
    z = df[Z_COL].astype(float).to_numpy()
    ha = df[HA_COL].astype(float).to_numpy()
    mstar = df[LOGM_COL].astype(float).to_numpy()

    mask = np.isfinite(z) & np.isfinite(ha) & (ha > 0) & np.isfinite(mstar)

    # require all 8 lines > 0 for log10
    for _, col in LINE_SPECS:
        if col not in df.columns:
            raise KeyError(f"Missing {col} in SDSS file.")
        x = df[col].astype(float).to_numpy()
        mask &= np.isfinite(x) & (x > 0)

    df = df.loc[mask].copy().reset_index(drop=True)

    # logs
    df["LOG10_HA_FLUX"] = np.log10(df[HA_COL].astype(float).to_numpy() * FLUX_SCALE)
    for _, col in LINE_SPECS:
        df[f"LOG10_{col}"] = np.log10(df[col].astype(float).to_numpy() * FLUX_SCALE)

    # LOG_LHA conditioner
    df["LOG_LHA"] = log10_lha_from_flux(df[Z_COL].to_numpy(float), df[HA_COL].to_numpy(float))
    return df

def true_logL_lines(df):
    """True log L_line = LOG_LHA + (logF_line - logF_Ha). Returns (N,8)."""
    logLHa = df["LOG_LHA"].to_numpy(float).reshape(-1, 1)
    logFHa = df["LOG10_HA_FLUX"].to_numpy(float).reshape(-1, 1)
    cols = [col for _, col in LINE_SPECS]
    out = []
    for col in cols:
        logF = df[f"LOG10_{col}"].to_numpy(float).reshape(-1, 1)
        out.append(logLHa + (logF - logFHa))
    return np.hstack(out)

def predict_logL_lines(flow, meta, df, *, seed=0, n_mc=50, batch_size=200_000):
    """Predict logL via sampling ratios from flow and adding LOG_LHA. Returns (N,8)."""
    logm_col = meta["resolved"]["logmstar_col"]   # should be LOGM_COLOR
    loglha_col = meta["resolved"]["loglha_col"]   # should be LOG_LHA

    U_all = df[[logm_col, loglha_col]].to_numpy(dtype=np.float32)
    logLHa_all = df[loglha_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    X_mean, X_std = meta["X_mean"], meta["X_std"]
    U_mean, U_std = meta["U_mean"], meta["U_std"]

    n = len(df)
    out = np.empty((n, len(meta["resolved"]["out_cols"])), dtype=np.float32)

    def sample_one(k, u):
        return flow.sample(k, sample_shape=(), condition=u)

    key = jr.key(seed + 999)

    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        U = U_all[lo:hi]
        logLHa = logLHa_all[lo:hi]
        Un = (U - U_mean) / U_std
        Un_jax = jnp.asarray(Un)
        nb = hi - lo

        ratio_samples = []
        for _ in range(int(n_mc)):
            key, subkey = jr.split(key)
            keys_i = jr.split(subkey, nb)
            Xn = jax.vmap(sample_one, in_axes=(0, 0))(keys_i, Un_jax)
            Xn = np.array(Xn)
            ratios = Xn * X_std + X_mean
            ratio_samples.append(ratios)

        ratios_mean = np.mean(np.stack(ratio_samples, axis=0), axis=0)
        out[lo:hi] = ratios_mean + logLHa

    return out

def per_line_stats(y_true, y_pred):
    """Return dict of rmse, bias (median), scatter=0.5*(p84-p16), rho."""
    resid = y_pred - y_true
    rmse = float(np.sqrt(np.mean(resid**2)))
    bias = float(np.median(resid))
    p16, p84 = np.percentile(resid, [16, 84])
    scat = float(0.5 * (p84 - p16))
    rho, _ = spearmanr(y_true, y_pred)
    return dict(rmse=rmse, bias=bias, scat=scat, rho=float(rho))


# -------------------------
# Plotting
# -------------------------
def plot_obs_vs_gen_grid(y_true, y_pred, out_png="fig1_sdss_obs_vs_gen.png"):
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 13.0), sharex=False, sharey=False, constrained_layout=True)
    axes = axes.ravel()

    # Use a robust range for each line to avoid a few extremes dominating
    for j, (label, col) in enumerate(LINE_SPECS):
        ax = axes[j]
        yt = y_true[:, j]
        yp = y_pred[:, j]

        lo = np.percentile(np.concatenate([yt, yp]), 0.5)
        hi = np.percentile(np.concatenate([yt, yp]), 99.5)
        lo = max(lo, 35)  # safety floor
        hi = min(hi, 46)  # safety cap

        hb = ax.hexbin(
            yt, yp,
            gridsize=70,
            extent=(lo, hi, lo, hi),
            bins="log",
            mincnt=1,
            cmap="viridis"
        )

        # 1:1 line
        ax.plot([lo, hi], [lo, hi], color="k", lw=1.2, alpha=0.8)

        st = per_line_stats(yt, yp)
        txt = (
            f"RMSE={st['rmse']:.3f} dex\n"
            f"bias={st['bias']:+.3f}\n"
            f"scat={st['scat']:.3f}\n"
            f"$\\rho$={st['rho']:.3f}"
        )
        ax.text(
            0.03, 0.97, txt,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round,pad=0.25")
        )

        ax.set_title(label, fontsize=12)
        ax.set_xlabel(r"$\log L_{\rm obs}\;[\mathrm{erg\,s^{-1}}]$")
        ax.set_ylabel(r"$\log L_{\rm gen}\;[\mathrm{erg\,s^{-1}}]$")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    # shared colorbar
    cbar = fig.colorbar(hb, ax=axes.tolist(), location="right", shrink=0.92, pad=0.01)
    cbar.set_label("log10(N per hex)", fontsize=12)

    fig.suptitle("SDSS→SDSS: observed vs NF-generated line luminosities", fontsize=14)
    fig.savefig(out_png, dpi=250)
    print("Saved:", out_png)
    plt.show()


def plot_ratio_plane(df, y_pred, out_png="fig1b_ratio_plane.png"):
    """
    Companion plot: BPT-like ratio plane comparing observed vs generated.
    Here we compare:
      x = log([NII]/Ha)
      y = log([OIII]/Hb)
    Using the same df (observed) and y_pred (generated logL).
    """
    # observed ratios from df logs (flux ratios = luminosity ratios)
    logNII_Ha_obs = df["LOG10_NII_6584_FLUX"].to_numpy() - df["LOG10_HA_FLUX"].to_numpy()
    logOIII_Hb_obs = df["LOG10_OIII_5007_FLUX"].to_numpy() - df["LOG10_H_BETA_FLUX"].to_numpy()

    # generated ratios from generated logL (difference cancels LOG_LHA)
    # y_pred columns are in LINE_SPECS order; find indices:
    idx_hb = [c for _, c in LINE_SPECS].index("H_BETA_FLUX")
    idx_nii = [c for _, c in LINE_SPECS].index("NII_6584_FLUX")
    idx_oiii = [c for _, c in LINE_SPECS].index("OIII_5007_FLUX")

    logNII_Ha_gen = (y_pred[:, idx_nii] - df["LOG_LHA"].to_numpy())  # = predicted ratio to Ha
    logOIII_Hb_gen = (y_pred[:, idx_oiii] - y_pred[:, idx_hb])       # ratio (OIII/Hb)

    # Make side-by-side density
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True, constrained_layout=True)

    xr = (-2.0, 1.0)
    yr = (-1.5, 1.5)

    h0 = axes[0].hexbin(logNII_Ha_obs, logOIII_Hb_obs, gridsize=90, extent=(*xr, *yr),
                        bins="log", mincnt=1, cmap="magma")
    h1 = axes[1].hexbin(logNII_Ha_gen, logOIII_Hb_gen, gridsize=90, extent=(*xr, *yr),
                        bins="log", mincnt=1, cmap="magma")

    axes[0].set_title("Observed (SDSS)")
    axes[1].set_title("Generated (NF samples)")

    for ax in axes:
        ax.set_xlabel(r"$\log([\mathrm{NII}]/\mathrm{H}\alpha)$")
        ax.set_xlim(*xr); ax.set_ylim(*yr)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(r"$\log([\mathrm{OIII}]/\mathrm{H}\beta)$")

    cbar = fig.colorbar(h1, ax=axes.tolist(), location="right", shrink=0.95, pad=0.01)
    cbar.set_label("log10(N per hex)")

    fig.savefig(out_png, dpi=250)
    print("Saved:", out_png)
    plt.show()


def main():
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    flow = load_flow(FLOW_PATH, meta)

    df = prepare_sdss_df(SDSS_FITS)
    print("N used (after masks):", len(df))

    y_true = true_logL_lines(df)
    y_pred = predict_logL_lines(flow, meta, df, seed=SEED, n_mc=N_MC)

    plot_obs_vs_gen_grid(y_true, y_pred, out_png="fig1_sdss_obs_vs_gen.png")
    # optional companion ratio plot:
    plot_ratio_plane(df, y_pred, out_png="fig1b_sdss_ratio_plane.png")


if __name__ == "__main__":
    main()
