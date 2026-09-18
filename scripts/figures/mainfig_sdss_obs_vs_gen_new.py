# make_main_figure_sdss_obs_vs_gen.py
from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import matplotlib.pyplot as plt

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
MINCNT = 5

# SDSS column names
Z_COL = "Z_1"
HA_COL = "H_ALPHA_FLUX"
LOGM_COL = "LOGM_COLOR"

# 8 target lines (SDSS naming)
LINE_SPECS = [
    ("H$\\beta$",    "H_BETA_FLUX"),
    ("H$\\gamma$",   "H_GAMMA_FLUX"),
    ("[NII]6584",    "NII_6584_FLUX"),
    ("[SII]6717",    "SII_6717_FLUX"),
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
    return np.log10(f) + np.log10(4 * np.pi) + 2 * np.log10(dl_cm)


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
    """Return df with LOG_LHA and LOG10 flux columns for Hα and the 8 lines, strict mask."""
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    t = t[names]
    df = t.to_pandas()

    z = df[Z_COL].astype(float).to_numpy()
    ha = df[HA_COL].astype(float).to_numpy()
    mstar = df[LOGM_COL].astype(float).to_numpy()

    mask = np.isfinite(z) & np.isfinite(ha) & (ha > 0) & np.isfinite(mstar)

    for _, col in LINE_SPECS:
        x = df[col].astype(float).to_numpy()
        mask &= np.isfinite(x) & (x > 0)

    df = df.loc[mask].copy().reset_index(drop=True)

    df["LOG10_HA_FLUX"] = np.log10(df[HA_COL].astype(float).to_numpy() * FLUX_SCALE)
    for _, col in LINE_SPECS:
        df[f"LOG10_{col}"] = np.log10(df[col].astype(float).to_numpy() * FLUX_SCALE)

    df["LOG_LHA"] = log10_lha_from_flux(df[Z_COL].to_numpy(float), df[HA_COL].to_numpy(float))
    return df


def true_logL_lines(df):
    """True log L_line = LOG_LHA + (logF_line - logF_Ha). Returns (N,8)."""
    logLHa = df["LOG_LHA"].to_numpy(float).reshape(-1, 1)
    logFHa = df["LOG10_HA_FLUX"].to_numpy(float).reshape(-1, 1)
    out = []
    for _, col in LINE_SPECS:
        logF = df[f"LOG10_{col}"].to_numpy(float).reshape(-1, 1)
        out.append(logLHa + (logF - logFHa))
    return np.hstack(out)


def predict_logL_lines(flow, meta, df, *, seed=0, n_mc=50, batch_size=200_000):
    """Predict logL via sampling ratios from flow and adding LOG_LHA. Returns (N,8)."""
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]

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
    """Return dict of rmse, scatter=0.5*(p84-p16), rho."""
    resid = y_pred - y_true
    rmse = float(np.sqrt(np.mean(resid**2)))
    p16, p84 = np.percentile(resid, [16, 84])
    scat = float(0.5 * (p84 - p16))
    rho, _ = spearmanr(y_true, y_pred)
    return dict(rmse=rmse, scat=scat, rho=float(rho))


# -------------------------
# Plotting
# -------------------------
def plot_obs_vs_gen_grid(y_true, y_pred, out_png="fig_sdss_obs_vs_gen.png"):
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 13.0), constrained_layout=True)
    axes = axes.ravel()
    hb_last = None

    for j, (label, _) in enumerate(LINE_SPECS):
        ax = axes[j]
        yt = y_true[:, j]
        yp = y_pred[:, j]

        # square limits (same x/y range)
        lo = np.percentile(np.concatenate([yt, yp]), 0.5)
        hi = np.percentile(np.concatenate([yt, yp]), 99.5)
        lo = max(lo, 35)
        hi = min(hi, 46)

        hb_last = ax.hexbin(
            yt, yp,
            gridsize=70,
            extent=(lo, hi, lo, hi),
            bins="log",
            mincnt=MINCNT,
            cmap="viridis",
        )

        ax.plot([lo, hi], [lo, hi], color="k", lw=1.2, alpha=0.8)

        st = per_line_stats(yt, yp)
        txt = (
            f"RMSE={st['rmse']:.3f} dex\n"
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
        ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(hb_last, ax=axes.tolist(), location="right", shrink=0.92, pad=0.01)
    cbar.set_label(f"log10(N per hex), mincnt={MINCNT}", fontsize=12)

    fig.suptitle("SDSS→SDSS: observed vs NF-generated line luminosities", fontsize=14)
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

    plot_obs_vs_gen_grid(y_true, y_pred, out_png="fig_sdss_obs_vs_gen.png")


if __name__ == "__main__":
    main()
