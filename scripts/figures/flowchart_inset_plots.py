# make_flowchart_inset_plots.py
#
# Produces 4 small, square, simple inset plots (no titles) for a schematic flow chart:
#   (1) logM* vs logL(Hα)  (DESI training sample)
#   (2) DESI in-survey:  logL_pred(Hβ)  vs  logL_true(Hβ)
#   (3) DESI in-survey:  logL_pred([OIII]5007) vs logL_true([OIII]5007)
#   (4) Cross-survey:    logL_pred([OIII]5007) vs logL_true([OIII]5007) on SDSS
#
# Uses: bubblegum colormap, red dotted 1:1 line, big axis labels (fontsize=16).
# Saves each panel as its own PNG for easy placement in a flow chart.

from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import cmasher as cmr

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow


# -----------------------
# Style
# -----------------------
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

CMAP = cmr.bubblegum
MINCNT = 5
LINE_11_KW = dict(color="black", lw=2.0, ls=":", alpha=0.95)

# -----------------------
# Inputs
# -----------------------
DESI_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")
SDSS_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")

FLOW_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs.eqx")
META_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs_meta.pkl")

FLUX_SCALE = 1e-17
SEED = 0
N_MC = 50          # MC mean prediction
N_PLOT = 200_000   # downsample for faster plotting (hexbin is fine with this)


# -----------------------
# Helpers
# -----------------------
def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    key = jr.key(int(meta.get("seed", 0)))
    template = block_neural_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=2,
    )
    return eqx.tree_deserialise_leaves(flow_path, template)

def load_df(fits_path):
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()

def log10_lum_from_flux(z, flux_1e17):
    z = np.asarray(z, float)
    f = np.asarray(flux_1e17, float) * FLUX_SCALE
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    return np.log10(f) + np.log10(4*np.pi) + 2*np.log10(dl_cm)

def ensure_loglha(df, *, z_col, ha_col):
    df = df.copy()
    if "LOG_LHA" not in df.columns:
        df["LOG_LHA"] = log10_lum_from_flux(df[z_col].to_numpy(float), df[ha_col].to_numpy(float))
    return df

def outcol_index(meta, substrs):
    out_cols = meta["resolved"]["out_cols"]
    for i, name in enumerate(out_cols):
        if any(s in name for s in substrs):
            return i
    raise KeyError(f"Could not find any of {substrs} in out_cols.")

def sample_ratios_mcmean(flow, meta, df, *, seed=0, n_mc=50, batch_size=200_000):
    """Return MC-mean log ratios (log F_line/F_Ha) for all out dims, shape (N,8)."""
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]

    U_all = df[[logm_col, loglha_col]].to_numpy(np.float32)
    Un_all = (U_all - meta["U_mean"]) / meta["U_std"]
    Un_all = jnp.asarray(Un_all)

    X_mean, X_std = meta["X_mean"], meta["X_std"]

    n = len(df)
    out = np.empty((n, len(meta["resolved"]["out_cols"])), dtype=np.float32)

    def sample_one(k, u):
        return flow.sample(k, sample_shape=(), condition=u)

    key = jr.key(seed + 999)

    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        Un = Un_all[lo:hi]
        nb = hi - lo

        rs = []
        for _ in range(int(n_mc)):
            key, subkey = jr.split(key)
            keys = jr.split(subkey, nb)
            Xn = jax.vmap(sample_one, in_axes=(0, 0))(keys, Un)
            Xn = np.array(Xn)
            rs.append(Xn * X_std + X_mean)

        out[lo:hi] = np.mean(np.stack(rs, axis=0), axis=0)

    return out

def thin_df(df, n, seed=0):
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def square_hexbin(ax, x, y, *, cmap=CMAP, mincnt=MINCNT, gridsize=70):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    lo = np.min(np.concatenate([x, y]))
    hi = np.max(np.concatenate([x, y]))
    pad = 0.02 * (hi - lo)
    lo -= pad; hi += pad

    hb = ax.hexbin(x, y, gridsize=gridsize, extent=(lo, hi, lo, hi),
                   bins="log", mincnt=mincnt, cmap=cmap)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    return hb, lo, hi

def save_square(fig, outpng):
    fig.savefig(outpng, dpi=250, bbox_inches="tight", pad_inches=0.02)
    print("Saved:", outpng)
    plt.close(fig)


# -----------------------
# Main
# -----------------------
def main():
    # Load DESI + SDSS
    df_desi = load_df(DESI_FITS)
    df_sdss = load_df(SDSS_FITS)

    # Ensure LOG_LHA exists (needed for conditioning / plotting)
    df_desi = ensure_loglha(df_desi, z_col="Z", ha_col="HALPHA_FLUX")
    df_sdss = ensure_loglha(df_sdss, z_col="Z_1", ha_col="H_ALPHA_FLUX")

    # Downsample for plotting
    df_desi_p = thin_df(df_desi, N_PLOT, seed=SEED)
    df_sdss_p = thin_df(df_sdss, N_PLOT, seed=SEED + 1)

    # Load DESI flow
    with open(META_DESI, "rb") as f:
        meta_desi = pickle.load(f)
    flow_desi = load_flow(FLOW_DESI, meta_desi)

    # --- (1) logM* vs logL(Ha) (DESI) ---
    fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
    x = df_desi_p["LOGM_COLOR"].to_numpy(float)
    y = df_desi_p["LOG_LHA"].to_numpy(float)
    ax.hexbin(x, y, gridsize=65, bins="log", mincnt=MINCNT, cmap=CMAP)
    ax.set_xlabel(r"$\log M_\star$")
    ax.set_ylabel(r"$\log L_{H\alpha}$")
    save_square(fig, "inset_01_logM_vs_logLHa_desi.png")

    # Compute MC-mean ratios for DESI on DESI (for Hb and OIII)
    ratios_desi_on_desi = sample_ratios_mcmean(flow_desi, meta_desi, df_desi_p, seed=SEED+10, n_mc=N_MC)

    # Indices in flow output
    i_hb = outcol_index(meta_desi, ["H_BETA", "HBETA"])
    i_oiii = outcol_index(meta_desi, ["OIII_5007"])

    # True logL for DESI
    logLHa_desi = df_desi_p["LOG_LHA"].to_numpy(float)
    logLhb_true_desi = log10_lum_from_flux(df_desi_p["Z"].to_numpy(float), df_desi_p["HBETA_FLUX"].to_numpy(float))
    logLoiii_true_desi = log10_lum_from_flux(df_desi_p["Z"].to_numpy(float), df_desi_p["OIII_5007_FLUX"].to_numpy(float))

    # Pred logL from ratios: logL(line) = logLHa + log(line/Ha)
    logLhb_pred_desi = logLHa_desi + ratios_desi_on_desi[:, i_hb].astype(float)
    logLoiii_pred_desi = logLHa_desi + ratios_desi_on_desi[:, i_oiii].astype(float)

    # --- (2) DESI in-survey: Hb pred vs true ---
    fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
    hb, lo, hi = square_hexbin(ax, logLhb_true_desi, logLhb_pred_desi)
    ax.plot([lo, hi], [lo, hi], **LINE_11_KW)
    ax.set_xlabel(r"$\log L_{H\beta,\ \mathrm{true}}$")
    ax.set_ylabel(r"$\log L_{H\beta,\ \mathrm{pred}}$")
    save_square(fig, "inset_02_Hb_pred_vs_true_desi.png")

    # --- (3) DESI in-survey: OIII pred vs true ---
    fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
    hb, lo, hi = square_hexbin(ax, logLoiii_true_desi, logLoiii_pred_desi)
    ax.plot([lo, hi], [lo, hi], **LINE_11_KW)
    ax.set_xlabel(r"$\log L_{\mathrm{[OIII]},\ \mathrm{true}}$")
    ax.set_ylabel(r"$\log L_{\mathrm{[OIII]},\ \mathrm{pred}}$")
    save_square(fig, "inset_03_OIII_pred_vs_true_desi.png")

    # --- (4) Cross-survey: DESI-trained flow on SDSS (OIII) ---
    ratios_desi_on_sdss = sample_ratios_mcmean(flow_desi, meta_desi, df_sdss_p, seed=SEED+20, n_mc=N_MC)

    logLHa_sdss = df_sdss_p["LOG_LHA"].to_numpy(float)
    logLoiii_true_sdss = log10_lum_from_flux(df_sdss_p["Z_1"].to_numpy(float), df_sdss_p["OIII_5007_FLUX"].to_numpy(float))
    logLoiii_pred_sdss = logLHa_sdss + ratios_desi_on_sdss[:, i_oiii].astype(float)

    fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
    hb, lo, hi = square_hexbin(ax, logLoiii_true_sdss, logLoiii_pred_sdss)
    ax.plot([lo, hi], [lo, hi], **LINE_11_KW)
    ax.set_xlabel(r"$\log L_{\mathrm{[OIII]},\ \mathrm{true}}$")
    ax.set_ylabel(r"$\log L_{\mathrm{[OIII]},\ \mathrm{pred}}$")
    save_square(fig, "inset_04_OIII_pred_vs_true_desi_to_sdss.png")


if __name__ == "__main__":
    main()
