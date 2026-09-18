# plot_cross_survey_balmer_vs_conditions.py
#
# Makes two diagnostic plots for cross-survey sampling:
#   (1) Balmer decrement R=Hα/Hβ vs logLHa
#   (2) Balmer decrement R=Hα/Hβ vs logM*
#
# for both cross-survey directions:
#   - SDSS-trained flow conditioned on DESI
#   - DESI-trained flow conditioned on SDSS
#
# Outputs PNGs.

from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow


# -----------------------
# Paths
# -----------------------
SDSS_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")
DESI_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

FLOW_SDSS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main.eqx")
META_SDSS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main_meta.pkl")

FLOW_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs.eqx")
META_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs_meta.pkl")

FLUX_SCALE = 1e-17
N_SAMP = 100_000
SEED = 0


# -----------------------
# Plot style
# -----------------------
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 250,
})


# -----------------------
# Helpers
# -----------------------
def load_scalar_df(fits_path):
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()

def thin_df(df, n, seed=0):
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def add_loglha(df, *, survey):
    df = df.copy()
    if survey == "sdss":
        z = df["Z_1"].to_numpy(float)
        ha = df["H_ALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    elif survey == "desi":
        z = df["Z"].to_numpy(float)
        ha = df["HALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    else:
        raise ValueError("survey must be 'sdss' or 'desi'")

    m = np.isfinite(z) & (z > 0) & np.isfinite(ha) & (ha > 0)
    loglha = np.full(len(df), np.nan, dtype=float)
    dl_cm = cosmo.luminosity_distance(z[m]).to("cm").value
    loglha[m] = np.log10(ha[m]) + np.log10(4*np.pi) + 2*np.log10(dl_cm)
    df["LOG_LHA"] = loglha
    return df

def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    key = jr.key(int(meta.get("seed", 0)))
    template = block_neural_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=2,
    )
    return eqx.tree_deserialise_leaves(flow_path, template)

def find_dim(out_cols, substrs):
    for i, name in enumerate(out_cols):
        for s in substrs:
            if s in name:
                return i
    raise KeyError(f"Could not find any of {substrs} in out_cols.")

def sample_log_ratios(flow, meta, df_cond, *, seed=0):
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]

    m = np.isfinite(df_cond[logm_col].to_numpy(float)) & np.isfinite(df_cond[loglha_col].to_numpy(float))
    df = df_cond.loc[m].reset_index(drop=True)

    U = df[[logm_col, loglha_col]].to_numpy(np.float32)
    Un = (U - meta["U_mean"]) / meta["U_std"]
    Un_j = jnp.asarray(Un)

    key = jr.key(seed + 1234)
    keys = jr.split(key, len(df))

    def sample_one(k, u):
        return flow.sample(k, sample_shape=(), condition=u)

    Xn = jax.vmap(sample_one, in_axes=(0, 0))(keys, Un_j)
    Xn = np.array(Xn)
    ratios = Xn * meta["X_std"] + meta["X_mean"]
    return df, ratios

def make_hex(ax, x, y, xlabel, ylabel, title):
    hb = ax.hexbin(x, y, gridsize=80, bins="log", mincnt=1, cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return hb


# -----------------------
# Main
# -----------------------
def main():
    # Load and add LOG_LHA
    df_sdss = add_loglha(thin_df(load_scalar_df(SDSS_FITS), N_SAMP, seed=SEED), survey="sdss")
    df_desi = add_loglha(thin_df(load_scalar_df(DESI_FITS), N_SAMP, seed=SEED+1), survey="desi")

    # Load flows and metas
    with open(META_SDSS, "rb") as f:
        meta_sdss = pickle.load(f)
    with open(META_DESI, "rb") as f:
        meta_desi = pickle.load(f)

    flow_sdss = load_flow(FLOW_SDSS, meta_sdss)
    flow_desi = load_flow(FLOW_DESI, meta_desi)

    # Indices for Hb ratio dim
    i_hb_sdss = find_dim(meta_sdss["resolved"]["out_cols"], ["H_BETA", "HBETA"])
    i_hb_desi = find_dim(meta_desi["resolved"]["out_cols"], ["H_BETA", "HBETA"])

    # Cross-survey samples
    # SDSS-trained conditioned on DESI
    dfA, ratA = sample_log_ratios(flow_sdss, meta_sdss, df_desi, seed=SEED+10)
    R_A = 10.0 ** (-ratA[:, i_hb_sdss])  # Hα/Hβ
    logLHa_A = dfA["LOG_LHA"].to_numpy(float)
    logM_A = dfA[meta_sdss["resolved"]["logmstar_col"]].to_numpy(float)

    # DESI-trained conditioned on SDSS
    dfB, ratB = sample_log_ratios(flow_desi, meta_desi, df_sdss, seed=SEED+20)
    R_B = 10.0 ** (-ratB[:, i_hb_desi])
    logLHa_B = dfB["LOG_LHA"].to_numpy(float)
    logM_B = dfB[meta_desi["resolved"]["logmstar_col"]].to_numpy(float)

    # Plot: R vs logLHa
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)

    hb1 = make_hex(
        axes[0],
        logLHa_A, R_A,
        xlabel=r"$\log L_{H\alpha}$",
        ylabel=r"$H\alpha/H\beta$",
        title="SDSS-trained flow conditioned on DESI",
    )
    axes[0].axhline(2.86, color="crimson", lw=1.5, ls="--", label="Case B (2.86)")
    axes[0].legend(frameon=True, fontsize=9)

    hb2 = make_hex(
        axes[1],
        logLHa_B, R_B,
        xlabel=r"$\log L_{H\alpha}$",
        ylabel=r"$H\alpha/H\beta$",
        title="DESI-trained flow conditioned on SDSS",
    )
    axes[1].axhline(2.86, color="crimson", lw=1.5, ls="--")

    cbar = fig.colorbar(hb2, ax=axes.ravel().tolist(), location="right", shrink=0.95, pad=0.01)
    cbar.set_label("log10(N)")

    out1 = "balmer_decrement_crosssurvey_vs_logLHa.png"
    fig.savefig(out1)
    print("Saved:", out1)
    plt.close(fig)

    # Plot: R vs logM
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)

    hb1 = make_hex(
        axes[0],
        logM_A, R_A,
        xlabel=r"$\log M_\star$",
        ylabel=r"$H\alpha/H\beta$",
        title="SDSS-trained flow conditioned on DESI",
    )
    axes[0].axhline(2.86, color="crimson", lw=1.5, ls="--", label="Case B (2.86)")
    axes[0].legend(frameon=True, fontsize=9)

    hb2 = make_hex(
        axes[1],
        logM_B, R_B,
        xlabel=r"$\log M_\star$",
        ylabel=r"$H\alpha/H\beta$",
        title="DESI-trained flow conditioned on SDSS",
    )
    axes[1].axhline(2.86, color="crimson", lw=1.5, ls="--")

    cbar = fig.colorbar(hb2, ax=axes.ravel().tolist(), location="right", shrink=0.95, pad=0.01)
    cbar.set_label("log10(N)")

    out2 = "balmer_decrement_crosssurvey_vs_logMstar.png"
    fig.savefig(out2)
    print("Saved:", out2)
    plt.close(fig)


if __name__ == "__main__":
    main()
