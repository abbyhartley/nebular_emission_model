"""
Covariance matrices of the 8 target log line ratios log10(L_line/L_Hα), for:
  (top-left)  SDSS training data        (top-right)  SDSS-trained NF samples
  (bot-left)  DESI training data        (bot-right)  DESI-trained NF samples
NF samples are drawn at the native survey's (logM*, logL_Ha) conditioning.
Shared diverging color scale so structure is directly comparable across panels.
"""
from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import cmasher as cmr

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
SDSS_FITS = Path(BASE + "SDSS_main_training_data.fits")
DESI_FITS = Path(BASE + "DESI_BGS_training_data.fits")
FLOW_SDSS, META_SDSS = Path(REPO + "nf_sdss_main.eqx"), Path(REPO + "nf_sdss_main_meta.pkl")
FLOW_DESI, META_DESI = Path(REPO + "nf_desi_bgs.eqx"), Path(REPO + "nf_desi_bgs_meta.pkl")

FLUX_SCALE = 1e-17   # cancels in the ratio; kept for clarity
N_SAMP = 100_000
SEED = 0

LABELS = [r"H$\beta$", r"H$\gamma$", r"[N\,II]", r"[S\,II]$_{6716}$",
          r"[S\,II]$_{6731}$", r"[O\,II]$_{3726}$", r"[O\,II]$_{3729}$", r"[O\,III]"]


def load_scalar_df(p):
    t = Table.read(p, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()


def thin_df(df, n, seed=0):
    return df.reset_index(drop=True) if len(df) <= n else df.sample(n=n, random_state=seed).reset_index(drop=True)


def add_loglha(df, *, survey):
    df = df.copy()
    if survey == "sdss":
        z = df["Z_1"].to_numpy(float); ha = df["H_ALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    else:
        z = df["Z"].to_numpy(float); ha = df["HALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    m = np.isfinite(z) & (z > 0) & np.isfinite(ha) & (ha > 0)
    loglha = np.full(len(df), np.nan)
    loglha[m] = np.log10(ha[m]) + np.log10(4 * np.pi) + 2 * np.log10(cosmo.luminosity_distance(z[m]).to("cm").value)
    df["LOG_LHA"] = loglha
    return df


def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    template = block_neural_autoregressive_flow(
        key=jr.key(int(meta.get("seed", 0))), base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    return eqx.tree_deserialise_leaves(flow_path, template)


def sample_log_ratios(flow, meta, df_cond, *, seed):
    logm, loglha = meta["resolved"]["logmstar_col"], meta["resolved"]["loglha_col"]
    m = np.isfinite(df_cond[logm].to_numpy(float)) & np.isfinite(df_cond[loglha].to_numpy(float))
    df = df_cond.loc[m].reset_index(drop=True)
    U = (df[[logm, loglha]].to_numpy(np.float32) - meta["U_mean"]) / meta["U_std"]
    keys = jr.split(jr.key(seed + 1234), len(df))
    Xn = jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, jnp.asarray(U))
    return np.array(Xn) * meta["X_std"] + meta["X_mean"]      # (N, 8) in dex


def data_log_ratios(df, meta, *, survey):
    # meta target/Ha cols are stored as LOG10_<...>_FLUX (logged in a wrapper); the FITS holds
    # the raw <...>_FLUX columns, so strip the LOG10_ prefix to recover them.
    tcols = meta["resolved"]["target_cols"]                    # 8 cols, out_cols order
    raw = [c[6:] if c.startswith("LOG10_") else c for c in tcols]
    hacol = meta["resolved"].get("logha_col") or ""
    ha_candidates = [hacol[6:] if hacol.startswith("LOG10_") else hacol,
                     "H_ALPHA_FLUX", "HALPHA_FLUX", "HA_FLUX"]
    ha = next((c for c in ha_candidates if c in df.columns), None)
    if ha is None:
        raise KeyError(f"no Ha flux column found among {ha_candidates}")
    missing = [c for c in raw if c not in df.columns]
    if missing:
        avail = [c for c in df.columns if "FLUX" in c.upper()][:25]
        raise KeyError(f"missing raw flux cols {missing}; available e.g. {avail}")
    F = np.column_stack([df[c].to_numpy(float) for c in raw])
    fha = df[ha].to_numpy(float)
    good = np.all(F > 0, axis=1) & (fha > 0) & np.isfinite(fha) & np.all(np.isfinite(F), axis=1)
    return np.log10(F[good]) - np.log10(fha[good])[:, None]    # (Ngood, 8) in dex


def main():
    df_sdss = add_loglha(thin_df(load_scalar_df(SDSS_FITS), N_SAMP, seed=SEED), survey="sdss")
    df_desi = add_loglha(thin_df(load_scalar_df(DESI_FITS), N_SAMP, seed=SEED + 1), survey="desi")
    meta_s = pickle.load(open(META_SDSS, "rb")); meta_d = pickle.load(open(META_DESI, "rb"))
    flow_s, flow_d = load_flow(FLOW_SDSS, meta_s), load_flow(FLOW_DESI, meta_d)

    cov = {
        "sdss_data": np.cov(data_log_ratios(df_sdss, meta_s, survey="sdss"), rowvar=False),
        "sdss_nf":   np.cov(sample_log_ratios(flow_s, meta_s, df_sdss, seed=SEED + 10), rowvar=False),
        "desi_data": np.cov(data_log_ratios(df_desi, meta_d, survey="desi"), rowvar=False),
        "desi_nf":   np.cov(sample_log_ratios(flow_d, meta_d, df_desi, seed=SEED + 20), rowvar=False),
    }
    vmax = max(np.abs(m).max() for m in cov.values())
    print("shared |cov| max =", vmax)

    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 13,
                         "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.4), constrained_layout=True)
    panels = [("sdss_data", "SDSS data", axes[0, 0]), ("sdss_nf", "SDSS-trained NF", axes[0, 1]),
              ("desi_data", "DESI data", axes[1, 0]), ("desi_nf", "DESI-trained NF", axes[1, 1])]
    n = len(LABELS)
    for key, title, ax in panels:
        M = cov[key]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(LABELS, rotation=45, ha="right")
        ax.set_yticklabels(LABELS)
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.2,
                        color="white" if abs(v) > 0.55 * vmax else "black")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, location="right", pad=0.02)
    cbar.set_label(r"Covariance of $\log_{10}(L_{\rm line}/L_{\mathrm{H}\alpha})$  [dex$^2$]", fontsize=13)
    fig.suptitle(r"Covariance of the 8 target log line ratios", fontsize=15)
    for ext in ("png", "pdf"):
        Path(REPO + "figs").mkdir(exist_ok=True)
        out = REPO + f"figs/covariance_matrices.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("Wrote:", out)


if __name__ == "__main__":
    main()
