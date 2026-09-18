# flowchart_inset_plots_ALTB_v4.py
#
# Standalone square inset PNGs for the schematic flow chart (ALT-B model):
#   (1) logM* vs logL(Ha)                       -> inset_v4_01_logM_vs_logLHa_desi.png
#   (2) DESI in-survey:   logL_pred(Hb) vs true -> inset_v4_02_Hb_insurvey_desi.png
#   (3) Cross-survey:     logL_pred(Hb) vs true -> inset_v4_03_Hb_crosssurvey_desi_to_sdss.png
#
# v4: dense = DARK on WHITE background (inverted bubblegum ramp); H-beta (best case).

from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scienceplots  # noqa: F401
import cmasher as cmr

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from functools import partial as _pf
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces
_ROBUST_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12})

# white background at low density -> bubblegum inverted (down to its DARK navy end) at high density
DENSE = LinearSegmentedColormap.from_list(
    "bubblegum_dense", ["white", cmr.bubblegum(0.85), cmr.bubblegum(0.6),
                        cmr.bubblegum(0.35), cmr.bubblegum(0.0)])
MINCNT = 5
LINE_11_KW = dict(color="black", lw=2.0, ls=":", alpha=0.95)

REPO = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/"
OUTDIR = REPO + "figs_ALTB/"
DESI_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data_ALTB.fits")
SDSS_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data_ALTB.fits")
FLOW_DESI = Path(REPO + "models/nf_desi_ALTB.eqx")
META_DESI = Path(REPO + "models/nf_desi_ALTB_meta.pkl")

FLUX_SCALE = 1e-17
SEED = 0
N_MC = 20
N_PLOT = 50_000


def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    template = block_neural_autoregressive_flow(
        key=jr.key(int(meta.get("seed", 0))), base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=2, inverter=_ROBUST_INV)
    return eqx.tree_deserialise_leaves(flow_path, template)

def load_df(p):
    t = Table.read(p, hdu=1)
    return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()

def log10_lum_from_flux(z, flux_1e17):
    f = np.asarray(flux_1e17, float) * FLUX_SCALE
    return np.log10(f) + np.log10(4 * np.pi) + 2 * np.log10(cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value)

def ensure_loglha(df, *, z_col, ha_col):
    df = df.copy()
    df["LOG_LHA"] = log10_lum_from_flux(df[z_col].to_numpy(float), df[ha_col].to_numpy(float))
    return df

def outcol_index(meta, substrs):
    for i, name in enumerate(meta["resolved"]["out_cols"]):
        if any(s in name for s in substrs):
            return i
    raise KeyError(substrs)

def sample_ratios_mcmean(flow, meta, df, *, seed=0, n_mc=20):
    lm, ll = meta["resolved"]["logmstar_col"], meta["resolved"]["loglha_col"]
    Un = jnp.asarray((df[[lm, ll]].to_numpy(np.float32) - meta["U_mean"]) / meta["U_std"])
    key = jr.key(seed + 999)
    acc = np.zeros((len(df), len(meta["resolved"]["out_cols"])), np.float32)
    for _ in range(n_mc):
        key, sk = jr.split(key)
        keys = jr.split(sk, len(df))
        Xn = jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un)
        acc += np.array(Xn) * meta["X_std"] + meta["X_mean"]
    return acc / n_mc

def thin_df(df, n, seed=0):
    return df.reset_index(drop=True) if len(df) <= n else df.sample(n=n, random_state=seed).reset_index(drop=True)

def square_hexbin(ax, x, y, *, gridsize=70):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    xy = np.concatenate([x, y]); q1, q3 = np.percentile(xy, [25, 75]); fe = 3.0 * (q3 - q1)
    kp = xy[(xy >= q1 - fe) & (xy <= q3 + fe)]; lo, hi = float(kp.min()), float(kp.max())
    pad = 0.02 * (hi - lo); lo -= pad; hi += pad
    ax.hexbin(x, y, gridsize=gridsize, extent=(lo, hi, lo, hi), bins="log", mincnt=MINCNT, cmap=DENSE)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    return lo, hi

def save_square(fig, name):
    fig.savefig(OUTDIR + name, dpi=250, bbox_inches="tight", pad_inches=0.02)
    print("Saved:", OUTDIR + name, flush=True); plt.close(fig)


def main():
    df_desi = ensure_loglha(load_df(DESI_FITS), z_col="Z", ha_col="HALPHA_FLUX")
    df_sdss = ensure_loglha(load_df(SDSS_FITS), z_col="Z_1", ha_col="H_ALPHA_FLUX")
    df_desi_p = thin_df(df_desi, N_PLOT, seed=SEED)
    df_sdss_p = thin_df(df_sdss, N_PLOT, seed=SEED + 1)
    meta = pickle.load(open(META_DESI, "rb")); flow = load_flow(FLOW_DESI, meta)
    i_hb = outcol_index(meta, ["H_BETA", "HBETA"])

    # (1) logM* vs logL(Ha)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.hexbin(df_desi_p["LOGM_COLOR"].to_numpy(float), df_desi_p["LOG_LHA"].to_numpy(float),
              gridsize=65, bins="log", mincnt=MINCNT, cmap=DENSE)
    ax.set_facecolor("white"); ax.set_xlabel(r"$\log M_\star$"); ax.set_ylabel(r"$\log L_{H\alpha}$")
    save_square(fig, "inset_v4_01_logM_vs_logLHa_desi.png")

    # (2) DESI in-survey Hb
    r_dd = sample_ratios_mcmean(flow, meta, df_desi_p, seed=SEED + 10, n_mc=N_MC)
    true_dd = log10_lum_from_flux(df_desi_p["Z"].to_numpy(float), df_desi_p["HBETA_FLUX"].to_numpy(float))
    pred_dd = df_desi_p["LOG_LHA"].to_numpy(float) + r_dd[:, i_hb].astype(float)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    lo, hi = square_hexbin(ax, true_dd, pred_dd); ax.plot([lo, hi], [lo, hi], **LINE_11_KW)
    ax.set_xlabel(r"$\log L_{H\beta,\ \mathrm{true}}$"); ax.set_ylabel(r"$\log L_{H\beta,\ \mathrm{pred}}$")
    save_square(fig, "inset_v4_02_Hb_insurvey_desi.png")

    # (3) cross-survey Hb (DESI-trained flow on SDSS)
    r_ds = sample_ratios_mcmean(flow, meta, df_sdss_p, seed=SEED + 20, n_mc=N_MC)
    true_ds = log10_lum_from_flux(df_sdss_p["Z_1"].to_numpy(float), df_sdss_p["H_BETA_FLUX"].to_numpy(float))  # SDSS col
    pred_ds = df_sdss_p["LOG_LHA"].to_numpy(float) + r_ds[:, i_hb].astype(float)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    lo, hi = square_hexbin(ax, true_ds, pred_ds); ax.plot([lo, hi], [lo, hi], **LINE_11_KW)
    ax.set_xlabel(r"$\log L_{H\beta,\ \mathrm{true}}$"); ax.set_ylabel(r"$\log L_{H\beta,\ \mathrm{pred}}$")
    save_square(fig, "inset_v4_03_Hb_crosssurvey_desi_to_sdss.png")


if __name__ == "__main__":
    main()
