"""
Correlation-matrix figure (paper) -- v2 design.
2x2 grid, 10x10 matrices (8 target log line ratios + logM* + log L_Halpha):
  Left column  = REAL data correlation      (SDSS top, DESI bottom)
  Right column = NF - data difference        (SDSS-NF - SDSS-data ; DESI-NF - DESI-data)
Data and NF matrices are computed over the SAME galaxies / conditioning, so the
right column isolates how well each flow reproduces its survey's joint structure.
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
from functools import partial as _pf
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

_ROBUST_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
SDSS_FITS = Path(BASE + "SDSS_main_training_data_ALTB.fits")
DESI_FITS = Path(BASE + "DESI_BGS_training_data_ALTB.fits")
FLOW_SDSS, META_SDSS = Path(REPO + "models/nf_sdss_ALTB.eqx"), Path(REPO + "models/nf_sdss_ALTB_meta.pkl")
FLOW_DESI, META_DESI = Path(REPO + "models/nf_desi_ALTB.eqx"), Path(REPO + "models/nf_desi_ALTB_meta.pkl")
OUT = REPO + "figs_ALTB/correlation_matrices_data_and_diff"

FLUX_SCALE = 1e-17
N_SAMP = 60_000
SEED = 0
LAB10 = [r"H$\beta$", r"H$\gamma$", r"[N II]", r"[S II]$_{6716}$", r"[S II]$_{6731}$",
         r"[O II]$_{3726}$", r"[O II]$_{3729}$", r"[O III]", r"$\log M_\star$", r"$\log L_{H\alpha}$"]


def load_scalar_df(p):
    t = Table.read(p, hdu=1)
    return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()

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
        key=jr.key(int(meta.get("seed", 0))), base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=2, inverter=_ROBUST_INV)
    return eqx.tree_deserialise_leaves(flow_path, template)

def prep(df, meta):
    """Common-mask data log-ratios (N,8), logM*, log L_Ha, and the masked df for NF conditioning."""
    raw = [c[6:] if c.startswith("LOG10_") else c for c in meta["resolved"]["target_cols"]]
    ha_col = next(c for c in ["H_ALPHA_FLUX", "HALPHA_FLUX", "HA_FLUX"] if c in df.columns)
    lm_col, ll_col = meta["resolved"]["logmstar_col"], meta["resolved"]["loglha_col"]
    F = np.column_stack([df[c].to_numpy(float) for c in raw]); fha = df[ha_col].to_numpy(float)
    lm = df[lm_col].to_numpy(float); ll = df[ll_col].to_numpy(float)
    good = (np.all(F > 0, axis=1) & (fha > 0) & np.isfinite(fha) & np.all(np.isfinite(F), axis=1)
            & np.isfinite(lm) & np.isfinite(ll))
    r8 = np.log10(F[good]) - np.log10(fha[good])[:, None]
    return r8, lm[good], ll[good], df.loc[good].reset_index(drop=True)

def sample_nf(flow, meta, df, *, seed):
    lm, ll = meta["resolved"]["logmstar_col"], meta["resolved"]["loglha_col"]
    U = (df[[lm, ll]].to_numpy(np.float32) - meta["U_mean"]) / meta["U_std"]
    keys = jr.split(jr.key(seed + 1234), len(df))
    Xn = jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, jnp.asarray(U))
    return np.array(Xn) * meta["X_std"] + meta["X_mean"]

def corr10(r8, lm, ll):
    X = np.column_stack([r8, lm, ll])
    return np.corrcoef(X, rowvar=False)


def matrices(fits, flow_path, meta_path, survey, seed):
    meta = pickle.load(open(meta_path, "rb"))
    flow = load_flow(flow_path, meta)
    df = add_loglha(thin_df(load_scalar_df(fits), N_SAMP, seed=seed), survey=survey)
    r8, lm, ll, dfm = prep(df, meta)
    nf8 = sample_nf(flow, meta, dfm, seed=seed + 10)
    C_data = corr10(r8, lm, ll)
    C_nf = corr10(nf8, lm, ll)
    return C_data, C_nf, C_nf - C_data


DIFF_V = 0.30   # difference-column color range (max |dr| is 0.27, so nothing saturates)


def add_matrix(ax, M, title, *, cmap="RdBu_r", vmin=-1, vmax=1, absval=False):
    n = M.shape[0]
    D = np.abs(M) if absval else M
    im = ax.imshow(D, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(LAB10, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(LAB10, fontsize=9)
    denom = max(abs(vmin), abs(vmax))
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if np.isfinite(v):
                shade = abs(D[i, j]) / denom
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if shade > 0.6 else "black")
    return im


def make_figure(mats, mode, out):
    Cs_data, Cs_nf, Ds, Cd_data, Cd_nf, Dd, ms, md = mats
    plt.style.use(["science", "no-latex"])
    fig, ax = plt.subplots(2, 3, figsize=(18.0, 11.6), constrained_layout=True)
    imc = add_matrix(ax[0, 0], Cs_data, "SDSS data")
    add_matrix(ax[0, 1], Cs_nf, "SDSS-trained NF")
    add_matrix(ax[1, 0], Cd_data, "DESI data")
    add_matrix(ax[1, 1], Cd_nf, "DESI-trained NF")
    if mode == "diff":
        imd = add_matrix(ax[0, 2], Ds, rf"SDSS NF $-$ data  (mean $|\Delta r|={ms:.2f}$)",
                         cmap="PuOr_r", vmin=-DIFF_V, vmax=DIFF_V)
        add_matrix(ax[1, 2], Dd, rf"DESI NF $-$ data  (mean $|\Delta r|={md:.2f}$)",
                   cmap="PuOr_r", vmin=-DIFF_V, vmax=DIFF_V)
        clab = rf"$r_{{\rm NF}}-r_{{\rm data}}$   (range $\pm{DIFF_V}$; data $r$ span $\pm0.9$)"
    else:  # closeness: |dr| white->dark
        imd = add_matrix(ax[0, 2], Ds, rf"SDSS $|$NF $-$ data$|$  (mean ${ms:.2f}$)",
                         cmap="Purples", vmin=0, vmax=DIFF_V, absval=True)
        add_matrix(ax[1, 2], Dd, rf"DESI $|$NF $-$ data$|$  (mean ${md:.2f}$)",
                   cmap="Purples", vmin=0, vmax=DIFF_V, absval=True)
        clab = rf"$|r_{{\rm NF}}-r_{{\rm data}}|$   (white = match; range 0--{DIFF_V})"
    cb1 = fig.colorbar(imc, ax=[ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1]],
                       location="bottom", shrink=0.55, pad=0.05, aspect=45)
    cb1.set_label(r"Pearson correlation $r$", fontsize=13)
    cb2 = fig.colorbar(imd, ax=[ax[0, 2], ax[1, 2]], location="bottom", shrink=0.9, pad=0.05)
    cb2.set_label(clab, fontsize=12)
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print("Wrote:", out + ".png", flush=True)


def make_scatter(Cs_d, Cs_n, Cd_d, Cd_n, out):
    iu = np.triu_indices(10, 1)
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    for lab, Cd, Cn, col in [("SDSS", Cs_d, Cs_n, "#0072B2"), ("DESI", Cd_d, Cd_n, "#E69F00")]:
        xd, yn = Cd[iu], Cn[iu]
        rmse = np.sqrt(np.mean((yn - xd) ** 2))
        ax.scatter(xd, yn, s=34, alpha=0.75, color=col, edgecolor="none",
                   label=f"{lab}  (RMSE $={rmse:.3f}$)")
    ax.plot([-1, 1], [-1, 1], "k--", lw=1.3, alpha=0.85)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"data correlation  $r_{\rm data}$", fontsize=15)
    ax.set_ylabel(r"flow correlation  $r_{\rm NF}$", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=13, loc="upper left", frameon=True)
    ax.set_title(r"Flow vs data: all off-diagonal correlations (8 lines + $M_\star$, $L_{H\alpha}$)", fontsize=13)
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print("Wrote:", out + ".png", flush=True)


def main():
    Cs_data, Cs_nf, Ds = matrices(SDSS_FITS, FLOW_SDSS, META_SDSS, "sdss", SEED)
    Cd_data, Cd_nf, Dd = matrices(DESI_FITS, FLOW_DESI, META_DESI, "desi", SEED + 1)
    iu = np.triu_indices(10, 1)
    stats = {}
    for tag, D in [("SDSS", Ds), ("DESI", Dd)]:
        a = np.abs(D[iu])
        stats[tag] = a.mean()
        print(f"{tag}: mean|dr|={a.mean():.3f}  median={np.median(a):.3f}  max={a.max():.3f}  "
              f"frac<0.05={np.mean(a < 0.05):.2f}  frac<0.10={np.mean(a < 0.10):.2f}", flush=True)
    ms, md = stats["SDSS"], stats["DESI"]
    mats = (Cs_data, Cs_nf, Ds, Cd_data, Cd_nf, Dd, ms, md)

    make_scatter(Cs_data, Cs_nf, Cd_data, Cd_nf, REPO + "figs_ALTB/corr_scatter_nf_vs_data")
    make_figure(mats, "diff", REPO + "figs_ALTB/corr_matrices_data_nf_diff")
    make_figure(mats, "closeness", REPO + "figs_ALTB/corr_matrices_data_nf_closeness")


if __name__ == "__main__":
    main()
