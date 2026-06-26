"""
Balmer-decrement distribution figure (review item 2.10 / Fig. for the paper).

Two panels: left = flows/data evaluated on the DESI conditioning distribution,
right = on the SDSS conditioning distribution. Each panel overlays the observed
decrement, the in-survey flow, and the cross-survey flow, with the case-B floor
R=2.86 marked. Annotates frac(R<2.86) for each distribution.
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
import scienceplots  # noqa: F401  (registers the "science" style)

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
SDSS_FITS = Path(BASE + "SDSS_main_training_data.fits")
DESI_FITS = Path(BASE + "DESI_BGS_training_data.fits")
FLOW_SDSS, META_SDSS = Path(REPO + "nf_sdss_main.eqx"), Path(REPO + "nf_sdss_main_meta.pkl")
FLOW_DESI, META_DESI = Path(REPO + "nf_desi_bgs.eqx"), Path(REPO + "nf_desi_bgs_meta.pkl")

FLUX_SCALE = 1e-17
N_SAMP = 100_000
SEED = 0
FLOOR = 2.86


def load_scalar_df(fits_path):
    t = Table.read(fits_path, hdu=1)
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
    template = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                                base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    return eqx.tree_deserialise_leaves(flow_path, template)


def find_dim(out_cols, substrs):
    for i, name in enumerate(out_cols):
        if any(s in name for s in substrs):
            return i
    raise KeyError(substrs)


def sample_R(flow, meta, df_cond, *, seed):
    logm, loglha = meta["resolved"]["logmstar_col"], meta["resolved"]["loglha_col"]
    m = np.isfinite(df_cond[logm].to_numpy(float)) & np.isfinite(df_cond[loglha].to_numpy(float))
    df = df_cond.loc[m].reset_index(drop=True)
    U = (df[[logm, loglha]].to_numpy(np.float32) - meta["U_mean"]) / meta["U_std"]
    keys = jr.split(jr.key(seed + 1234), len(df))
    Xn = jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, jnp.asarray(U))
    ratios = np.array(Xn) * meta["X_std"] + meta["X_mean"]
    i_hb = find_dim(meta["resolved"]["out_cols"], ["H_BETA", "HBETA"])
    return 10.0 ** (-ratios[:, i_hb])     # Ha/Hb = 1 / (Hb/Ha)


def obs_R(df, *, survey):
    if survey == "sdss":
        ha, hb = df["H_ALPHA_FLUX"].to_numpy(float), df["H_BETA_FLUX"].to_numpy(float)
    else:
        ha, hb = df["HALPHA_FLUX"].to_numpy(float), df["HBETA_FLUX"].to_numpy(float)
    m = np.isfinite(ha) & np.isfinite(hb) & (ha > 0) & (hb > 0)
    return ha[m] / hb[m]


def frac_floor(R):
    return float(np.mean(np.asarray(R) < FLOOR))


def main():
    df_sdss = add_loglha(thin_df(load_scalar_df(SDSS_FITS), N_SAMP, seed=SEED), survey="sdss")
    df_desi = add_loglha(thin_df(load_scalar_df(DESI_FITS), N_SAMP, seed=SEED + 1), survey="desi")
    meta_s = pickle.load(open(META_SDSS, "rb")); meta_d = pickle.load(open(META_DESI, "rb"))
    flow_s, flow_d = load_flow(FLOW_SDSS, meta_s), load_flow(FLOW_DESI, meta_d)

    R = {
        "obs_sdss": obs_R(df_sdss, survey="sdss"),
        "obs_desi": obs_R(df_desi, survey="desi"),
        "ss": sample_R(flow_s, meta_s, df_sdss, seed=SEED + 10),     # SDSS->SDSS in-survey
        "dd": sample_R(flow_d, meta_d, df_desi, seed=SEED + 20),     # DESI->DESI in-survey
        "sd": sample_R(flow_s, meta_s, df_desi, seed=SEED + 30),     # SDSS->DESI cross
        "ds": sample_R(flow_d, meta_d, df_sdss, seed=SEED + 40),     # DESI->SDSS cross
    }

    # House style: SciencePlots + Okabe-Ito colorblind-safe palette (matches Fig. 2/4),
    # large axis labels (>=15), square-ish panels matching the BPT comparison figure.
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 250,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 15,
        "axes.labelsize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    # Color encodes the comparison ROLE, identically in both panels (the panel title already
    # states the survey). Blue = the native survey: observed data (fill) and in-survey flow
    # (solid line), which overlap closely. Pink dashed = the foreign cross-survey flow.
    # Blue vs pink is colorblind friendly; pink is shared with the BPT figure.
    C_NATIVE = "#0072B2"   # blue: observed data + native in-survey flow
    C_CROSS  = "#CC79A7"   # pink: foreign cross-survey flow

    bins = np.linspace(1.5, 8.0, 70)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.3), sharey=True, constrained_layout=True)

    panels = [
        ("DESI", "obs_desi", "dd", "sd",
         r"Observed DESI", r"DESI$\rightarrow$DESI (in-survey)", r"SDSS$\rightarrow$DESI (cross)",
         C_NATIVE, C_NATIVE, C_CROSS),
        ("SDSS", "obs_sdss", "ss", "ds",
         r"Observed SDSS", r"SDSS$\rightarrow$SDSS (in-survey)", r"DESI$\rightarrow$SDSS (cross)",
         C_NATIVE, C_NATIVE, C_CROSS),
    ]
    for ax, (cond, k_obs, k_in, k_cross, l_obs, l_in, l_cross, c_obs, c_in, c_cross) in zip(axes, panels):
        ax.hist(R[k_obs], bins=bins, density=True, histtype="stepfilled",
                facecolor=c_obs, edgecolor=c_obs, alpha=0.30, lw=1.8,
                label=f"{l_obs} ({frac_floor(R[k_obs]):.1%})")
        ax.hist(R[k_in], bins=bins, density=True, histtype="step",
                color=c_in, lw=2.6, label=f"{l_in} ({frac_floor(R[k_in]):.1%})")
        ax.hist(R[k_cross], bins=bins, density=True, histtype="step",
                color=c_cross, lw=2.6, ls="--", label=f"{l_cross} ({frac_floor(R[k_cross]):.1%})")
        ax.axvspan(bins[0], FLOOR, color="0.5", alpha=0.07, lw=0)
        ax.axvline(FLOOR, color="0.35", ls=":", lw=1.3)
        ax.text(FLOOR - 0.06, ax.get_ylim()[1] * 0.97, "case B (2.86)", rotation=90,
                va="top", ha="right", fontsize=15, color="0.35")
        ax.set_title(f"Evaluated on {cond} conditioning")
        ax.set_xlabel(r"Balmer decrement $R = F_{\mathrm{H}\alpha}/F_{\mathrm{H}\beta}$")
        ax.set_xlim(bins[0], bins[-1])
        ax.legend(frameon=True, framealpha=0.9, loc="upper right",
                  title="Fraction below floor:", title_fontsize=13)
    axes[0].set_ylabel("Normalized density")
    for ext in ("png", "pdf"):
        Path(REPO + "figs").mkdir(exist_ok=True)
        out = REPO + f"figs/balmer_decrement_dist.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print("Wrote:", out)


if __name__ == "__main__":
    main()
