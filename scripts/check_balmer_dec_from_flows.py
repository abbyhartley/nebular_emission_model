# check_balmer_decrement_from_flows.py  (fixed LOG_LHA)
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow


SDSS_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")
DESI_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

FLOW_SDSS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main.eqx")
META_SDSS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main_meta.pkl")

FLOW_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs.eqx")
META_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs_meta.pkl")

FLUX_SCALE = 1e-17
N_SAMP = 100_000
SEED = 0


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
    logm_col = meta["resolved"]["logmstar_col"]   # LOGM_COLOR
    loglha_col = meta["resolved"]["loglha_col"]   # LOG_LHA

    # require finite conditioning
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
    ratios = Xn * meta["X_std"] + meta["X_mean"]  # (N,8)
    return ratios

def summarize(name, R):
    R = np.asarray(R, float)
    R = R[np.isfinite(R)]
    p1, p16, p50, p84, p99 = np.percentile(R, [1, 16, 50, 84, 99])
    frac_lt_2p86 = float(np.mean(R < 2.86))
    frac_gt_5 = float(np.mean(R > 5.0))
    print(f"\n{name}")
    print(f"N={len(R):,}")
    print(f"p1/p16/p50/p84/p99 = {p1:.3f} / {p16:.3f} / {p50:.3f} / {p84:.3f} / {p99:.3f}")
    print(f"frac(R<2.86)={frac_lt_2p86:.3%}  frac(R>5)={frac_gt_5:.3%}")

def observed_balmer_sdss(df):
    ha = df["H_ALPHA_FLUX"].to_numpy(float)
    hb = df["H_BETA_FLUX"].to_numpy(float)
    m = np.isfinite(ha) & np.isfinite(hb) & (ha > 0) & (hb > 0)
    return ha[m] / hb[m]

def observed_balmer_desi(df):
    ha = df["HALPHA_FLUX"].to_numpy(float)
    hb = df["HBETA_FLUX"].to_numpy(float)
    m = np.isfinite(ha) & np.isfinite(hb) & (ha > 0) & (hb > 0)
    if "HALPHA_FLUX_IVAR" in df.columns:
        m &= (df["HALPHA_FLUX_IVAR"].to_numpy(float) > 0)
    if "HBETA_FLUX_IVAR" in df.columns:
        m &= (df["HBETA_FLUX_IVAR"].to_numpy(float) > 0)
    return ha[m] / hb[m]


def main():
    # Load + add LOG_LHA
    df_sdss = add_loglha(thin_df(load_scalar_df(SDSS_FITS), N_SAMP, seed=SEED), survey="sdss")
    df_desi = add_loglha(thin_df(load_scalar_df(DESI_FITS), N_SAMP, seed=SEED+1), survey="desi")

    summarize("Observed SDSS (Hα/Hβ)", observed_balmer_sdss(df_sdss))
    summarize("Observed DESI (Hα/Hβ)", observed_balmer_desi(df_desi))

    # Load flows + metas
    with open(META_SDSS, "rb") as f:
        meta_sdss = pickle.load(f)
    with open(META_DESI, "rb") as f:
        meta_desi = pickle.load(f)

    flow_sdss = load_flow(FLOW_SDSS, meta_sdss)
    flow_desi = load_flow(FLOW_DESI, meta_desi)

    # Sample ratios at native conditions
    ratios_sdss = sample_log_ratios(flow_sdss, meta_sdss, df_sdss, seed=SEED+10)
    ratios_desi = sample_log_ratios(flow_desi, meta_desi, df_desi, seed=SEED+20)

    i_hb_sdss = find_dim(meta_sdss["resolved"]["out_cols"], ["H_BETA", "HBETA"])
    i_hb_desi = find_dim(meta_desi["resolved"]["out_cols"], ["H_BETA", "HBETA"])

    R_sdss_nf = 10.0 ** (-ratios_sdss[:, i_hb_sdss])
    R_desi_nf = 10.0 ** (-ratios_desi[:, i_hb_desi])

    summarize("NF samples: SDSS-trained (conditioned on SDSS)", R_sdss_nf)
    summarize("NF samples: DESI-trained (conditioned on DESI)", R_desi_nf)

    # Optional: cross-survey Balmer decrement (often revealing)
    ratios_sdss_on_desi = sample_log_ratios(flow_sdss, meta_sdss, df_desi, seed=SEED+30)
    ratios_desi_on_sdss = sample_log_ratios(flow_desi, meta_desi, df_sdss, seed=SEED+40)
    R_sdss_on_desi = 10.0 ** (-ratios_sdss_on_desi[:, i_hb_sdss])
    R_desi_on_sdss = 10.0 ** (-ratios_desi_on_sdss[:, i_hb_desi])

    summarize("NF samples: SDSS-trained (conditioned on DESI)", R_sdss_on_desi)
    summarize("NF samples: DESI-trained (conditioned on SDSS)", R_desi_on_sdss)

    out = pd.DataFrame({
        "R_sdss_obs": observed_balmer_sdss(df_sdss),
        "R_desi_obs": observed_balmer_desi(df_desi),
    })
    out.to_csv("balmer_obs_samples.csv", index=False)
    print("\nWrote: balmer_obs_samples.csv")


if __name__ == "__main__":
    main()
