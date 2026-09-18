"""
Print p16/p50/p84 (and frac below case-B floor) of the Balmer decrement R=Ha/Hb
for the observed data and the in-survey flows, matching plot_balmer_decrement.py
exactly (same N_SAMP, seeds, standardization) so the medians correspond to Fig. 3.
"""
from pathlib import Path
import pickle
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
SDSS_FITS = Path(BASE + "SDSS_main_training_data.fits")
DESI_FITS = Path(BASE + "DESI_BGS_training_data.fits")
FLOW_SDSS, META_SDSS = Path(REPO + "nf_sdss_main.eqx"), Path(REPO + "nf_sdss_main_meta.pkl")
FLOW_DESI, META_DESI = Path(REPO + "nf_desi_bgs.eqx"), Path(REPO + "nf_desi_bgs_meta.pkl")
FLUX_SCALE, N_SAMP, SEED, FLOOR = 1e-17, 100_000, 0, 2.86


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
    return 10.0 ** (-ratios[:, i_hb])

def obs_R(df, *, survey):
    if survey == "sdss":
        ha, hb = df["H_ALPHA_FLUX"].to_numpy(float), df["H_BETA_FLUX"].to_numpy(float)
    else:
        ha, hb = df["HALPHA_FLUX"].to_numpy(float), df["HBETA_FLUX"].to_numpy(float)
    m = np.isfinite(ha) & np.isfinite(hb) & (ha > 0) & (hb > 0)
    return ha[m] / hb[m]


df_sdss = add_loglha(thin_df(load_scalar_df(SDSS_FITS), N_SAMP, seed=SEED), survey="sdss")
df_desi = add_loglha(thin_df(load_scalar_df(DESI_FITS), N_SAMP, seed=SEED + 1), survey="desi")
meta_s = pickle.load(open(META_SDSS, "rb")); meta_d = pickle.load(open(META_DESI, "rb"))
flow_s, flow_d = load_flow(FLOW_SDSS, meta_s), load_flow(FLOW_DESI, meta_d)

R = {
    "Observed SDSS      ": obs_R(df_sdss, survey="sdss"),
    "SDSS->SDSS in-survey": sample_R(flow_s, meta_s, df_sdss, seed=SEED + 10),
    "Observed DESI      ": obs_R(df_desi, survey="desi"),
    "DESI->DESI in-survey": sample_R(flow_d, meta_d, df_desi, seed=SEED + 20),
    "SDSS->DESI cross   ": sample_R(flow_s, meta_s, df_desi, seed=SEED + 30),
    "DESI->SDSS cross   ": sample_R(flow_d, meta_d, df_sdss, seed=SEED + 40),
}
print(f"{'distribution':22s}  {'p16':>6s} {'p50':>6s} {'p84':>6s}   frac<2.86")
for k, r in R.items():
    p16, p50, p84 = np.percentile(r, [16, 50, 84])
    print(f"{k:22s}  {p16:6.3f} {p50:6.3f} {p84:6.3f}   {np.mean(r < FLOOR):6.2%}")
