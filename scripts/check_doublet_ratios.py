"""
Review item 2.8: density-sensitive doublet-ratio physical-bounds check.

The [S II] 6716/6731 and [O II] 3729/3726 ratios are set by atomic physics and are
bounded between the low-density (n_e -> 0) and high-density (n_e -> infinity) limits
(Osterbrock & Ferland 2006, T_e ~ 1e4 K):
    [S II] 6716/6731 : [0.44 (high-n_e), 1.45 (low-n_e)]
    [O II] 3729/3726 : [0.35 (high-n_e), 1.47 (low-n_e)]

We compute these ratios from (a) the observed selected samples and (b) samples drawn
from each flow (in-survey and cross-survey), and report the fraction that fall outside
the physical band. Because the flow learns the *noise-convolved* observed distribution,
the relevant baseline is the observed out-of-bounds fraction, not zero. This tests the
"split-[O II] heavy tails" hypothesis from the review.
"""
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

FLUX_SCALE = 1e-17

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
SDSS_FITS = Path(BASE + "SDSS_main_training_data.fits")
DESI_FITS = Path(BASE + "DESI_BGS_training_data.fits")
FLOW_SDSS = Path(REPO + "nf_sdss_main.eqx")
META_SDSS = Path(REPO + "nf_sdss_main_meta.pkl")
FLOW_DESI = Path(REPO + "nf_desi_bgs.eqx")
META_DESI = Path(REPO + "nf_desi_bgs_meta.pkl")

N_SAMP = 100_000
SEED = 0

# physical bounds: (low, high) = (high-n_e limit, low-n_e limit)
SII_LO, SII_HI = 0.44, 1.45     # 6716/6731
OII_LO, OII_HI = 0.35, 1.47     # 3729/3726

_ROWS = []


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
    else:
        z = df["Z"].to_numpy(float)
        ha = df["HALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    m = np.isfinite(z) & (z > 0) & np.isfinite(ha) & (ha > 0)
    loglha = np.full(len(df), np.nan, dtype=float)
    dl_cm = cosmo.luminosity_distance(z[m]).to("cm").value
    loglha[m] = np.log10(ha[m]) + np.log10(4 * np.pi) + 2 * np.log10(dl_cm)
    df["LOG_LHA"] = loglha
    return df


def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    key = jr.key(int(meta.get("seed", 0)))
    template = block_neural_autoregressive_flow(
        key=key, base_dist=Normal(jnp.zeros(xdim)), cond_dim=2,
    )
    return eqx.tree_deserialise_leaves(flow_path, template)


def find_dim(out_cols, substr):
    for i, name in enumerate(out_cols):
        if substr in name:
            return i
    raise KeyError(f"{substr} not in {out_cols}")


def sample_log_ratios(flow, meta, df_cond, *, seed=0):
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]
    m = np.isfinite(df_cond[logm_col].to_numpy(float)) & np.isfinite(df_cond[loglha_col].to_numpy(float))
    df = df_cond.loc[m].reset_index(drop=True)
    U = df[[logm_col, loglha_col]].to_numpy(np.float32)
    Un = (U - meta["U_mean"]) / meta["U_std"]
    keys = jr.split(jr.key(seed + 1234), len(df))
    Xn = jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, jnp.asarray(Un))
    return np.array(Xn) * meta["X_std"] + meta["X_mean"]


def doublet_from_samples(ratios, meta, *, kind):
    oc = meta["resolved"]["out_cols"]
    if kind == "sii":
        num = find_dim(oc, "SII_6716") if any("SII_6716" in c for c in oc) else find_dim(oc, "SII_6717")
        den = find_dim(oc, "SII_6731")
    else:  # oii  -> 3729 / 3726
        num = find_dim(oc, "OII_3729")
        den = find_dim(oc, "OII_3726")
    return 10.0 ** (ratios[:, num] - ratios[:, den])


def observed_doublet(df, *, kind, survey):
    if kind == "sii":
        ncol = "SII_6716_FLUX" if survey == "desi" else "SII_6717_FLUX"
        dcol = "SII_6731_FLUX"
    else:
        ncol, dcol = "OII_3729_FLUX", "OII_3726_FLUX"
    nf = df[ncol].to_numpy(float)
    dfl = df[dcol].to_numpy(float)
    m = np.isfinite(nf) & np.isfinite(dfl) & (nf > 0) & (dfl > 0)
    return nf[m] / dfl[m]


def report(label, R, lo, hi, *, kind, src):
    R = np.asarray(R, float)
    R = R[np.isfinite(R) & (R > 0)]
    p1, p50, p99 = np.percentile(R, [1, 50, 99])
    f_lo = float(np.mean(R < lo))
    f_hi = float(np.mean(R > hi))
    f_out = f_lo + f_hi
    print(f"{label:42s}  N={len(R):7d}  p1/p50/p99={p1:.3f}/{p50:.3f}/{p99:.3f}  "
          f"frac<{lo}={f_lo:.2%}  frac>{hi}={f_hi:.2%}  OUT={f_out:.2%}")
    _ROWS.append(dict(doublet=kind, source=src, label=label, n=len(R),
                      p1=p1, p50=p50, p99=p99, frac_below=f_lo, frac_above=f_hi, frac_out=f_out))


def main():
    df_sdss = add_loglha(thin_df(load_scalar_df(SDSS_FITS), N_SAMP, seed=SEED), survey="sdss")
    df_desi = add_loglha(thin_df(load_scalar_df(DESI_FITS), N_SAMP, seed=SEED + 1), survey="desi")

    meta_sdss = pickle.load(open(META_SDSS, "rb"))
    meta_desi = pickle.load(open(META_DESI, "rb"))
    flow_sdss = load_flow(FLOW_SDSS, meta_sdss)
    flow_desi = load_flow(FLOW_DESI, meta_desi)

    r_ss = sample_log_ratios(flow_sdss, meta_sdss, df_sdss, seed=SEED + 10)
    r_dd = sample_log_ratios(flow_desi, meta_desi, df_desi, seed=SEED + 20)
    r_sd = sample_log_ratios(flow_sdss, meta_sdss, df_desi, seed=SEED + 30)
    r_ds = sample_log_ratios(flow_desi, meta_desi, df_sdss, seed=SEED + 40)

    for kind, lo, hi, title in [("sii", SII_LO, SII_HI, "[S II] 6716/6731"),
                                ("oii", OII_LO, OII_HI, "[O II] 3729/3726")]:
        print(f"\n===== {title}   physical band [{lo}, {hi}] =====")
        report("Observed SDSS", observed_doublet(df_sdss, kind=kind, survey="sdss"), lo, hi, kind=kind, src="obs_sdss")
        report("Observed DESI", observed_doublet(df_desi, kind=kind, survey="desi"), lo, hi, kind=kind, src="obs_desi")
        report("NF SDSS->SDSS (in-survey)", doublet_from_samples(r_ss, meta_sdss, kind=kind), lo, hi, kind=kind, src="nf_ss")
        report("NF DESI->DESI (in-survey)", doublet_from_samples(r_dd, meta_desi, kind=kind), lo, hi, kind=kind, src="nf_dd")
        report("NF SDSS->DESI (cross)", doublet_from_samples(r_sd, meta_sdss, kind=kind), lo, hi, kind=kind, src="nf_sd")
        report("NF DESI->SDSS (cross)", doublet_from_samples(r_ds, meta_desi, kind=kind), lo, hi, kind=kind, src="nf_ds")

    out = Path(REPO + "docs/doublet_ratio_bounds.csv")
    pd.DataFrame(_ROWS).to_csv(out, index=False)
    print("\nWrote:", out)


if __name__ == "__main__":
    main()
