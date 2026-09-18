# eval_altb_stats.py — in-survey per-line stats (RMSE, scatter, Spearman rho) for the
# ALT-B flows on their ALT-B samples, with the strict baselines alongside. Both surveys.
import pickle
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow

REPO = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model"
GFC = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs"
FLUX_SCALE = 1e-17; N_MC = 50

COMBOS = [
    ("DESI ALT-B",           f"{REPO}/models/nf_desi_ALTB.eqx",  f"{REPO}/models/nf_desi_ALTB_meta.pkl",  f"{GFC}/DESI_BGS_training_data_ALTB.fits"),
    ("DESI strict (baseline)", f"{REPO}/nf_desi_bgs.eqx", f"{REPO}/nf_desi_bgs_meta.pkl",   f"{GFC}/DESI_BGS_training_data.fits"),
    ("SDSS ALT-B",           f"{REPO}/models/nf_sdss_ALTB.eqx",  f"{REPO}/models/nf_sdss_ALTB_meta.pkl",  f"{GFC}/SDSS_main_training_data_ALTB.fits"),
    ("SDSS strict (baseline)", f"{REPO}/nf_sdss_main.eqx", f"{REPO}/nf_sdss_main_meta.pkl", f"{GFC}/SDSS_main_training_data.fits"),
]


def load_flow(fp, meta):
    xdim = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    return eqx.tree_deserialise_leaves(fp, tmpl)


def log10_lum(z, f1e17):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f1e17, float) * FLUX_SCALE) + np.log10(4*np.pi) + 2*np.log10(dl)


def sample_ratios(flow, meta, U, seed=0, n_mc=N_MC, batch=200_000):
    Un_all = jnp.asarray(((U - meta["U_mean"]) / meta["U_std"]).astype(np.float32))
    Xm, Xs = meta["X_mean"], meta["X_std"]; n = len(U)
    out = np.zeros((n, len(meta["resolved"]["out_cols"])))
    key = jr.key(seed + 7)
    for lo in range(0, n, batch):
        hi = min(n, lo + batch); Un = Un_all[lo:hi]; acc = np.zeros((hi-lo, out.shape[1]))
        for _ in range(n_mc):
            key, sk = jr.split(key); keys = jr.split(sk, hi-lo)
            Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un))
            acc += Xn * Xs + Xm
        out[lo:hi] = acc / n_mc
    return out


def eval_one(label, fp, mp, fits_path):
    meta = pickle.load(open(mp, "rb")); flow = load_flow(fp, meta)
    t = Table.read(fits_path, hdu=1)
    df = t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()
    z_col = "Z" if "Z" in df.columns else "Z_1"
    ha_col = "HALPHA_FLUX" if "HALPHA_FLUX" in df.columns else "H_ALPHA_FLUX"
    z = df[z_col].to_numpy(float); logm = df["LOGM_COLOR"].to_numpy(float)
    loglha = log10_lum(z, df[ha_col].to_numpy(float))
    out_cols = meta["resolved"]["out_cols"]
    fcs = [oc.replace("LOG10_", "").replace("_RATIO_TO_HA", "") for oc in out_cols]
    m = np.isfinite(z) & (z > 0) & np.isfinite(logm) & np.isfinite(loglha)
    for fc in fcs:
        m &= np.isfinite(df[fc].to_numpy(float)) & (df[fc].to_numpy(float) > 0)
    df, z, logm, loglha = df[m].reset_index(drop=True), z[m], logm[m], loglha[m]
    ratios = sample_ratios(flow, meta, np.column_stack([logm, loglha]))
    print(f"\n=== {label}  (N={len(z):,}) ===", flush=True)
    print(f"  {'line':10s} {'rmse':>7s} {'scat':>7s} {'rho':>7s}", flush=True)
    R, S, P = [], [], []
    for j, oc in enumerate(out_cols):
        true = log10_lum(z, df[fcs[j]].to_numpy(float)); pred = loglha + ratios[:, j]
        r = pred - true; rmse = float(np.sqrt(np.mean(r**2)))
        p16, p84 = np.percentile(r, [16, 84]); scat = float(0.5*(p84-p16))
        rho = float(spearmanr(true, pred).correlation)
        R.append(rmse); S.append(scat); P.append(rho)
        print(f"  {fcs[j].replace('_FLUX',''):10s} {rmse:7.3f} {scat:7.3f} {rho:7.3f}", flush=True)
    print(f"  {'MEAN':10s} {np.mean(R):7.3f} {np.mean(S):7.3f} {np.mean(P):7.3f}", flush=True)


def main():
    for label, fp, mp, ev in COMBOS:
        if not (Path(fp).exists() and Path(mp).exists() and Path(ev).exists()):
            print(f"SKIP {label}: missing file", flush=True); continue
        eval_one(label, fp, mp, ev)


if __name__ == "__main__":
    main()
