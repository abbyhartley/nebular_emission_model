# eval_alt.py — in-survey per-line stats (RMSE, scatter, Spearman rho) for the baseline
# and alternative-selection DESI flows, plus ALT flows on the STRICT sample (clean-test control).
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
FLUX_SCALE = 1e-17
N_MC = 50

STRICT = f"{GFC}/DESI_BGS_training_data.fits"
COMBOS = [
    ("baseline (strict) on strict", f"{REPO}/nf_desi_bgs.eqx", f"{REPO}/nf_desi_bgs_meta.pkl", STRICT),
    ("ALT-A on ALT-A",  f"{REPO}/nf_desi_ALT_A.eqx", f"{REPO}/nf_desi_ALT_A_meta.pkl", f"{GFC}/DESI_ALT_A.fits"),
    ("ALT-A on STRICT",  f"{REPO}/nf_desi_ALT_A.eqx", f"{REPO}/nf_desi_ALT_A_meta.pkl", STRICT),
    ("ALT-B on ALT-B",  f"{REPO}/nf_desi_ALT_B.eqx", f"{REPO}/nf_desi_ALT_B_meta.pkl", f"{GFC}/DESI_ALT_B.fits"),
    ("ALT-B on STRICT",  f"{REPO}/nf_desi_ALT_B.eqx", f"{REPO}/nf_desi_ALT_B_meta.pkl", STRICT),
]


def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    return eqx.tree_deserialise_leaves(flow_path, tmpl)


def log10_lum(z, flux1e17):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(flux1e17, float) * FLUX_SCALE) + np.log10(4*np.pi) + 2*np.log10(dl)


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


def flux_col_from_outcol(oc):
    # 'LOG10_HBETA_FLUX_RATIO_TO_HA' -> 'HBETA_FLUX'
    return oc.replace("LOG10_", "").replace("_RATIO_TO_HA", "")


def eval_one(label, flow_path, meta_path, fits_path):
    meta = pickle.load(open(meta_path, "rb"))
    flow = load_flow(flow_path, meta)
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    df = t[names].to_pandas()
    z = df["Z"].to_numpy(float)
    logm = df["LOGM_COLOR"].to_numpy(float)
    loglha = log10_lum(z, df["HALPHA_FLUX"].to_numpy(float))
    m = np.isfinite(z) & (z > 0) & np.isfinite(logm) & np.isfinite(loglha)
    out_cols = meta["resolved"]["out_cols"]
    fcs = [flux_col_from_outcol(oc) for oc in out_cols]
    for fc in fcs:
        m &= np.isfinite(df[fc].to_numpy(float)) & (df[fc].to_numpy(float) > 0)
    df, z, logm, loglha = df[m].reset_index(drop=True), z[m], logm[m], loglha[m]
    U = np.column_stack([logm, loglha])
    ratios = sample_ratios(flow, meta, U)
    print(f"\n=== {label}  (N={len(z):,}) ===", flush=True)
    print(f"  {'line':11s} {'rmse':>7s} {'scat':>7s} {'rho':>7s}", flush=True)
    rmses, scats, rhos = [], [], []
    for j, oc in enumerate(out_cols):
        true = log10_lum(z, df[fcs[j]].to_numpy(float))
        pred = loglha + ratios[:, j]
        r = pred - true
        rmse = float(np.sqrt(np.mean(r**2)))
        p16, p84 = np.percentile(r, [16, 84]); scat = float(0.5*(p84-p16))
        rho = float(spearmanr(true, pred).correlation)
        name = fcs[j].replace("_FLUX", "")
        rmses.append(rmse); scats.append(scat); rhos.append(rho)
        print(f"  {name:11s} {rmse:7.3f} {scat:7.3f} {rho:7.3f}", flush=True)
    print(f"  {'MEAN':11s} {np.mean(rmses):7.3f} {np.mean(scats):7.3f} {np.mean(rhos):7.3f}", flush=True)


def main():
    for label, fp, mp, ev in COMBOS:
        if not (Path(fp).exists() and Path(mp).exists()):
            print(f"SKIP {label}: flow/meta not found yet", flush=True); continue
        eval_one(label, fp, mp, ev)


if __name__ == "__main__":
    main()
