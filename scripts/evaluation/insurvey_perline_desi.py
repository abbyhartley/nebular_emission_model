#!/usr/bin/env python3
"""
Per-line in-survey DESI->DESI metrics (Hb, [OII]3726, [OII]3729, [OIII]5007), using the SAME
metric definitions and MC-mean prediction as the high-z test, so the two are directly comparable.
"""
import pickle, numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr
from functools import partial as _pf
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO = BASE + "nebular_emission_model/"
FLUX_SCALE = 1e-17; NMC = 40; N_EVAL = 40000
_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))
LINE_ALIASES = [["HBETA_FLUX"], ["HGAMMA_FLUX"], ["NII_6584_FLUX"], ["SII_6716_FLUX"],
                ["SII_6731_FLUX"], ["OII_3726_FLUX"], ["OII_3729_FLUX"], ["OIII_5007_FLUX"]]
FOUR = [("Hbeta", 0), ("OII3726", 5), ("OII3729", 6), ("OIII5007", 7)]


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


meta = pickle.load(open(REPO + "models/nf_desi_ALTB_meta.pkl", "rb"))
xdim = len(meta["resolved"]["out_cols"])
tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                        base_dist=Normal(jnp.zeros(xdim)), cond_dim=2, inverter=_INV)
flow = eqx.tree_deserialise_leaves(REPO + "models/nf_desi_ALTB.eqx", tmpl)

t = Table.read(BASE + "DESI_BGS_training_data_ALTB.fits", hdu=1)
df = t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()
z = df["Z"].to_numpy(float); ha = df["HALPHA_FLUX"].to_numpy(float)
logm = df["LOGM_COLOR"].to_numpy(float); loglha = log10_lum(z, ha)
F = np.column_stack([df[a[0]].to_numpy(float) for a in LINE_ALIASES])
m = (np.isfinite(z) & (z > 0) & np.isfinite(logm) & np.isfinite(loglha) & (ha > 0)
     & np.all(F > 0, axis=1) & np.all(np.isfinite(F), axis=1))
F, ha, logm, loglha = F[m], ha[m], logm[m], loglha[m]
true = np.log10(F) - np.log10(ha)[:, None]
rng = np.random.default_rng(3)
idx = np.arange(len(ha)) if len(ha) <= N_EVAL else rng.choice(len(ha), N_EVAL, replace=False)
logm, loglha, true = logm[idx], loglha[idx], true[idx]

U = np.column_stack([logm, loglha]).astype(np.float32)
Un = jnp.asarray((U - meta["U_mean"]) / meta["U_std"])
Xm, Xs = np.asarray(meta["X_mean"]), np.asarray(meta["X_std"])
acc = np.zeros((len(U), 8)); key = jr.key(9)
for _ in range(NMC):
    key, sk = jr.split(key); keys = jr.split(sk, len(U))
    Xnn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un))
    acc += Xnn * Xs + Xm
pred = acc / NMC

print("=== IN-SURVEY DESI->DESI per-line (N=%d, MC-mean, same defs as high-z) ===" % len(U))
for name, j in FOUR:
    r = pred[:, j] - true[:, j]
    b = np.median(r); s = 0.5 * (np.percentile(r, 84) - np.percentile(r, 16))
    rm = np.sqrt(np.mean(r**2)); rho = spearmanr(pred[:, j], true[:, j]).correlation
    print("   %-9s bias=%+.3f scat=%.3f rmse=%.3f rho=%.3f" % (name, b, s, rm, rho))
print("=== DONE ===")
