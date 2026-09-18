#!/usr/bin/env python3
"""
Compare the high-z DESI-COSMOS validation metrics to paper Table 1, computed the SAME way
(eval_altb_table1.py): ratio space log(L_line/L_Ha), MC-mean n_mc=50, pooled residuals for
RMSE/bias/scatter/NMAD, rho = MEAN of per-line Spearman.
  (1) Reproduce the 4 train->test combos on all 8 lines (sanity vs published table).
  (2) Recompute them on the SAME 4-line subset DESI-COSMOS has: Hb,[OII]3726,[OII]3729,[OIII]5007.
  (3) hiz Tier 1 (rel Ha, 4 lines, from cache) and Tier 2 (rel Hb, 3 lines, recomputed).
"""
import pickle, numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr
from functools import partial as _pf
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces

_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))
BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
FLUX_SCALE = 1e-17; N_EVAL = 100_000; N_MC = 50; MASS_ZP = 0.13
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
IHB, IOII_A, IOII_B, IOIII = 0, 5, 6, 7
SUB4 = [0, 5, 6, 7]  # Hb, OII3726, OII3729, OIII5007 = the DESI-COSMOS-observable lines within the 8
LINE_ALIASES = [["HBETA_FLUX", "H_BETA_FLUX"], ["HGAMMA_FLUX", "H_GAMMA_FLUX"], ["NII_6584_FLUX"],
                ["SII_6716_FLUX", "SII_6717_FLUX"], ["SII_6731_FLUX"], ["OII_3726_FLUX"],
                ["OII_3729_FLUX"], ["OIII_5007_FLUX"]]
FLOWS = {"SDSS": (REPO + "models/nf_sdss_ALTB.eqx", REPO + "models/nf_sdss_ALTB_meta.pkl"),
         "DESI": (REPO + "models/nf_desi_ALTB.eqx", REPO + "models/nf_desi_ALTB_meta.pkl")}
FITS = {"SDSS": BASE + "SDSS_main_training_data_ALTB.fits", "DESI": BASE + "DESI_BGS_training_data_ALTB.fits"}


def sp(m): print(m, flush=True)


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def load_flow(fp, meta):
    xd = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xd)), cond_dim=2, inverter=_INV)
    return eqx.tree_deserialise_leaves(fp, tmpl)


def resolve(df, aliases):
    for c in aliases:
        if c in df.columns:
            return c
    raise KeyError(aliases)


def load_survey(tag, seed):
    t = Table.read(FITS[tag], hdu=1)
    df = t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()
    z_col = "Z_1" if tag == "SDSS" else "Z"; ha_col = "H_ALPHA_FLUX" if tag == "SDSS" else "HALPHA_FLUX"
    z = df[z_col].to_numpy(float); ha = df[ha_col].to_numpy(float)
    logm = df["LOGM_COLOR"].to_numpy(float); loglha = log10_lum(z, ha)
    cols = [resolve(df, a) for a in LINE_ALIASES]
    F = np.column_stack([df[c].to_numpy(float) for c in cols])
    m = (np.isfinite(z) & (z > 0) & np.isfinite(logm) & np.isfinite(loglha) & (ha > 0)
         & np.all(F > 0, axis=1) & np.all(np.isfinite(F), axis=1))
    F, ha, logm, loglha = F[m], ha[m], logm[m], loglha[m]
    tr = np.log10(F) - np.log10(ha)[:, None]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ha)) if len(ha) <= N_EVAL else rng.choice(len(ha), N_EVAL, replace=False)
    return dict(logm=logm[idx], loglha=loglha[idx], ratios=tr[idx])


def mc_mean(flow, meta, logm, loglha, seed, n_mc=N_MC, batch=50_000):
    U = np.column_stack([logm, loglha]).astype(np.float32)
    Un_all = jnp.asarray((U - meta["U_mean"]) / meta["U_std"])
    Xm, Xs = meta["X_mean"], meta["X_std"]; n = len(U)
    out = np.zeros((n, len(meta["resolved"]["out_cols"]))); key = jr.key(seed + 7)
    for lo in range(0, n, batch):
        hi = min(n, lo + batch); Un = Un_all[lo:hi]; acc = np.zeros((hi - lo, out.shape[1]))
        for _ in range(n_mc):
            key, sk = jr.split(key); keys = jr.split(sk, hi - lo)
            Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un))
            acc += Xn * Xs + Xm
        out[lo:hi] = acc / n_mc
    return out


def metrics(pred, true):
    res = (pred - true).ravel()
    rmse = float(np.sqrt(np.mean(res ** 2))); bias = float(np.median(res))
    p16, p84 = np.percentile(res, [16, 84]); scat = float(0.5 * (p84 - p16))
    nmad = float(1.4826 * np.median(np.abs(res - np.median(res))))
    rho_mean = float(np.mean([spearmanr(true[:, j], pred[:, j]).correlation for j in range(pred.shape[1])]))
    rho_pool = float(spearmanr(true.ravel(), pred.ravel()).correlation)
    return rmse, bias, scat, nmad, rho_mean, rho_pool


def row(name, m):
    sp("%-30s RMSE=%.3f  bias=%+.3f  scat=%.3f  NMAD=%.3f  rho_pool=%.3f  rho_mean=%.3f" % (name, m[0], m[1], m[2], m[3], m[5], m[4]))


def decode(a):
    a = np.asarray(a)
    if a.dtype.kind == "S": a = np.char.decode(a)
    return np.char.strip(a.astype(str))


def main():
    metas = {t: pickle.load(open(FLOWS[t][1], "rb")) for t in FLOWS}
    flows = {t: load_flow(FLOWS[t][0], metas[t]) for t in FLOWS}
    data = {"SDSS": load_survey("SDSS", 1), "DESI": load_survey("DESI", 2)}

    sp("\n==== PAPER COMBOS: all 8 lines (should match Table 1) | same 4-line subset [Hb,OII3726,OII3729,OIII] ====")
    for ci, (tr, te) in enumerate([("SDSS", "SDSS"), ("SDSS", "DESI"), ("DESI", "DESI"), ("DESI", "SDSS")]):
        D = data[te]; pred = mc_mean(flows[tr], metas[tr], D["logm"], D["loglha"], seed=100 + ci)
        row("%s->%s  [8 lines]" % (tr, te), metrics(pred, D["ratios"]))
        row("%s->%s  [4-line subset]" % (tr, te), metrics(pred[:, SUB4], D["ratios"][:, SUB4]))

    # ---- hiz Tier 1 (rel Ha, 4 lines) from cache ----
    A = np.load(REPO + "figs_ALTB/hiz_arrays.npz", allow_pickle=True)
    t1o = A["t1_obs"]; t1p = A["t1_predmean"]; t1s = A["t1_sn"]; ib = A["t1_inbox"]
    g4 = ib.copy()
    for c in range(4):
        g4 &= (t1s[:, c] > 3) & np.isfinite(t1o[:, c])
    sp("\n==== hiz DESI-COSMOS (DESI-trained NF; ratio space, same recipe) ====")
    row("Tier1 0.05<z<0.49 [4 lines/Ha]  N=%d" % g4.sum(), metrics(t1p[g4], t1o[g4]))

    # ---- hiz Tier 2 (rel Hb, 3 lines) recomputed ----
    h = fits.open(COSMOS); d1, d6 = h[1].data, h[6].data
    z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
    stype = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float)

    def cf(nm):
        raw = np.asarray(d1[nm + "_FLUX"], float); iv = np.asarray(d1[nm + "_FLUX_IVAR"], float)
        return raw * 10 ** CAL[nm], raw * np.sqrt(np.clip(iv, 0, None))
    Fha, Sha = cf("HALPHA"); Fhb, Shb = cf("HBETA")
    Foa, Soa = cf("OII_3726"); Fob, Sob = cf("OII_3729"); Fo3, So3 = cf("OIII_5007")
    base = (zwarn == 0) & (stype == "GALAXY") & (z > 0.05) & np.isfinite(lpm) & (lpm > 6) & (lpm < 13)
    logm = lpm + MASS_ZP
    calm = base & (z < 0.49) & (Sha > 3) & (Shb > 3) & (Fha > 0) & (Fhb > 0)
    dec = (log10_lum(z, Fha) - log10_lum(z, Fhb))[calm]; mm = logm[calm]
    mb = np.percentile(mm, np.linspace(0, 100, 9)); mc = 0.5 * (mb[:-1] + mb[1:])
    Rm = np.array([np.median(dec[(mm >= a) & (mm < b)]) for a, b in zip(mb[:-1], mb[1:])])
    R_of_M = lambda x: np.interp(x, mc, Rm, left=Rm[0], right=Rm[-1])
    t2 = base & (z >= 0.49) & (z < 1.00) & (Shb > 3) & (Fhb > 0)
    idx = np.where(t2)[0]
    zt = z[idx]; mt = logm[idx]; LHb = log10_lum(zt, Fhb[idx]); LHa = LHb + R_of_M(mt)
    inbox = (mt >= BOX_M[0]) & (mt <= BOX_M[1]) & (LHa >= BOX_L[0]) & (LHa <= BOX_L[1])
    m8 = mc_mean(flows["DESI"], metas["DESI"], mt, LHa, seed=11)
    obs3 = np.column_stack([np.log10(Fo3[idx]) - np.log10(Fhb[idx]),
                            np.log10(Foa[idx]) - np.log10(Fhb[idx]),
                            np.log10(Fob[idx]) - np.log10(Fhb[idx])])
    pred3 = np.column_stack([m8[:, IOIII] - m8[:, IHB], m8[:, IOII_A] - m8[:, IHB], m8[:, IOII_B] - m8[:, IHB]])
    sn3 = np.column_stack([So3[idx], Soa[idx], Sob[idx]])
    g3 = inbox & (Shb[idx] > 3)
    for c in range(3):
        g3 &= (sn3[:, c] > 3) & np.isfinite(obs3[:, c])
    row("Tier2 0.49<z<1.0 [3 lines/Hb]  N=%d" % g3.sum(), metrics(pred3[g3], obs3[g3]))
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
