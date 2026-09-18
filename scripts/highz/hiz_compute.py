#!/usr/bin/env python3
"""
Compute + cache high-z validation arrays (DESI-trained NF on real DESI-COSMOS spectra).
Saves everything needed for plotting so figures can be iterated without re-sampling the flow.

Tier 1 (0.05<z<0.49): observed L_Ha conditioning. 4 observable ratios vs Ha:
    [Hb, OII3726, OII3729, OIII5007]
Tier 2 (0.49<z<1.0):  L_Ha from Hb x 3.90.  [OIII]5007/Hb.
No lightcone here (object-level, real spectra only).
"""
import pickle, numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr
from functools import partial as _pf
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces


def sp(m): print(m, flush=True)


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
OUT = REPO + "figs_ALTB/hiz_arrays.npz"
FLUX_SCALE = 1e-17; MASS_ZP = 0.13; R_DECREMENT = 3.90
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
# indices into the flow's 8-target vector
IHB, INII, ISIIa, ISIIb, IOII_A, IOII_B, IOIII = 0, 2, 3, 4, 5, 6, 7
FOUR = [("Hbeta", IHB), ("OII3726", IOII_A), ("OII3729", IOII_B), ("OIII5007", IOIII)]
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
NMC = 40; KDRAW = 8
_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def load_flow():
    meta = pickle.load(open(REPO + "models/nf_desi_ALTB_meta.pkl", "rb"))
    xdim = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xdim)), cond_dim=2, inverter=_INV)
    return eqx.tree_deserialise_leaves(REPO + "models/nf_desi_ALTB.eqx", tmpl), meta


def sample(flow, meta, logm, loglha, seed, n_mc=NMC, kdraw=KDRAW, batch=40000):
    U = np.column_stack([logm, loglha]).astype(np.float32)
    Un_all = jnp.asarray((U - meta["U_mean"]) / meta["U_std"])
    Xm, Xs = np.asarray(meta["X_mean"]), np.asarray(meta["X_std"]); n = len(U)
    mean = np.zeros((n, 8)); draws = np.zeros((n, kdraw, 8)); key = jr.key(seed + 7)
    for lo in range(0, n, batch):
        hi = min(n, lo + batch); Un = Un_all[lo:hi]; acc = np.zeros((hi - lo, 8))
        for j in range(n_mc):
            key, sk = jr.split(key); keys = jr.split(sk, hi - lo)
            Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un))
            r = Xn * Xs + Xm; acc += r
            if j < kdraw:
                draws[lo:hi, j] = r
        mean[lo:hi] = acc / n_mc
    return mean, draws


def decode(a):
    a = np.asarray(a)
    if a.dtype.kind == "S":
        a = np.char.decode(a)
    return np.char.strip(a.astype(str))


def stat(pred, obs):
    r = pred - obs; g = np.isfinite(r); rr = r[g]
    return (int(g.sum()), float(np.median(rr)),
            float(0.5 * (np.percentile(rr, 84) - np.percentile(rr, 16))),
            float(np.sqrt(np.mean(rr**2))), float(spearmanr(pred[g], obs[g]).correlation))


def main():
    h = fits.open(COSMOS); d1, d6 = h[1].data, h[6].data
    z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
    stype = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float)

    def flux(n):
        f = np.asarray(d1[n + "_FLUX"], float) * 10.0 ** CAL[n]
        sn = np.asarray(d1[n + "_FLUX"], float) * np.sqrt(np.clip(np.asarray(d1[n + "_FLUX_IVAR"], float), 0, None))
        return f, sn
    Fha, SNha = flux("HALPHA"); Fhb, SNhb = flux("HBETA")
    Foa, SNoa = flux("OII_3726"); Fob, SNob = flux("OII_3729"); Fo3, SNo3 = flux("OIII_5007")
    base = (zwarn == 0) & (stype == "GALAXY") & np.isfinite(lpm) & (lpm > 6) & (lpm < 13) & (z > 0.05)
    logm = lpm + MASS_ZP
    flow, meta = load_flow()

    # ---------------- TIER 1 ----------------
    t1 = base & (z < 0.49) & (SNha > 3) & (Fha > 0)
    idx = np.where(t1)[0]
    loglha1 = log10_lum(z, Fha)
    mean8, draws8 = sample(flow, meta, logm[idx], loglha1[idx], seed=11)
    inbox = ((logm[idx] >= BOX_M[0]) & (logm[idx] <= BOX_M[1]) &
             (loglha1[idx] >= BOX_L[0]) & (loglha1[idx] <= BOX_L[1]))
    linefluxes = {"Hbeta": (Fhb, SNhb), "OII3726": (Foa, SNoa), "OII3729": (Fob, SNob), "OIII5007": (Fo3, SNo3)}
    obs4 = np.full((len(idx), 4), np.nan); sn4 = np.zeros((len(idx), 4))
    predmean4 = np.zeros((len(idx), 4)); preddraw4 = np.zeros((len(idx), KDRAW, 4))
    sp("===== TIER 1 (0.05<z<0.49) =====  N(Ha S/N>3)=%d  in-box=%d (%.1f%%)"
       % (len(idx), inbox.sum(), 100 * inbox.mean()))
    for c, (name, j) in enumerate(FOUR):
        F, SN = linefluxes[name]
        obs = np.log10(F[idx]) - np.log10(Fha[idx])
        obs4[:, c] = obs; sn4[:, c] = SN[idx]
        predmean4[:, c] = mean8[:, j]; preddraw4[:, :, c] = draws8[:, :, j]
        good = (SN[idx] > 3) & (F[idx] > 0) & np.isfinite(obs) & inbox
        N, b, s, rm, rho = stat(mean8[good, j], obs[good])
        sp("  %-9s [inbox] N=%5d bias=%+.3f scat=%.3f rmse=%.3f rho=%.3f" % (name, N, b, s, rm, rho))

    # ---------------- TIER 2 ----------------
    t2 = base & (z >= 0.49) & (z < 1.00) & (SNhb > 3) & (SNo3 > 3) & (Fhb > 0) & (Fo3 > 0)
    idx2 = np.where(t2)[0]
    loglha2 = log10_lum(z, Fhb) + np.log10(R_DECREMENT)
    inbox2 = ((logm[idx2] >= BOX_M[0]) & (logm[idx2] <= BOX_M[1]) &
              (loglha2[idx2] >= BOX_L[0]) & (loglha2[idx2] <= BOX_L[1]))
    mean8b, draws8b = sample(flow, meta, logm[idx2], loglha2[idx2], seed=22)
    pred_o3hb_mean = mean8b[:, IOIII] - mean8b[:, IHB]
    pred_o3hb_draw = draws8b[:, :, IOIII] - draws8b[:, :, IHB]
    obs_o3hb = np.log10(Fo3[idx2]) - np.log10(Fhb[idx2])
    N, b, s, rm, rho = stat(pred_o3hb_mean[inbox2], obs_o3hb[inbox2])
    sp("===== TIER 2 (0.49<z<1.0) =====  N=%d in-box=%d" % (len(idx2), inbox2.sum()))
    sp("  [OIII]/Hb [inbox] N=%5d bias=%+.3f scat=%.3f rmse=%.3f rho=%.3f" % (N, b, s, rm, rho))

    np.savez_compressed(
        OUT,
        # Tier 1
        t1_z=z[idx], t1_logm=logm[idx], t1_loglha=loglha1[idx], t1_inbox=inbox,
        t1_obs=obs4, t1_sn=sn4, t1_predmean=predmean4, t1_preddraw=preddraw4,
        t1_names=np.array([n for n, _ in FOUR]),
        # Tier 2
        t2_z=z[idx2], t2_inbox=inbox2, t2_obs_o3hb=obs_o3hb,
        t2_pred_o3hb_mean=pred_o3hb_mean, t2_pred_o3hb_draw=pred_o3hb_draw,
    )
    sp("Saved arrays -> " + OUT)
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
