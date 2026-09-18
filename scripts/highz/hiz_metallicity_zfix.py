#!/usr/bin/env python3
"""
Step 2 of the redshift-extrapolation appendix: can a METALLICITY-conditioned flow, fed the
externally-evolved MZR metallicity at z~1, remove the [OII]/[OIII] under-prediction?

Train on DESI-BGS ALT-B (same selection as the fiducial DESI flow), dropping [NII] from the
targets and conditioning on 12+log(O/H) from N2 (=8.90+0.57*log[NII]/Ha, Pettini&Pagel 2004):
  baseline : (logM*, logL_Ha)            -> 7 lines
  +Z       : (logM*, logL_Ha, 12+logOH)  -> 7 lines
Fit the LOCAL MZR from the training N2 metallicities (median in M* bins).
Test on DESI-COSMOS (Tier1 obs L_Ha z<0.49; Tier2 L_Ha from mass-dep Balmer decrement 0.49<z<1),
assigning each galaxy a metallicity from:
  local MZR(M*)                         -> control, should reproduce the baseline behaviour
  evolved MZR(M*,z)=local - 0.11*z      -> Sanders+2021 d(O/H)/dz, the correction
Diagnostic: median residual (pred-obs) of log([OII]/Hb),[OIII]/Hb vs z for baseline vs +Z-evolved.
Caveat: our metallicities are N2/PP04; Sanders+2021 use different calibrations, so the -0.11/dz
differential is applied as an approximate shift on our local MZR scale.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
import jax, jax.numpy as jnp, jax.random as jr
import optax, equinox as eqx
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import scienceplots; plt.style.use(["science", "no-latex"])
except Exception:
    pass
import cmasher as cmr  # noqa

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = Path(BASE + "nebular_emission_model")
TRAIN_FITS = BASE + "DESI_BGS_training_data_ALTB.fits"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
FLUX_SCALE = 1e-17; MASS_ZP = 0.13
SEED = 0; EPOCHS = 200; BATCH = 4096; LR = 3e-4; CLIP = 1.0; NMC = 40
PP04_A, PP04_B = 8.90, 0.57
DZDOH = -0.11              # Sanders+2021 d[12+log(O/H)]/dz at fixed M*
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
LINE_ALIASES = [["HBETA_FLUX", "H_BETA_FLUX"], ["HGAMMA_FLUX", "H_GAMMA_FLUX"], ["NII_6584_FLUX"],
                ["SII_6716_FLUX", "SII_6717_FLUX"], ["SII_6731_FLUX"], ["OII_3726_FLUX"],
                ["OII_3729_FLUX"], ["OIII_5007_FLUX"]]
NII_COL = 2; KEEP = [0, 1, 3, 4, 5, 6, 7]        # 7 targets (drop [NII])
IHB7, IO2A7, IO2B7, IO37 = 0, 4, 5, 6            # within the 7-line array
DATA_C = "#0072B2"; NF_C = "#CC79A7"; EVO_C = "#009E73"; ORA_C = "#E69F00"


def sp(m): print(m, flush=True)


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def decode(a):
    a = np.asarray(a)
    if a.dtype.kind == "S": a = np.char.decode(a)
    return np.char.strip(a.astype(str))


def resolve(df, aliases):
    for c in aliases:
        if c in df.columns:
            return c
    raise KeyError(aliases)


def train_flow(key, Xn, Un, xdim, cond_dim, tag=""):
    flow = block_neural_autoregressive_flow(key=key, base_dist=Normal(jnp.zeros(xdim)), cond_dim=cond_dim)
    opt = optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(LR))
    st = opt.init(eqx.filter(flow, eqx.is_inexact_array))

    @eqx.filter_jit
    def loss_fn(fl, x, u):
        return -jnp.mean(fl.log_prob(x, condition=u))

    @eqx.filter_jit
    def step(fl, s, x, u):
        loss, g = eqx.filter_value_and_grad(loss_fn)(fl, x, u)
        upd, s = opt.update(eqx.filter(g, eqx.is_inexact_array), s, params=eqx.filter(fl, eqx.is_inexact_array))
        return eqx.apply_updates(fl, upd), s, loss
    rng = np.random.default_rng(SEED + 7); n = Xn.shape[0]
    for ep in range(1, EPOCHS + 1):
        order = rng.permutation(n); ls = []
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            flow, st, loss = step(flow, st, jnp.asarray(Xn[idx]), jnp.asarray(Un[idx]))
            ls.append(float(loss))
        if ep % 50 == 0 or ep == 1:
            sp(f"  [{tag}] epoch {ep:3d} loss={np.mean(ls):.4f}")
    return flow


def sample_mean(flow, U, Um, Us, Xm, Xs, xdim, key, n_mc=NMC, batch=40000):
    Un_all = jnp.asarray((U - Um) / Us); out = np.zeros((len(U), xdim))
    for lo in range(0, len(U), batch):
        hi = min(len(U), lo + batch); Un = Un_all[lo:hi]; acc = np.zeros((hi - lo, xdim))
        for _ in range(n_mc):
            key, sk = jr.split(key); keys = jr.split(sk, hi - lo)
            Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un))
            acc += Xn
        out[lo:hi] = acc / n_mc
    return out * Xs + Xm


def build_training():
    t = Table.read(TRAIN_FITS, hdu=1)
    df = t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()
    cols = [resolve(df, a) for a in LINE_ALIASES]
    sp("train flux cols (idx2=[NII]): " + str(cols))
    F = np.column_stack([df[c].to_numpy(float) for c in cols])
    ha = resolve(df, ["HALPHA_FLUX", "H_ALPHA_FLUX"]); fha = df[ha].to_numpy(float)
    logm = df["LOGM_COLOR"].to_numpy(float); z = df["Z"].to_numpy(float)
    good = (np.all(F > 0, axis=1) & np.all(np.isfinite(F), axis=1) & (fha > 0) & np.isfinite(fha)
            & np.isfinite(logm) & np.isfinite(z) & (z > 0))
    X8 = np.log10(F[good]) - np.log10(fha[good])[:, None]
    loglha = log10_lum(z[good], fha[good])
    logOH = PP04_A + PP04_B * X8[:, NII_COL]
    return logm[good], loglha, logOH, X8[:, KEEP]


def build_cosmos():
    h = fits.open(COSMOS); d1, d6 = h[1].data, h[6].data
    z = np.asarray(d1["Z"], float); zw = np.asarray(d1["ZWARN"], float)
    st = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float)

    def cf(nm):
        raw = np.asarray(d1[nm + "_FLUX"], float); iv = np.asarray(d1[nm + "_FLUX_IVAR"], float)
        return raw * 10 ** CAL[nm], raw * np.sqrt(np.clip(iv, 0, None))
    Fha, Sha = cf("HALPHA"); Fhb, Shb = cf("HBETA")
    Foa, Soa = cf("OII_3726"); Fob, Sob = cf("OII_3729"); Fo3, So3 = cf("OIII_5007")
    base = (zw == 0) & (st == "GALAXY") & (z > 0.05) & np.isfinite(lpm) & (lpm > 6) & (lpm < 13)
    logm = lpm + MASS_ZP
    # mass-dependent Balmer decrement from z<0.49
    calm = base & (z < 0.49) & (Sha > 3) & (Shb > 3) & (Fha > 0) & (Fhb > 0)
    dec = (log10_lum(z, Fha) - log10_lum(z, Fhb))[calm]; mm = logm[calm]
    mb = np.percentile(mm, np.linspace(0, 100, 9)); mc = 0.5 * (mb[:-1] + mb[1:])
    Rm = np.array([np.median(dec[(mm >= a) & (mm < b)]) for a, b in zip(mb[:-1], mb[1:])])
    Rof = lambda x: np.interp(x, mc, Rm, left=Rm[0], right=Rm[-1])
    out = {}
    # Tier 1
    t1 = base & (z >= 0.05) & (z < 0.49) & (Sha > 3) & (Shb > 3) & (Foa > 0) & (Fob > 0) & (Fo3 > 0) & (Fha > 0) & (Fhb > 0)
    i1 = np.where(t1)[0]
    out["t1"] = dict(z=z[i1], logm=logm[i1], loglha=log10_lum(z[i1], Fha[i1]),
                     o2a=np.log10(Foa[i1]) - np.log10(Fhb[i1]), o2b=np.log10(Fob[i1]) - np.log10(Fhb[i1]),
                     o3=np.log10(Fo3[i1]) - np.log10(Fhb[i1]))
    # Tier 2
    t2 = base & (z >= 0.49) & (z < 1.00) & (Shb > 3) & (Soa > 3) & (Sob > 3) & (So3 > 3) & (Fhb > 0) & (Foa > 0) & (Fob > 0) & (Fo3 > 0)
    i2 = np.where(t2)[0]
    lhb = log10_lum(z[i2], Fhb[i2]); lha2 = lhb + Rof(logm[i2])
    out["t2"] = dict(z=z[i2], logm=logm[i2], loglha=lha2,
                     o2a=np.log10(Foa[i2]) - np.log10(Fhb[i2]), o2b=np.log10(Fob[i2]) - np.log10(Fhb[i2]),
                     o3=np.log10(Fo3[i2]) - np.log10(Fhb[i2]))
    return out


def main():
    logm_tr, loglha_tr, logOH_tr, X7_tr = build_training()
    sp("train N=%d  12+log(O/H) p16/50/84 = %.2f/%.2f/%.2f" %
       (len(logm_tr), *np.percentile(logOH_tr, [16, 50, 84])))
    # local MZR (median N2 metallicity vs M*)
    mb = np.percentile(logm_tr, np.linspace(0, 100, 13)); mc = 0.5 * (mb[:-1] + mb[1:])
    Zmed = np.array([np.median(logOH_tr[(logm_tr >= a) & (logm_tr < b)]) for a, b in zip(mb[:-1], mb[1:])])
    MZR = lambda m: np.interp(m, mc, Zmed, left=Zmed[0], right=Zmed[-1])

    # FMR-consistent metallicity from (M*, L_Ha): quadratic least-squares (matches what the flow assumes)
    def _design(m, l):
        m = np.asarray(m, float); l = np.asarray(l, float)
        return np.column_stack([np.ones_like(m), m, l, m * m, l * l, m * l])
    _coef, *_ = np.linalg.lstsq(_design(logm_tr, loglha_tr), logOH_tr, rcond=None)
    ZPRED = lambda m, l: _design(m, l) @ _coef
    _rz = ZPRED(logm_tr, loglha_tr) - logOH_tr
    sp("Z-predictor (quadratic, M*,L_Ha): RMSE=%.3f dex" % float(np.sqrt(np.mean(_rz ** 2))))

    U2 = np.column_stack([logm_tr, loglha_tr])
    U3 = np.column_stack([logm_tr, loglha_tr, logOH_tr])
    U2m, U2s = U2.mean(0), U2.std(0); U2s = np.where(U2s == 0, 1, U2s)
    U3m, U3s = U3.mean(0), U3.std(0); U3s = np.where(U3s == 0, 1, U3s)
    X7m, X7s = X7_tr.mean(0), X7_tr.std(0); X7s = np.where(X7s == 0, 1, X7s)
    Xn = ((X7_tr - X7m) / X7s).astype(np.float32)
    Un2 = ((U2 - U2m) / U2s).astype(np.float32)
    Un3 = ((U3 - U3m) / U3s).astype(np.float32)
    k = jr.key(SEED); k1, k2, ks = jr.split(k, 3)
    sp("Training DESI baseline (2D->7)...")
    f_base = train_flow(k1, Xn, Un2, 7, 2, tag="base")
    sp("Training DESI +Z (3D->7)...")
    f_Z = train_flow(k2, Xn, Un3, 7, 3, tag="+Z")

    cos = build_cosmos()
    sp("COSMOS Tier1 N=%d  Tier2 N=%d" % (len(cos["t1"]["z"]), len(cos["t2"]["z"])))

    # assemble predictions across both tiers, using ratios relative to Hbeta
    def predict(d, key):
        m, lha, zz = d["logm"], d["loglha"], d["z"]
        inbox = (m >= BOX_M[0]) & (m <= BOX_M[1]) & (lha >= BOX_L[0]) & (lha <= BOX_L[1])
        Zloc = ZPRED(m, lha); Zevo = ZPRED(m, lha) + DZDOH * zz
        U2t = np.column_stack([m, lha])
        Pb = sample_mean(f_base, U2t, U2m, U2s, X7m, X7s, 7, key)
        Ploc = sample_mean(f_Z, np.column_stack([m, lha, Zloc]), U3m, U3s, X7m, X7s, 7, key)
        Pevo = sample_mean(f_Z, np.column_stack([m, lha, Zevo]), U3m, U3s, X7m, X7s, 7, key)
        # ratios relative to Hbeta
        def relhb(P, idx):
            return P[:, idx] - P[:, IHB7]
        obs = {"o2a": d["o2a"], "o2b": d["o2b"], "o3": d["o3"]}
        pred = {}
        for nm, idx in [("o2a", IO2A7), ("o2b", IO2B7), ("o3", IO37)]:
            pred[nm] = dict(base=relhb(Pb, idx), loc=relhb(Ploc, idx), evo=relhb(Pevo, idx))
        return zz, inbox, obs, pred

    z1, ib1, obs1, pr1 = predict(cos["t1"], ks)
    z2, ib2, obs2, pr2 = predict(cos["t2"], ks)
    Z = np.concatenate([z1[ib1], z2[ib2]])
    OBS = {nm: np.concatenate([obs1[nm][ib1], obs2[nm][ib2]]) for nm in ("o2a", "o2b", "o3")}
    PR = {nm: {c: np.concatenate([pr1[nm][c][ib1], pr2[nm][c][ib2]]) for c in ("base", "loc", "evo")}
          for nm in ("o2a", "o2b", "o3")}

    # residual (pred-obs) in z bins
    zb = np.linspace(0.05, 1.0, 11); zc = 0.5 * (zb[:-1] + zb[1:])
    LAB = {"o2a": r"[OII]3726/H$\beta$", "o2b": r"[OII]3729/H$\beta$", "o3": r"[OIII]5007/H$\beta$"}
    sp("\n=== median residual (pred-obs), low-z(<0.35) / high-z(>0.6) ===")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for c, nm in enumerate(("o2a", "o2b", "o3")):
        for case, col, lab in [("base", DATA_C, "baseline (M*, L_Ha)"),
                               ("loc", "0.5", "+Z local FMR (control)"),
                               ("evo", EVO_C, "+Z evolved-MZR (Sanders+21)")]:
            med = []
            for a, b in zip(zb[:-1], zb[1:]):
                s = (Z >= a) & (Z < b)
                med.append(np.median(PR[nm][case][s] - OBS[nm][s]) if s.sum() > 15 else np.nan)
            ax[c].plot(zc, med, "-o", color=col, lw=2, ms=4, label=lab)
        lo = (Z < 0.35); hi = (Z > 0.6)
        sp("  %-16s base %+.3f/%+.3f  loc %+.3f/%+.3f  evo %+.3f/%+.3f" % (
            nm,
            np.median(PR[nm]["base"][lo] - OBS[nm][lo]), np.median(PR[nm]["base"][hi] - OBS[nm][hi]),
            np.median(PR[nm]["loc"][lo] - OBS[nm][lo]), np.median(PR[nm]["loc"][hi] - OBS[nm][hi]),
            np.median(PR[nm]["evo"][lo] - OBS[nm][lo]), np.median(PR[nm]["evo"][hi] - OBS[nm][hi])))
        ax[c].axhline(0, color="k", lw=0.8, ls=":")
        ax[c].set_xlabel("redshift"); ax[c].set_title(LAB[nm])
        if c == 0:
            ax[c].set_ylabel(r"median residual  pred$-$obs (dex)"); ax[c].legend(frameon=False, fontsize=10)
    fig.suptitle(r"Metallicity conditioning + evolved MZR: does it de-bias [OII]/[OIII] to $z\sim1$? (DESI-COSMOS)", y=1.0)
    fig.tight_layout()
    out = REPO / "figs_ALTB" / "hiz_metallicity_zfix.png"
    fig.savefig(out, bbox_inches="tight", dpi=160); sp("Saved: " + str(out))
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
