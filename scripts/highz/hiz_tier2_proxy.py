#!/usr/bin/env python3
"""
Tier-2 (0.49<z<1.0) [OIII]/Hb validation with the OPTIMAL L_Ha proxy.
Replaces the fixed decrement R=3.9 with:
  (A) mass-dependent decrement R(M*)  [primary]  -- calibrated on z<0.49 all-line sample
  (B) flow-self-consistent inference   [cross-check] -- L_Ha = L_Hb - <flow log(Hb/Ha)|M*,L_Ha>, fixed point
Compares A, B, and the old fixed R=3.9. Regenerates the Tier-2 figure with proxy A.
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import scienceplots  # noqa
    plt.style.use(["science", "no-latex"])
except Exception:
    pass
plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 12, "axes.titlesize": 14, "figure.dpi": 130})


def sp(m): print(m, flush=True)


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
FLUX_SCALE = 1e-17; MASS_ZP = 0.13
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OIII_5007": 0.0502}
IHB, IOIII = 0, 7
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
DATA_C = "#0072B2"; NF_C = "#CC79A7"
_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def load_flow():
    meta = pickle.load(open(REPO + "models/nf_desi_ALTB_meta.pkl", "rb"))
    xd = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xd)), cond_dim=2, inverter=_INV)
    return eqx.tree_deserialise_leaves(REPO + "models/nf_desi_ALTB.eqx", tmpl), meta


def sample(flow, meta, logm, loglha, seed, n_mc, kdraw=0, batch=40000):
    U = np.column_stack([logm, loglha]).astype(np.float32)
    Un_all = jnp.asarray((U - meta["U_mean"]) / meta["U_std"])
    Xm, Xs = np.asarray(meta["X_mean"]), np.asarray(meta["X_std"]); n = len(U)
    mean = np.zeros((n, 8)); draws = np.zeros((n, max(kdraw, 1), 8)); key = jr.key(seed + 7)
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
    return dict(N=int(g.sum()), bias=float(np.median(rr)),
                scat=float(0.5 * (np.percentile(rr, 84) - np.percentile(rr, 16))),
                rmse=float(np.sqrt(np.mean(rr ** 2))), rho=float(spearmanr(pred[g], obs[g]).correlation))


def main():
    h = fits.open(COSMOS); d1, d6 = h[1].data, h[6].data
    z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
    stype = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float)

    def cf(name):
        raw = np.asarray(d1[name + "_FLUX"], float); iv = np.asarray(d1[name + "_FLUX_IVAR"], float)
        return raw * 10 ** CAL[name], raw * np.sqrt(np.clip(iv, 0, None))
    Fha, Sha = cf("HALPHA"); Fhb, Shb = cf("HBETA"); Fo3, So3 = cf("OIII_5007")
    base = (zwarn == 0) & (stype == "GALAXY") & (z > 0.05) & np.isfinite(lpm) & (lpm > 6) & (lpm < 13)
    logm = lpm + MASS_ZP

    # ---- calibrate mass-dependent decrement R(M*) from z<0.49 all-Balmer sample ----
    calm = base & (z < 0.49) & (Sha > 3) & (Shb > 3) & (Fha > 0) & (Fhb > 0)
    dec = (log10_lum(z, Fha) - log10_lum(z, Fhb))[calm]      # log10(Ha/Hb)
    mm = logm[calm]
    mbins = np.percentile(mm, np.linspace(0, 100, 9)); mcen = 0.5 * (mbins[:-1] + mbins[1:])
    Rm = np.array([np.median(dec[(mm >= a) & (mm < b)]) for a, b in zip(mbins[:-1], mbins[1:])])
    Rmed = float(np.median(dec))
    sp("R(M*) calibration (z<0.49, N=%d): median log(Ha/Hb)=%.3f (R=%.2f)" % (calm.sum(), Rmed, 10 ** Rmed))
    sp("  R(M*) by mass: " + "  ".join("M%.1f:%.2f" % (c, r) for c, r in zip(mcen, Rm)))
    R_of_M = lambda m: np.interp(m, mcen, Rm, left=Rm[0], right=Rm[-1])

    # ---- Tier-2 sample ----
    t2 = base & (z >= 0.49) & (z < 1.00) & (Shb > 3) & (So3 > 3) & (Fhb > 0) & (Fo3 > 0)
    idx = np.where(t2)[0]
    zt = z[idx]; mt = logm[idx]; LHb = log10_lum(zt, Fhb[idx])
    obs_o3hb = np.log10(Fo3[idx]) - np.log10(Fhb[idx])
    sp("Tier-2 N=%d" % len(idx))
    flow, meta = load_flow()

    # ---- three L_Ha proxies ----
    LHa_fixed = LHb + np.log10(3.90)
    LHa_massdec = LHb + R_of_M(mt)
    # flow self-consistent fixed point
    L = LHb + Rmed
    for it in range(4):
        rm, _ = sample(flow, meta, mt, L, seed=100 + it, n_mc=12)
        L = LHb - rm[:, IHB]        # log Ha = log Hb - log(Hb/Ha)
    LHa_self = L
    sp("median L_Ha: fixed=%.3f massdec=%.3f self=%.3f" % (np.median(LHa_fixed), np.median(LHa_massdec), np.median(LHa_self)))

    def evaluate(LHa, seed, kdraw=0):
        mean, draws = sample(flow, meta, mt, LHa, seed=seed, n_mc=40, kdraw=kdraw)
        pred = mean[:, IOIII] - mean[:, IHB]
        inbox = (mt >= BOX_M[0]) & (mt <= BOX_M[1]) & (LHa >= BOX_L[0]) & (LHa <= BOX_L[1])
        st = stat(pred[inbox], obs_o3hb[inbox])
        pdr = (draws[:, :, IOIII] - draws[:, :, IHB]) if kdraw else None
        return pred, inbox, st, pdr

    sp("\n[OIII]/Hb Tier-2 (in-box) by proxy:")
    for name, LHa, sd in [("fixed R=3.9", LHa_fixed, 21), ("mass-dep R(M*)", LHa_massdec, 22), ("flow self-consistent", LHa_self, 23)]:
        _, ib, st, _ = evaluate(LHa, sd)
        sp("  %-22s N=%5d bias=%+.3f scat=%.3f rmse=%.3f rho=%.3f" % (name, st["N"], st["bias"], st["scat"], st["rmse"], st["rho"]))

    # ---- figure with mass-dep proxy ----
    pred_md, ib_md, st_md, pdr_md = evaluate(LHa_massdec, 55, kdraw=8)
    fig = plt.figure(figsize=(15.5, 4.4)); gs = fig.add_gridspec(1, 3, wspace=0.32)
    # (1) R(M*)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(mcen, Rm, "-o", color="#009E73")
    ax.axhline(np.log10(3.9), ls=":", color="0.5", label="fixed R=3.9")
    ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$"); ax.set_ylabel(r"decrement $\log_{10}(\mathrm{H}\alpha/\mathrm{H}\beta)$")
    ax.set_title(r"mass-dep. decrement ($z<0.49$)"); ax.legend(frameon=False)
    # (2) distribution
    ax = fig.add_subplot(gs[0, 1]); mfin = ib_md & np.isfinite(obs_o3hb)
    o = obs_o3hb[mfin]; p = pdr_md[mfin].ravel()
    lo, hi = np.percentile(np.concatenate([o, p]), [0.5, 99.5]); bins = np.linspace(lo, hi, 36)
    ax.hist(o, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=DATA_C, label="DESI-COSMOS")
    ax.hist(p, bins=bins, density=True, histtype="step", lw=2.4, color=NF_C, label="NF (mass-dep. proxy)")
    ax.set_xlabel(r"$\log_{10}(\mathrm{[OIII]}5007/\mathrm{H}\beta)$"); ax.set_yticks([])
    ax.set_title(r"$0.49<z<1.0$ (in-box)"); ax.legend(frameon=False, fontsize=11)
    ax.text(0.03, 0.97, "bias $%+.3f$\n$\\rho=%.2f$" % (st_md["bias"], st_md["rho"]), transform=ax.transAxes,
            va="top", fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    # (3) evolution
    ax = fig.add_subplot(gs[0, 2]); zz = zt[mfin]; oo = obs_o3hb[mfin]; pp = pred_md[mfin]
    zb = np.linspace(0.49, 1.0, 8); zc = 0.5 * (zb[:-1] + zb[1:])
    om, ol, oh, PM, PL, PH = ([] for _ in range(6))
    for a, b in zip(zb[:-1], zb[1:]):
        s = (zz >= a) & (zz < b)
        if s.sum() < 15:
            for L2 in (om, ol, oh, PM, PL, PH): L2.append(np.nan)
            continue
        om.append(np.median(oo[s])); ol.append(np.percentile(oo[s], 16)); oh.append(np.percentile(oo[s], 84))
        PM.append(np.median(pp[s])); PL.append(np.percentile(pp[s], 16)); PH.append(np.percentile(pp[s], 84))
    om, ol, oh, PM, PL, PH = map(np.array, (om, ol, oh, PM, PL, PH))
    ax.fill_between(zc, ol, oh, color=DATA_C, alpha=0.2); ax.plot(zc, om, "-o", color=DATA_C, lw=2, label="DESI-COSMOS")
    ax.fill_between(zc, PL, PH, color=NF_C, alpha=0.18); ax.plot(zc, PM, "--s", color=NF_C, lw=2, label="NF prediction")
    ax.set_xlabel("redshift"); ax.set_ylabel(r"$\log_{10}(\mathrm{[OIII]}/\mathrm{H}\beta)$")
    ax.set_title(r"median evolution to $z\sim1$"); ax.legend(frameon=False)
    fig.suptitle(r"Tier 2 with mass-dependent decrement proxy (real DESI-COSMOS, no lightcone)", fontsize=14, y=1.02)
    fig.tight_layout()
    out = REPO + "figs_ALTB/hiz_tier2_massdec.png"; fig.savefig(out, bbox_inches="tight", dpi=150); sp("Saved: " + out)
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
