#!/usr/bin/env python3
"""
Does the [OII]/Hb under-prediction survive controlling for SFR (and for BOTH M* and SFR)?
Bins the predicted-minus-observed [OII]3727/Hb residual vs redshift by an INDEPENDENT SFR
(COSMOS2020 SED SFR, lp_SFR) -- not the flow's L_Ha conditioning -- to test the Kaasinen-style
"is it just SFR?" concern. Panels:
  A: residual vs z in stellar-mass bins            (fixed M*)
  B: residual vs z in SED-SFR bins                 (fixed SFR)  <- requested
  C: residual vs z in SED-SFR bins at FIXED M*     (9.5<logM*<10.3)  (control for BOTH)
Conditioning: L_Ha = L_Hb + R(M*) (mass-dep decrement); logM* = lp_mass+0.13.
"""
import pickle, numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from functools import partial as _pf
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import scienceplots; plt.style.use(["science", "no-latex"])
except Exception: pass
plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 11, "axes.titlesize": 14})


def sp(m): print(m, flush=True)


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
FLUX_SCALE = 1e-17; MASS_ZP = 0.13
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
IHB, IOII_A, IOII_B = 0, 5, 6
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
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


def sample_mean(flow, meta, logm, loglha, seed, n_mc=40, batch=40000):
    U = np.column_stack([logm, loglha]).astype(np.float32)
    Un_all = jnp.asarray((U - meta["U_mean"]) / meta["U_std"])
    Xm, Xs = np.asarray(meta["X_mean"]), np.asarray(meta["X_std"]); n = len(U)
    acc = np.zeros((n, 8)); key = jr.key(seed + 7)
    for lo in range(0, n, batch):
        hi = min(n, lo + batch); Un = Un_all[lo:hi]; a = np.zeros((hi - lo, 8))
        for _ in range(n_mc):
            key, sk = jr.split(key); keys = jr.split(sk, hi - lo)
            Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un))
            a += Xn * Xs + Xm
        acc[lo:hi] = a / n_mc
    return acc


def decode(a):
    a = np.asarray(a)
    if a.dtype.kind == "S": a = np.char.decode(a)
    return np.char.strip(a.astype(str))


def main():
    h = fits.open(COSMOS); d1, d6 = h[1].data, h[6].data
    z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
    stype = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float); lpsfr = np.asarray(d6["lp_SFR_med"], float)

    def cf(name):
        raw = np.asarray(d1[name + "_FLUX"], float); iv = np.asarray(d1[name + "_FLUX_IVAR"], float)
        return raw * 10 ** CAL[name], raw * np.sqrt(np.clip(iv, 0, None))
    Fhb, Shb = cf("HBETA"); Foa, Soa = cf("OII_3726"); Fob, Sob = cf("OII_3729")
    lpsfr_log = lpsfr if np.nanmedian(np.abs(lpsfr[np.isfinite(lpsfr)])) < 5 else np.log10(np.clip(lpsfr, 1e-4, None))
    base = ((zwarn == 0) & (stype == "GALAXY") & (z > 0.05) & (z < 1.0) & np.isfinite(lpm) & (lpm > 6) & (lpm < 13)
            & np.isfinite(lpsfr_log) & (lpsfr_log > -2.5) & (lpsfr_log < 2.5)
            & (Shb > 3) & (Soa > 3) & (Sob > 3) & (Fhb > 0) & (Foa > 0) & (Fob > 0))
    logm = lpm + MASS_ZP

    # mass-dependent decrement R(M*) from z<0.49 (Ha & Hb detected)
    Fha, Sha = cf("HALPHA")  # uses HALPHA CAL? add
    idx = np.where(base)[0]
    zt = z[idx]; mt = logm[idx]; sfrt = lpsfr_log[idx]
    LHb = log10_lum(zt, Fhb[idx]); LHa = LHb + decrement_R(z, Fha, Sha, Fhb, Shb, base, logm)(mt)
    inbox = (mt >= BOX_M[0]) & (mt <= BOX_M[1]) & (LHa >= BOX_L[0]) & (LHa <= BOX_L[1])
    idx, zt, mt, sfrt, LHb, LHa = idx[inbox], zt[inbox], mt[inbox], sfrt[inbox], LHb[inbox], LHa[inbox]
    sp("sample 0.05<z<1, Hb+[OII] detected, valid SED-SFR, in-box: N=%d" % len(idx))
    flow, meta = load_flow()
    mean8 = sample_mean(flow, meta, mt, LHa, seed=17)
    pred = np.log10(10 ** mean8[:, IOII_A] + 10 ** mean8[:, IOII_B]) - mean8[:, IHB]
    obs = np.log10(Foa[idx] + Fob[idx]) - np.log10(Fhb[idx])
    res = pred - obs

    zb = np.linspace(0.05, 1.0, 9); zc = 0.5 * (zb[:-1] + zb[1:])

    def track(ax, binvar, edges, cols, labels, extra=None, title=""):
        for (lo, hi), col, lab in zip(edges, cols, labels):
            sel = (binvar >= lo) & (binvar < hi)
            if extra is not None: sel = sel & extra
            my = [np.median(res[sel & (zt >= a) & (zt < b)]) if (sel & (zt >= a) & (zt < b)).sum() > 15 else np.nan
                  for a, b in zip(zb[:-1], zb[1:])]
            ax.plot(zc, my, "-o", color=col, ms=4, lw=2, label=lab)
        ax.axhline(0, color="k", lw=0.7, ls=":"); ax.set_xlabel("redshift"); ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    mcols = ["#009E73", "#D55E00", "#7030A0"]
    track(ax[0], mt, [(8.8, 9.7), (9.7, 10.3), (10.3, 11.05)], mcols,
          [r"$8.8<\log M_\star<9.7$", r"$9.7<\log M_\star<10.3$", r"$10.3<\log M_\star<11.1$"], title="fixed stellar mass")
    ax[0].set_ylabel(r"median residual [OII]3727/H$\beta$ (pred$-$obs)")
    # SFR tertiles
    sq = np.percentile(sfrt, [33, 67]); sedges = [(-2.5, sq[0]), (sq[0], sq[1]), (sq[1], 2.5)]
    slabs = [r"low SFR", r"mid SFR", r"high SFR"]
    track(ax[1], sfrt, sedges, mcols, slabs, title="fixed SED SFR (independent)")
    # fixed M* slice, SFR sub-bins
    msl = (mt >= 9.5) & (mt < 10.3)
    sq2 = np.percentile(sfrt[msl], [33, 67]); sedges2 = [(-2.5, sq2[0]), (sq2[0], sq2[1]), (sq2[1], 2.5)]
    track(ax[2], sfrt, sedges2, mcols, slabs, extra=msl, title=r"fixed $M_\star$ (9.5-10.3) $\times$ SFR")
    fig.suptitle(r"[OII]/H$\beta$ under-prediction vs $z$, controlling for SFR (and both M$_\star$+SFR)", y=1.0)
    fig.tight_layout()
    out = REPO + "figs_ALTB/hiz_oii_sfr_control.png"; fig.savefig(out, bbox_inches="tight", dpi=150); sp("Saved: " + out)
    # print z-slopes
    for lab, sel in [("all", np.ones(len(res), bool)),
                     ("lowSFR", (sfrt < sq[0])), ("highSFR", (sfrt >= sq[1])),
                     ("fixedM*,lowSFR", msl & (sfrt < sq2[0])), ("fixedM*,highSFR", msl & (sfrt >= sq2[1]))]:
        my = np.array([np.median(res[sel & (zt >= a) & (zt < b)]) if (sel & (zt >= a) & (zt < b)).sum() > 15 else np.nan
                       for a, b in zip(zb[:-1], zb[1:])])
        ok = np.isfinite(my); slope = np.polyfit(zc[ok], my[ok], 1)[0] if ok.sum() > 2 else np.nan
        sp("  %-18s residual z-slope = %+.3f dex/unit-z ; median res=%+.3f" % (lab, slope, np.median(res[sel])))
    sp("=== DONE ===")


def decrement_R(z, Fha, Sha, Fhb, Shb, base, logm):
    calm = base & (z < 0.49) & (Sha > 3) & (Shb > 3) & (Fha > 0) & (Fhb > 0)
    dec = (log10_lum(z, Fha) - log10_lum(z, Fhb))[calm]; mm = logm[calm]
    mb = np.percentile(mm, np.linspace(0, 100, 9)); mc = 0.5 * (mb[:-1] + mb[1:])
    Rm = np.array([np.median(dec[(mm >= a) & (mm < b)]) for a, b in zip(mb[:-1], mb[1:])])
    return lambda m: np.interp(m, mc, Rm, left=Rm[0], right=Rm[-1])


if __name__ == "__main__":
    main()
