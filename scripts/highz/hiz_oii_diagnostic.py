#!/usr/bin/env python3
"""
Characterize the ONE discrepancy: the [OII] under-prediction, as a function of redshift (0.05<z<1.0).
Hb is in the DESI window across the whole range, so we use [OII]/Hb as the consistent quantity
throughout (Ha is not). Conditioning L_Ha = L_Hb + R(M*) (mass-dep decrement) uniformly, so the
only thing varying with z is redshift itself (ratios are insensitive to the L_Ha source anyway).

Panels:
  A: median residual (predicted - observed) vs z for [OII]3726/Hb, [OII]3729/Hb, total [OII]3727/Hb,
     with [OIII]5007/Hb as a CONTROL (should stay ~0).
  B: observed vs predicted median [OII]3727/Hb vs z (the divergence).
  C: [OII]3727/Hb residual vs z split by stellar mass -> is the growth genuine z-evolution (at fixed M*)?
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
    import scienceplots  # noqa
    plt.style.use(["science", "no-latex"])
except Exception:
    pass
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "legend.fontsize": 12, "axes.titlesize": 15, "figure.dpi": 130})


def sp(m): print(m, flush=True)


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
FLUX_SCALE = 1e-17; MASS_ZP = 0.13
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
IHB, IOII_A, IOII_B, IOIII = 0, 5, 6, 7
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
    if a.dtype.kind == "S":
        a = np.char.decode(a)
    return np.char.strip(a.astype(str))


def med_err(y, nboot=300, rng=None):
    if len(y) < 15:
        return np.nan, np.nan
    bs = [np.median(rng.choice(y, len(y), replace=True)) for _ in range(nboot)]
    return np.median(y), np.std(bs)


def main():
    rng = np.random.default_rng(3)
    h = fits.open(COSMOS); d1, d6 = h[1].data, h[6].data
    z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
    stype = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float)

    def cf(name):
        raw = np.asarray(d1[name + "_FLUX"], float); iv = np.asarray(d1[name + "_FLUX_IVAR"], float)
        return raw * 10 ** CAL[name], raw * np.sqrt(np.clip(iv, 0, None))
    Fha, Sha = cf("HALPHA"); Fhb, Shb = cf("HBETA")
    Foa, Soa = cf("OII_3726"); Fob, Sob = cf("OII_3729"); Fo3, So3 = cf("OIII_5007")
    base = (zwarn == 0) & (stype == "GALAXY") & (z > 0.05) & np.isfinite(lpm) & (lpm > 6) & (lpm < 13)
    logm = lpm + MASS_ZP

    # mass-dependent decrement R(M*) from z<0.49
    calm = base & (z < 0.49) & (Sha > 3) & (Shb > 3) & (Fha > 0) & (Fhb > 0)
    dec = (log10_lum(z, Fha) - log10_lum(z, Fhb))[calm]; mm = logm[calm]
    mb = np.percentile(mm, np.linspace(0, 100, 9)); mc = 0.5 * (mb[:-1] + mb[1:])
    Rm = np.array([np.median(dec[(mm >= a) & (mm < b)]) for a, b in zip(mb[:-1], mb[1:])])
    R_of_M = lambda m: np.interp(m, mc, Rm, left=Rm[0], right=Rm[-1])

    # sample across 0.05<z<1.0 with Hb + [OII] doublet detected
    sel = (base & (z >= 0.05) & (z < 1.00) & (Shb > 3) & (Soa > 3) & (Sob > 3)
           & (Fhb > 0) & (Foa > 0) & (Fob > 0))
    idx = np.where(sel)[0]
    zt = z[idx]; mt = logm[idx]; LHb = log10_lum(zt, Fhb[idx]); LHa = LHb + R_of_M(mt)
    inbox = (mt >= BOX_M[0]) & (mt <= BOX_M[1]) & (LHa >= BOX_L[0]) & (LHa <= BOX_L[1])
    idx, zt, mt, LHa = idx[inbox], zt[inbox], mt[inbox], LHa[inbox]
    sp("sample 0.05<z<1.0, Hb+[OII] detected, in-box: N=%d  (z med=%.2f)" % (len(idx), np.median(zt)))
    flow, meta = load_flow()
    mean8 = sample_mean(flow, meta, mt, LHa, seed=11)

    # observed & predicted ratios rel Hb
    o_a = np.log10(Foa[idx]) - np.log10(Fhb[idx]); p_a = mean8[:, IOII_A] - mean8[:, IHB]
    o_b = np.log10(Fob[idx]) - np.log10(Fhb[idx]); p_b = mean8[:, IOII_B] - mean8[:, IHB]
    o_t = np.log10(Foa[idx] + Fob[idx]) - np.log10(Fhb[idx])
    p_t = np.log10(10 ** mean8[:, IOII_A] + 10 ** mean8[:, IOII_B]) - mean8[:, IHB]
    res = {"OII3726": p_a - o_a, "OII3729": p_b - o_b, "OII3727": p_t - o_t}
    # [OIII] control (needs [OIII] detected & in window z<0.96)
    o3g = (So3[idx] > 3) & (Fo3[idx] > 0) & (zt < 0.96)
    res_o3 = (mean8[:, IOIII] - mean8[:, IHB]) - (np.log10(Fo3[idx]) - np.log10(Fhb[idx]))

    # ---- Panel A: residual vs z ----
    zb = np.linspace(0.05, 1.0, 11); zc = 0.5 * (zb[:-1] + zb[1:])
    fig = plt.figure(figsize=(16, 4.8)); gs = fig.add_gridspec(1, 3, wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    cols = {"OII3727": "#CC79A7", "OII3726": "#E69F00", "OII3729": "#56B4E9"}
    labs = {"OII3727": r"[OII]3727 (total)", "OII3726": r"[OII]3726", "OII3729": r"[OII]3729"}
    sp("\nz-bin   N    res[OII3727]  res[OIII](ctrl)")
    for key in ["OII3727", "OII3726", "OII3729"]:
        my, ey = [], []
        for a, b in zip(zb[:-1], zb[1:]):
            s = (zt >= a) & (zt < b)
            m, e = med_err(res[key][s], rng=rng); my.append(m); ey.append(e)
        axA.errorbar(zc, my, yerr=ey, fmt="-o", color=cols[key], lw=2, ms=5, capsize=2, label=labs[key])
    # control [OIII]
    myo, eyo = [], []
    for a, b in zip(zb[:-1], zb[1:]):
        s = (zt >= a) & (zt < b) & o3g
        m, e = med_err(res_o3[s], rng=rng); myo.append(m); eyo.append(e)
    axA.errorbar(zc, myo, yerr=eyo, fmt="--s", color="0.4", lw=1.8, ms=4, capsize=2, label=r"[OIII]5007 (control)")
    axA.axhline(0, color="k", lw=0.8, ls=":")
    axA.set_xlabel("redshift"); axA.set_ylabel(r"median residual  pred$-$obs (dex)")
    axA.set_title(r"[OII]/H$\beta$ under-prediction vs $z$"); axA.legend(frameon=False, fontsize=11)
    for a, b in zip(zb[:-1], zb[1:]):
        s = (zt >= a) & (zt < b)
        sc = s & o3g
        sp("[%.2f,%.2f] %4d  %+.3f      %+.3f" % (a, b, s.sum(), np.median(res["OII3727"][s]),
           np.median(res_o3[sc]) if sc.sum() > 10 else np.nan))

    # ---- Panel B: observed vs predicted median [OII]3727/Hb vs z ----
    axB = fig.add_subplot(gs[0, 1])
    om, pm = [], []
    for a, b in zip(zb[:-1], zb[1:]):
        s = (zt >= a) & (zt < b)
        om.append(np.median(o_t[s]) if s.sum() > 10 else np.nan)
        pm.append(np.median(p_t[s]) if s.sum() > 10 else np.nan)
    axB.plot(zc, om, "-o", color="#0072B2", lw=2, label="observed")
    axB.plot(zc, pm, "--s", color="#CC79A7", lw=2, label="NF predicted")
    axB.set_xlabel("redshift"); axB.set_ylabel(r"median $\log_{10}(\mathrm{[OII]}3727/\mathrm{H}\beta)$")
    axB.set_title(r"observed vs predicted [OII]/H$\beta$"); axB.legend(frameon=False)

    # ---- Panel C: residual vs z in mass bins ----
    axC = fig.add_subplot(gs[0, 2])
    mbins3 = [(8.8, 9.7), (9.7, 10.3), (10.3, 11.05)]
    mcols = ["#009E73", "#D55E00", "#7030A0"]
    for (ma, mbb), col in zip(mbins3, mcols):
        msk = (mt >= ma) & (mt < mbb)
        my = []
        for a, b in zip(zb[:-1], zb[1:]):
            s = msk & (zt >= a) & (zt < b)
            my.append(np.median(res["OII3727"][s]) if s.sum() > 15 else np.nan)
        axC.plot(zc, my, "-o", color=col, lw=2, ms=4, label=r"$%.1f<\log M_\star<%.1f$" % (ma, mbb))
    axC.axhline(0, color="k", lw=0.8, ls=":")
    axC.set_xlabel("redshift"); axC.set_ylabel(r"median residual [OII]3727/H$\beta$ (dex)")
    axC.set_title(r"at fixed stellar mass"); axC.legend(frameon=False, fontsize=11)
    fig.suptitle(r"Characterizing the [OII] under-prediction (DESI-trained NF vs DESI-COSMOS, in-box)", fontsize=15, y=1.02)
    fig.tight_layout()
    out = REPO + "figs_ALTB/hiz_oii_diagnostic.png"; fig.savefig(out, bbox_inches="tight", dpi=150); sp("Saved: " + out)

    # slope of residual vs z (total [OII])
    good = np.isfinite(res["OII3727"])
    sl = np.polyfit(zt[good], res["OII3727"][good], 1)
    sp("\n[OII]3727/Hb residual vs z: slope=%.3f dex per unit z; at z=0.1 -> %+.3f, z=0.9 -> %+.3f"
       % (sl[0], sl[0] * 0.1 + sl[1], sl[0] * 0.9 + sl[1]))
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
