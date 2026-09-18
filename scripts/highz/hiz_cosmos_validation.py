#!/usr/bin/env python3
"""
High-z validation of the DESI-trained NF on real DESI-COSMOS spectra.

PRIMARY (object-level), two tiers:
  Tier 1  (0.05<z<0.49):  Ha in window -> condition on OBSERVED logL_Ha + color-scale M*,
                          validate predicted vs observed  Hb/Ha, [OIII]5007/Ha, [OII]3726/Ha, [OII]3729/Ha
  Tier 2  (0.49<z<1.00):  Ha out of window -> logL_Ha from observed Hb x R (R=3.90),
                          validate predicted vs observed  [OIII]5007/Hb

Consistency with training:
  - line fluxes (1e-17 erg/s/cm2) get the SAME per-line SDSS-scale calibration used in training
  - logL_Ha: Planck15, FLUX_SCALE=1e-17 (identical to eval_altb_table1.log10_lum)
  - M* = COSMOS2020 lp_mass_med + 0.13  (measured LePhare->LOGM_COLOR zero-point on same-galaxy pairs)
  - conditioning + targets standardized with the flow's own meta
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
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "legend.fontsize": 13, "axes.titlesize": 15, "figure.dpi": 130})


def sp(msg):
    print(msg, flush=True)


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
FLUX_SCALE = 1e-17
MASS_ZP = 0.13
R_DECREMENT = 3.90
_ROBUST_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
IHB, IOII_A, IOII_B, IOIII = 0, 5, 6, 7
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)


def log10_lum(z, f1e17):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f1e17, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def load_desi_flow():
    meta = pickle.load(open(REPO + "models/nf_desi_ALTB_meta.pkl", "rb"))
    xdim = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xdim)), cond_dim=2, inverter=_ROBUST_INV)
    return eqx.tree_deserialise_leaves(REPO + "models/nf_desi_ALTB.eqx", tmpl), meta


def sample_ratios(flow, meta, logm, loglha, seed, n_mc, batch=40_000):
    U = np.column_stack([logm, loglha]).astype(np.float32)
    Un_all = jnp.asarray((U - meta["U_mean"]) / meta["U_std"])
    Xm, Xs = np.asarray(meta["X_mean"]), np.asarray(meta["X_std"]); n = len(U)
    mean = np.zeros((n, 8)); one = np.zeros((n, 8)); key = jr.key(seed + 7)
    for lo in range(0, n, batch):
        hi = min(n, lo + batch); Un = Un_all[lo:hi]; acc = np.zeros((hi - lo, 8))
        for j in range(n_mc):
            key, sk = jr.split(key); keys = jr.split(sk, hi - lo)
            Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un))
            r = Xn * Xs + Xm; acc += r
            if j == 0:
                one[lo:hi] = r
        mean[lo:hi] = acc / n_mc
    return mean, one


def stats(pred, obs):
    r = pred - obs; g = np.isfinite(r); r = r[g]
    return dict(N=int(g.sum()), bias=float(np.median(r)),
                scat=float(0.5 * (np.percentile(r, 84) - np.percentile(r, 16))),
                rmse=float(np.sqrt(np.mean(r**2))),
                rho=float(spearmanr(pred[g], obs[g]).correlation))


def decode(a):
    a = np.asarray(a)
    if a.dtype.kind == "S":
        a = np.char.decode(a)
    return np.char.strip(a.astype(str))


def main():
    h = fits.open(COSMOS)
    d1, d6 = h[1].data, h[6].data
    z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
    sp("loading %d DESI-COSMOS rows" % len(z))
    stype = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float)

    def flux(name):
        f = np.asarray(d1[name + "_FLUX"], float) * 10.0 ** CAL[name]
        iv = np.asarray(d1[name + "_FLUX_IVAR"], float)
        sn = np.asarray(d1[name + "_FLUX"], float) * np.sqrt(np.clip(iv, 0, None))
        return f, sn
    Fha, SNha = flux("HALPHA"); Fhb, SNhb = flux("HBETA")
    Foa, SNoa = flux("OII_3726"); Fob, SNob = flux("OII_3729"); Fo3, SNo3 = flux("OIII_5007")

    base = (zwarn == 0) & (stype == "GALAXY") & np.isfinite(lpm) & (lpm > 6) & (lpm < 13) & (z > 0.05)
    logm = lpm + MASS_ZP
    sp("base sample (zwarn0, GALAXY, valid mass, z>0.05): %d" % base.sum())

    flow, meta = load_desi_flow()

    # ---------------- TIER 1 ----------------
    t1 = base & (z < 0.49) & (SNha > 3) & (Fha > 0)
    loglha1 = log10_lum(z, Fha)
    sp("\n===== TIER 1  (0.05<z<0.49, Ha observed) =====")
    sp("  candidates with Ha S/N>3: %d" % t1.sum())
    lines_t1 = [("Hbeta", IHB, Fhb, SNhb), ("OII3726", IOII_A, Foa, SNoa),
                ("OII3729", IOII_B, Fob, SNob), ("OIII5007", IOIII, Fo3, SNo3)]
    idx = np.where(t1)[0]
    mean8, one8 = sample_ratios(flow, meta, logm[idx], loglha1[idx], seed=11, n_mc=40)
    inbox = ((logm[idx] >= BOX_M[0]) & (logm[idx] <= BOX_M[1]) &
             (loglha1[idx] >= BOX_L[0]) & (loglha1[idx] <= BOX_L[1]))
    sp("  in training (M*,L_Ha) box: %d / %d (%.1f%%)" % (inbox.sum(), len(idx), 100 * inbox.mean()))
    z1 = z[idx]
    t1res = {}
    for name, j, F, SN in lines_t1:
        obs = np.log10(F[idx]) - np.log10(Fha[idx])
        good = (SN[idx] > 3) & (F[idx] > 0) & np.isfinite(obs)
        for tag, msk in [("all", good), ("inbox", good & inbox)]:
            st = stats(mean8[msk, j], obs[msk])
            t1res[(name, tag)] = (st, mean8[msk, j], one8[msk, j], obs[msk], z1[msk])
            sp("  %-9s [%-5s] N=%5d  bias=%+.3f  scat=%.3f  rmse=%.3f  rho=%.3f"
               % (name, tag, st["N"], st["bias"], st["scat"], st["rmse"], st["rho"]))

    # ---------------- TIER 2 ----------------
    sp("\n===== TIER 2  (0.49<z<1.00, Ha out of window; L_Ha from Hb x %.2f) =====" % R_DECREMENT)
    t2 = base & (z >= 0.49) & (z < 1.00) & (SNhb > 3) & (SNo3 > 3) & (Fhb > 0) & (Fo3 > 0)
    loglhb = log10_lum(z, Fhb); loglha2 = loglhb + np.log10(R_DECREMENT)
    idx2 = np.where(t2)[0]
    sp("  candidates (Hb & [OIII] S/N>3): %d" % len(idx2))
    inbox2 = ((logm[idx2] >= BOX_M[0]) & (logm[idx2] <= BOX_M[1]) &
              (loglha2[idx2] >= BOX_L[0]) & (loglha2[idx2] <= BOX_L[1]))
    sp("  in training box: %d / %d (%.1f%%)" % (inbox2.sum(), len(idx2), 100 * inbox2.mean()))
    mean8b, one8b = sample_ratios(flow, meta, logm[idx2], loglha2[idx2], seed=22, n_mc=40)
    pred_o3hb = mean8b[:, IOIII] - mean8b[:, IHB]
    pred_o3hb_1 = one8b[:, IOIII] - one8b[:, IHB]
    obs_o3hb = np.log10(Fo3[idx2]) - np.log10(Fhb[idx2])
    for tag, msk in [("all", np.isfinite(obs_o3hb)), ("inbox", np.isfinite(obs_o3hb) & inbox2)]:
        st = stats(pred_o3hb[msk], obs_o3hb[msk])
        sp("  [OIII]/Hb [%-5s] N=%5d  bias=%+.3f  scat=%.3f  rmse=%.3f  rho=%.3f"
           % (tag, st["N"], st["bias"], st["scat"], st["rmse"], st["rho"]))
    for Rtest in (3.5, 4.5):
        lh = loglhb[idx2] + np.log10(Rtest)
        m8, _ = sample_ratios(flow, meta, logm[idx2], lh, seed=33, n_mc=15)
        p = m8[:, IOIII] - m8[:, IHB]; msk = np.isfinite(obs_o3hb) & inbox2
        sp("    R=%.1f -> [OIII]/Hb inbox bias=%+.3f" % (Rtest, np.median((p - obs_o3hb)[msk])))
    z2 = z[idx2]

    make_figure(t1res, lines_t1, pred_o3hb, pred_o3hb_1, obs_o3hb, z2, inbox2)
    sp("\n=== DONE ===")


def make_figure(t1res, lines_t1, pred_o3hb, pred_o3hb_1, obs_o3hb, z2, inbox2):
    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.34)
    labels = {"Hbeta": r"$\log_{10}(\mathrm{H}\beta/\mathrm{H}\alpha)$",
              "OII3726": r"$\log_{10}(\mathrm{[OII]}3726/\mathrm{H}\alpha)$",
              "OII3729": r"$\log_{10}(\mathrm{[OII]}3729/\mathrm{H}\alpha)$",
              "OIII5007": r"$\log_{10}(\mathrm{[OIII]}5007/\mathrm{H}\alpha)$"}
    for k, (name, j, F, SN) in enumerate(lines_t1):
        ax = fig.add_subplot(gs[0, k])
        st, predmean, predone, obs, zc = t1res[(name, "inbox")]
        lo, hi = np.percentile(np.concatenate([obs, predone]), [1, 99]); bins = np.linspace(lo, hi, 34)
        ax.hist(obs, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color="#3b6fb0", label="DESI-COSMOS")
        ax.hist(predone, bins=bins, density=True, histtype="step", lw=2.2, color="#d072d6", label="NF (DESI-trained)")
        ax.set_xlabel(labels[name]); ax.set_yticks([])
        ax.set_title(r"$z<0.49$ (bias $%+.2f$)" % st["bias"])
        if k == 0:
            ax.legend(frameon=False, loc="upper left", fontsize=11); ax.set_ylabel("density")
    # Tier-1 [OIII]/Ha 1:1
    ax = fig.add_subplot(gs[1, 0])
    st, pm, po, obs, zc = t1res[("OIII5007", "inbox")]
    ax.plot([-1.4, 0.7], [-1.4, 0.7], "k--", lw=1)
    ax.scatter(obs, pm, s=4, alpha=0.22, color="#d072d6", rasterized=True)
    ax.set_xlabel(r"observed $\log_{10}(\mathrm{[OIII]}/\mathrm{H}\alpha)$"); ax.set_ylabel("predicted")
    ax.set_title(r"Tier 1: [OIII]/H$\alpha$ 1:1"); ax.set_xlim(-1.4, 0.7); ax.set_ylim(-1.4, 0.7)
    # Tier-2 [OIII]/Hb distribution (in-box)
    ax = fig.add_subplot(gs[1, 1])
    mfin = inbox2 & np.isfinite(obs_o3hb)
    o = obs_o3hb[mfin]; p = pred_o3hb_1[mfin]
    lo, hi = np.percentile(np.concatenate([o, p]), [1, 99]); bins = np.linspace(lo, hi, 34)
    ax.hist(o, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color="#3b6fb0", label="DESI-COSMOS")
    ax.hist(p, bins=bins, density=True, histtype="step", lw=2.2, color="#d072d6", label="NF")
    ax.set_xlabel(r"$\log_{10}(\mathrm{[OIII]}5007/\mathrm{H}\beta)$"); ax.set_yticks([])
    ax.set_title(r"Tier 2: $0.49<z<1.0$"); ax.legend(frameon=False, fontsize=11)
    # Tier-2 [OIII]/Hb vs redshift
    ax = fig.add_subplot(gs[1, 2:4])
    zz = z2[mfin]; oo = obs_o3hb[mfin]; pp = pred_o3hb[mfin]
    zb = np.linspace(0.49, 1.0, 8); zcen = 0.5 * (zb[:-1] + zb[1:])
    om, ol, oh, pm2, pl, ph = ([] for _ in range(6))
    for a, b in zip(zb[:-1], zb[1:]):
        s = (zz >= a) & (zz < b)
        if s.sum() < 15:
            for L in (om, ol, oh, pm2, pl, ph): L.append(np.nan)
            continue
        om.append(np.median(oo[s])); ol.append(np.percentile(oo[s], 16)); oh.append(np.percentile(oo[s], 84))
        pm2.append(np.median(pp[s])); pl.append(np.percentile(pp[s], 16)); ph.append(np.percentile(pp[s], 84))
    om, ol, oh, pm2, pl, ph = map(np.array, (om, ol, oh, pm2, pl, ph))
    ax.fill_between(zcen, ol, oh, color="#3b6fb0", alpha=0.25)
    ax.plot(zcen, om, "-o", color="#3b6fb0", lw=2, label="DESI-COSMOS (obs)")
    ax.fill_between(zcen, pl, ph, color="#d072d6", alpha=0.25)
    ax.plot(zcen, pm2, "--s", color="#d072d6", lw=2, label="NF prediction")
    ax.set_xlabel("redshift"); ax.set_ylabel(r"$\log_{10}(\mathrm{[OIII]}/\mathrm{H}\beta)$")
    ax.set_title(r"[OIII]/H$\beta$ evolution to $z\sim1$"); ax.legend(frameon=False)
    fig.suptitle("High-z validation: DESI-trained NF vs real DESI-COSMOS spectra", fontsize=16, y=0.98)
    out = REPO + "figs_ALTB/hiz_cosmos_validation.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    sp("Saved: " + out)


if __name__ == "__main__":
    main()
