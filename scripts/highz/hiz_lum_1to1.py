#!/usr/bin/env python3
"""
LUMINOSITY-space analog of hiz_ratios_1to1.py -- directly comparable to the paper's
in-survey/cross-survey figures (which plot log L_pred vs log L_obs).
NF is conditioned on (log M*, log L_Ha). To turn the predicted RATIO into a predicted
luminosity we add a reference line luminosity: Tier 1 uses observed Ha (= the L_Ha
conditioning); Tier 2 uses observed Hb (Ha out of the DESI window).
  L_pred = ref_obs + (MC-mean predicted ratio) ; L_obs = measured line luminosity.
Same convention as in_survey_lums_fig_pretty_ALTB.py: MC-mean, Tukey fence, aspect equal,
dotted 1:1, bubblegum_dense cmap, RMSE/scat/rho box. rho is HIGH here (shared reference
luminosity), RMSE/scat identical to ratio space (reference cancels in the residual).
Hb is NOT shown for Tier 2: it defines the L_Ha input there (via the Balmer decrement),
so a Hb panel would be a circular consistency check, not an independent test.
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
from matplotlib.colors import LinearSegmentedColormap as _LSC
try:
    import scienceplots  # noqa
    plt.style.use(["science", "no-latex"])
except Exception:
    pass
import cmasher as cmr
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 14, "axes.titlesize": 18, "figure.dpi": 130})

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
FLUX_SCALE = 1e-17; MASS_ZP = 0.13
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
IHB, IOII_A, IOII_B, IOIII = 0, 5, 6, 7
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
CMAP = _LSC.from_list("bubblegum_dense", ["white", cmr.bubblegum(0.85), cmr.bubblegum(0.6),
                                          cmr.bubblegum(0.35), cmr.bubblegum(0.0)])
GRID = 70; MINCNT = 2
TITLE = {"Hbeta": r"H$\beta$", "OII3726": r"[OII]3726", "OII3729": r"[OII]3729", "OIII5007": r"[OIII]5007"}
_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))


def sp(m): print(m, flush=True)


def _paper_lims(x, y):
    xy = np.concatenate([np.asarray(x, float).ravel(), np.asarray(y, float).ravel()])
    xy = xy[np.isfinite(xy)]
    q1, q3 = np.percentile(xy, [25, 75]); fe = 3.0 * (q3 - q1)
    kp = xy[(xy >= q1 - fe) & (xy <= q3 + fe)]
    if kp.size == 0: kp = xy
    loo, hii = float(kp.min()), float(kp.max()); pad = 0.04 * (hii - loo)
    return loo - pad, hii + pad


def _paper_stats(true, pred):
    r = pred - true
    p16, p84 = np.percentile(r, [16, 84])
    return float(np.sqrt(np.mean(r ** 2))), float(0.5 * (p84 - p16)), float(spearmanr(true, pred).correlation)


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def decode(a):
    a = np.asarray(a)
    if a.dtype.kind == "S": a = np.char.decode(a)
    return np.char.strip(a.astype(str))


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
    mean = np.zeros((n, 8)); key = jr.key(seed + 7)
    for lo in range(0, n, batch):
        hi = min(n, lo + batch); Un = Un_all[lo:hi]; acc = np.zeros((hi - lo, 8))
        for j in range(n_mc):
            key, sk = jr.split(key); keys = jr.split(sk, hi - lo)
            Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un))
            acc += Xn * Xs + Xm
        mean[lo:hi] = acc / n_mc
    return mean


def hex_panel(ax, o, pm, xlab, title=None):
    lo, hi = _paper_lims(o, pm)
    rmse, scat, rho = _paper_stats(o, pm)
    hb = ax.hexbin(o, pm, gridsize=GRID, extent=(lo, hi, lo, hi), bins="log", mincnt=MINCNT, cmap=CMAP)
    ax.plot([lo, hi], [lo, hi], color="black", lw=2.6, ls=":", alpha=0.95)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.text(0.04, 0.96, "RMSE=%.3f\nscat=%.3f\n$\\rho$=%.3f" % (rmse, scat, rho), transform=ax.transAxes,
            va="top", ha="left", fontsize=12.5,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round,pad=0.2"))
    return hb, (rmse, scat, rho)


def main():
    COLS = ["Hbeta", "OII3726", "OII3729", "OIII5007"]

    # ---- Tier 1 luminosities from cache: L = t1_loglha (obs L_Ha) + ratio ----
    A = np.load(REPO + "figs_ALTB/hiz_arrays.npz", allow_pickle=True)
    t1_inbox = A["t1_inbox"]; t1_obs = A["t1_obs"]; t1_sn = A["t1_sn"]; t1_pm = A["t1_predmean"]
    t1_ref = A["t1_loglha"]  # observed log L_Ha (Tier-1 conditioning)
    T1 = [str(n) for n in A["t1_names"]]

    # ---- Tier 2 computed; reference = observed log L_Hb ----
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
    calm = base & (z < 0.49) & (Sha > 3) & (Shb > 3) & (Fha > 0) & (Fhb > 0)
    dec = (log10_lum(z, Fha) - log10_lum(z, Fhb))[calm]; mm = logm[calm]
    mbins = np.percentile(mm, np.linspace(0, 100, 9)); mcen = 0.5 * (mbins[:-1] + mbins[1:])
    Rm = np.array([np.median(dec[(mm >= a) & (mm < b)]) for a, b in zip(mbins[:-1], mbins[1:])])
    R_of_M = lambda m: np.interp(m, mcen, Rm, left=Rm[0], right=Rm[-1])
    t2 = base & (z >= 0.49) & (z < 1.00) & (Shb > 3) & (Fhb > 0)
    idx = np.where(t2)[0]
    zt = z[idx]; mt = logm[idx]; LHb = log10_lum(zt, Fhb[idx]); LHa = LHb + R_of_M(mt)
    inbox = (mt >= BOX_M[0]) & (mt <= BOX_M[1]) & (LHa >= BOX_L[0]) & (LHa <= BOX_L[1])
    flow, meta = load_flow()
    mean8 = sample_mean(flow, meta, mt, LHa, seed=11)
    T2Lobs = {"OIII5007": log10_lum(zt, Fo3[idx]), "OII3726": log10_lum(zt, Foa[idx]), "OII3729": log10_lum(zt, Fob[idx])}
    T2Lpred = {"OIII5007": LHb + (mean8[:, IOIII] - mean8[:, IHB]),
               "OII3726": LHb + (mean8[:, IOII_A] - mean8[:, IHB]),
               "OII3729": LHb + (mean8[:, IOII_B] - mean8[:, IHB])}
    T2SN = {"OIII5007": So3[idx], "OII3726": Soa[idx], "OII3729": Sob[idx]}

    # ---- figure ----
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.7))
    hb_last = None
    XL = r"observed $\log L$ [erg s$^{-1}$]"
    sp("TIER 1 luminosities (ref=obs L_Ha):")
    for jc, col in enumerate(COLS):
        i = T1.index(col)
        g = (t1_sn[:, i] > 3) & np.isfinite(t1_obs[:, i]) & t1_inbox
        Lo = t1_ref[g] + t1_obs[g, i]; Lp = t1_ref[g] + t1_pm[g, i]
        hb_last, r = hex_panel(axes[0, jc], Lo, Lp, XL, title=TITLE[col])
        sp("  %-9s N=%5d RMSE=%.3f scat=%.3f rho=%.3f" % (col, g.sum(), *r))
    sp("TIER 2 luminosities (ref=obs L_Hb):")
    axes[1, 0].axis("off")
    for jc, col in enumerate(COLS):
        if col == "Hbeta":
            continue
        g = (T2SN[col] > 3) & (Shb[idx] > 3) & np.isfinite(T2Lobs[col]) & inbox
        hb_last, r = hex_panel(axes[1, jc], T2Lobs[col][g], T2Lpred[col][g], XL)
        sp("  %-9s N=%5d RMSE=%.3f scat=%.3f rho=%.3f" % (col, g.sum(), *r))
    axes[0, 0].set_ylabel(r"predicted $\log L$")
    axes[1, 1].set_ylabel(r"predicted $\log L$")
    fig.text(0.010, 0.75, "Tier 1\n$0.05<z<0.49$", rotation=90, va="center", ha="left", fontsize=16)
    fig.text(0.010, 0.29, "Tier 2\n$0.49<z<1.0$", rotation=90, va="center", ha="left", fontsize=16)
    fig.tight_layout(rect=[0.055, 0, 0.93, 0.99])
    cax = fig.add_axes([0.945, 0.13, 0.014, 0.72])
    cb = fig.colorbar(hb_last, cax=cax); cb.set_label(r"$\log_{10}(N)$ per hexbin", fontsize=15)
    out = REPO + "figs_ALTB/hiz_lum_1to1.png"; fig.savefig(out, bbox_inches="tight", dpi=200); sp("Saved: " + out)
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
