#!/usr/bin/env python3
"""
Tier-2 (0.49<z<1.0) analogs of the Tier-1 ratio + corner figures.
Ha is out of the DESI window, so ratios are taken relative to Hb (observed to z~1.02).
Observable flow lines here: [OIII]5007, [OII]3726, [OII]3729  ->  ratios /Hb.
Conditioning L_Ha from the mass-dependent decrement proxy  L_Ha = L_Hb + R(M*).
Outputs:
  hiz_tier2_ratios.png  - 2x3: obs vs NF distributions (top) + predicted-vs-observed 1:1 (bottom)
  hiz_tier2_corner.png  - 3x3 corner: real DESI-COSMOS (solid) vs NF (dashed)
"""
import pickle, numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr, ks_2samp, gaussian_kde
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
import cmasher as cmr
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 13, "axes.titlesize": 15, "figure.dpi": 130})


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


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
FLUX_SCALE = 1e-17; MASS_ZP = 0.13
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
IHB, IOII_A, IOII_B, IOIII = 0, 5, 6, 7
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
DATA_C = "#0072B2"; NF_C = "#CC79A7"
# ratios relative to Hb: (name, flow index for numerator)
RATIOS = [("OIII5007", IOIII), ("OII3726", IOII_A), ("OII3729", IOII_B)]
LAB = {"OIII5007": r"$\log_{10}(\mathrm{[OIII]}5007/\mathrm{H}\beta)$",
       "OII3726": r"$\log_{10}(\mathrm{[OII]}3726/\mathrm{H}\beta)$",
       "OII3729": r"$\log_{10}(\mathrm{[OII]}3729/\mathrm{H}\beta)$"}
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


def sample(flow, meta, logm, loglha, seed, n_mc, kdraw, batch=40000):
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


def hpd_levels(kde, pts, levels=(0.95, 0.68)):
    d = kde(pts); ds = np.sort(d)[::-1]; cs = np.cumsum(ds); cs /= cs[-1]
    return sorted([ds[np.searchsorted(cs, L)] for L in levels])


def main():
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
    mbins = np.percentile(mm, np.linspace(0, 100, 9)); mcen = 0.5 * (mbins[:-1] + mbins[1:])
    Rm = np.array([np.median(dec[(mm >= a) & (mm < b)]) for a, b in zip(mbins[:-1], mbins[1:])])
    R_of_M = lambda m: np.interp(m, mcen, Rm, left=Rm[0], right=Rm[-1])

    # Tier-2 sample
    t2 = base & (z >= 0.49) & (z < 1.00) & (Shb > 3) & (Fhb > 0)
    idx = np.where(t2)[0]
    zt = z[idx]; mt = logm[idx]; LHb = log10_lum(zt, Fhb[idx])
    LHa = LHb + R_of_M(mt)
    inbox = (mt >= BOX_M[0]) & (mt <= BOX_M[1]) & (LHa >= BOX_L[0]) & (LHa <= BOX_L[1])
    sp("Tier-2 N=%d (Hb S/N>3), in-box=%d" % (len(idx), inbox.sum()))
    flow, meta = load_flow()
    mean8, draws8 = sample(flow, meta, mt, LHa, seed=11, n_mc=40, kdraw=8)

    # observed & predicted ratios /Hb
    obs = {}; sn = {}
    obs["OIII5007"] = np.log10(Fo3[idx]) - np.log10(Fhb[idx]); sn["OIII5007"] = So3[idx]
    obs["OII3726"] = np.log10(Foa[idx]) - np.log10(Fhb[idx]); sn["OII3726"] = Soa[idx]
    obs["OII3729"] = np.log10(Fob[idx]) - np.log10(Fhb[idx]); sn["OII3729"] = Sob[idx]
    predmean = {n: mean8[:, j] - mean8[:, IHB] for n, j in RATIOS}
    preddraw = {n: draws8[:, :, j] - draws8[:, :, IHB] for n, j in RATIOS}

    # ===== FIG 1: distributions + 1:1 =====
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    sp("\nTier-2 /Hb ratio metrics (in-box):")
    for c, (name, j) in enumerate(RATIOS):
        g = (sn[name] > 3) & (Shb[idx] > 3) & np.isfinite(obs[name]) & inbox
        o = obs[name][g]; pm = predmean[name][g]; pdr = preddraw[name][g].ravel()
        lo, hi = _paper_lims(o, pm)
        rmse, scat, rho = _paper_stats(o, pm)
        sp("  %-9s N=%5d RMSE=%.3f scat=%.3f rho=%.3f (bias %+.3f)" % (name, g.sum(), rmse, scat, rho, np.median(pm - o)))
        ax = axes[0, c]
        bins = np.linspace(lo, hi, 38)
        ax.hist(o, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=DATA_C, label="DESI-COSMOS")
        ax.hist(pdr, bins=bins, density=True, histtype="step", lw=2.3, color=NF_C, label="NF (DESI-trained)")
        ax.set_xlim(lo, hi); ax.set_xlabel(LAB[name]); ax.set_yticks([])
        if c == 0:
            ax.legend(frameon=False, fontsize=11, loc="upper left"); ax.set_ylabel("density")
        ax = axes[1, c]
        ax.hexbin(o, pm, gridsize=42, cmap=cmr.bubblegum_r, mincnt=2, bins="log", extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], color="black", lw=3.0, ls=":", alpha=0.95)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("observed " + LAB[name])
        if c == 0:
            ax.set_ylabel("predicted (NF)")
        ax.text(0.03, 0.97, "RMSE=%.3f\nscat=%.3f\n$\\rho$=%.3f" % (rmse, scat, rho), transform=ax.transAxes,
                va="top", ha="left", fontsize=11, bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round,pad=0.2"))
    fig.suptitle(r"Tier 2 ($0.49<z<1.0$): DESI-trained NF vs real DESI-COSMOS spectra (ratios rel. H$\beta$)",
                 fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = REPO + "figs_ALTB/hiz_tier2_ratios.png"; fig.savefig(out, bbox_inches="tight", dpi=150); sp("Saved: " + out)

    # ===== FIG 2: corner =====
    names = [n for n, _ in RATIOS]
    allg = (So3[idx] > 3) & (Soa[idx] > 3) & (Sob[idx] > 3) & (Shb[idx] > 3) & inbox
    for n in names:
        allg &= np.isfinite(obs[n])
    D = np.column_stack([obs[n][allg] for n in names])
    Nf = np.column_stack([preddraw[n][allg].reshape(-1) for n in names])
    sp("\nCorner sample: data N=%d, NF draws N=%d" % (len(D), len(Nf)))
    sp("per-ratio  median(data/NF)  width(data/NF)  KS")
    lims = []
    for c, n in enumerate(names):
        lo, hi = np.percentile(np.concatenate([D[:, c], Nf[:, c]]), [0.5, 99.5]); lims.append((lo, hi))
        wd = 0.5 * (np.percentile(D[:, c], 84) - np.percentile(D[:, c], 16))
        wn = 0.5 * (np.percentile(Nf[:, c], 84) - np.percentile(Nf[:, c], 16))
        sp("  %-9s  %.3f / %.3f   %.3f / %.3f   %.3f"
           % (n, np.median(D[:, c]), np.median(Nf[:, c]), wd, wn, ks_2samp(D[:, c], Nf[:, c]).statistic))
    sp("pairwise Spearman rho (data | NF):")
    for i in range(3):
        for jj in range(i):
            sp("  %-9s x %-9s : %+.3f | %+.3f" % (names[i], names[jj],
               spearmanr(D[:, i], D[:, jj]).correlation, spearmanr(Nf[:, i], Nf[:, jj]).correlation))

    nn = 3; fig, axes = plt.subplots(nn, nn, figsize=(11.5, 11.5))
    rng = np.random.default_rng(0)
    dsub = D if len(D) <= 4000 else D[rng.choice(len(D), 4000, replace=False)]
    nsub = Nf if len(Nf) <= 8000 else Nf[rng.choice(len(Nf), 8000, replace=False)]
    for i in range(nn):
        for jj in range(nn):
            ax = axes[i, jj]
            if jj > i:
                ax.axis("off"); continue
            if i == jj:
                lo, hi = lims[i]; bins = np.linspace(lo, hi, 40)
                ax.hist(D[:, i], bins=bins, density=True, histtype="step", lw=2.2, color=DATA_C)
                ax.hist(Nf[:, i], bins=bins, density=True, histtype="step", lw=2.2, ls="--", color=NF_C)
                ax.set_xlim(lo, hi); ax.set_yticks([])
            else:
                xl, yl = lims[jj], lims[i]
                xg, yg = np.mgrid[xl[0]:xl[1]:80j, yl[0]:yl[1]:80j]; gp = np.vstack([xg.ravel(), yg.ravel()])
                for pts, col, ls in [(dsub[:, [jj, i]].T, DATA_C, "-"), (nsub[:, [jj, i]].T, NF_C, "--")]:
                    kde = gaussian_kde(pts); zz = kde(gp).reshape(xg.shape)
                    ax.contour(xg, yg, zz, levels=hpd_levels(kde, pts), colors=col, linewidths=1.8, linestyles=ls)
                ax.set_xlim(*xl); ax.set_ylim(*yl)
            if i == nn - 1:
                ax.set_xlabel(LAB[names[jj]])
            else:
                ax.set_xticklabels([])
            if jj == 0 and i != 0:
                ax.set_ylabel(LAB[names[i]])
            elif i != jj:
                ax.set_yticklabels([])
    from matplotlib.lines import Line2D
    axes[0, nn - 1].axis("on"); axes[0, nn - 1].set_frame_on(False)
    axes[0, nn - 1].set_xticks([]); axes[0, nn - 1].set_yticks([])
    axes[0, nn - 1].legend(handles=[Line2D([], [], color=DATA_C, lw=2.4, label="real DESI-COSMOS"),
                                    Line2D([], [], color=NF_C, lw=2.4, ls="--", label="NF (DESI-trained)")],
                           frameon=False, loc="center", fontsize=14)
    fig.suptitle(r"Tier 2 joint structure: NF vs real DESI-COSMOS ($0.49<z<1.0$, in-box, rel. H$\beta$)",
                 fontsize=16, y=0.93)
    out = REPO + "figs_ALTB/hiz_tier2_corner.png"; fig.savefig(out, bbox_inches="tight", dpi=150); sp("Saved: " + out)
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
