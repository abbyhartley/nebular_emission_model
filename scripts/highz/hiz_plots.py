#!/usr/bin/env python3
"""
Plot high-z validation from cached arrays (hiz_arrays.npz). Fast; no flow sampling.
Figures:
  hiz_tier1_ratios.png  - 2x4: observed vs NF distributions (top) + predicted-vs-observed 1:1 (bottom)
  hiz_tier1_corner.png  - 4x4 BPT-like corner: real DESI-COSMOS (solid) vs NF (dashed) contours
  hiz_tier2.png         - [OIII]/Hb distribution + redshift evolution (real DESI-COSMOS, NO lightcone)
"""
import numpy as np
from scipy.stats import spearmanr, ks_2samp, gaussian_kde
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

REPO = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/"
A = np.load(REPO + "figs_ALTB/hiz_arrays.npz", allow_pickle=True)
DATA_C = "#0072B2"; NF_C = "#CC79A7"
LAB = {"Hbeta": r"$\log_{10}(\mathrm{H}\beta/\mathrm{H}\alpha)$",
       "OII3726": r"$\log_{10}(\mathrm{[OII]}3726/\mathrm{H}\alpha)$",
       "OII3729": r"$\log_{10}(\mathrm{[OII]}3729/\mathrm{H}\alpha)$",
       "OIII5007": r"$\log_{10}(\mathrm{[OIII]}5007/\mathrm{H}\alpha)$"}
NAMES = [str(n) for n in A["t1_names"]]


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


def hpd_thresholds(kde, pts, levels=(0.95, 0.68)):
    d = kde(pts); ds = np.sort(d)[::-1]; cs = np.cumsum(ds); cs /= cs[-1]
    return [ds[np.searchsorted(cs, L)] for L in levels]


# ---------- shared Tier-1 masks ----------
inbox = A["t1_inbox"]; obs = A["t1_obs"]; sn = A["t1_sn"]
predmean = A["t1_predmean"]; preddraw = A["t1_preddraw"]  # (N,4),(N,K,4)
K = preddraw.shape[1]
good_line = [(sn[:, c] > 3) & np.isfinite(obs[:, c]) & inbox for c in range(4)]
all4 = np.all(sn > 3, axis=1) & np.all(np.isfinite(obs), axis=1) & inbox


# =================== FIG 1: distributions + 1:1 ===================
def fig_ratios():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for c, name in enumerate(NAMES):
        g = good_line[c]
        o = obs[g, c]; pmean = predmean[g, c]; pdr = preddraw[g][:, :, c].ravel()
        # shared per-column limits (paper convention: Tukey 3xIQR fence on obs + MC-mean)
        lo, hi = _paper_lims(o, pmean)
        # top: NF-draw vs data distributions (same x-range as the 1:1 panel below)
        ax = axes[0, c]
        bins = np.linspace(lo, hi, 40)
        ax.hist(o, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=DATA_C, label="DESI-COSMOS")
        ax.hist(pdr, bins=bins, density=True, histtype="step", lw=2.4, color=NF_C, label="NF (DESI-trained)")
        ax.set_xlim(lo, hi); ax.set_xlabel(LAB[name]); ax.set_yticks([])
        if c == 0:
            ax.legend(frameon=False, fontsize=12, loc="upper left"); ax.set_ylabel("density")
        # bottom: predicted (MC-mean) vs observed -- same convention as paper in-survey/cross-survey 1:1 figs
        ax = axes[1, c]
        rmse, scat, rho = _paper_stats(o, pmean)
        sp("  %-9s RMSE %.3f scat %.3f rho %.3f (bias %+.3f)" % (name, rmse, scat, rho, np.median(pmean - o)))
        ax.hexbin(o, pmean, gridsize=42, cmap=cmr.bubblegum_r, mincnt=2, bins="log", extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], color="black", lw=3.0, ls=":", alpha=0.95)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("observed " + LAB[name])
        if c == 0:
            ax.set_ylabel("predicted (NF)")
        ax.text(0.03, 0.97, "RMSE=%.3f\nscat=%.3f\n$\\rho$=%.3f" % (rmse, scat, rho),
                transform=ax.transAxes, va="top", ha="left", fontsize=13,
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round,pad=0.2"))
    fig.suptitle(r"Tier 1 ($0.05<z<0.49$): DESI-trained NF vs real DESI-COSMOS spectra", fontsize=17, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = REPO + "figs_ALTB/hiz_tier1_ratios.png"; fig.savefig(out, bbox_inches="tight", dpi=150)
    sp("Saved: " + out)


# =================== FIG 2: BPT-like corner ===================
def fig_corner():
    D = obs[all4]                       # (M,4) real data
    Nf = preddraw[all4].reshape(-1, 4)  # (M*K,4) NF draws
    sp("\nCorner sample: data N=%d, NF draws N=%d" % (len(D), len(Nf)))
    # summary stats
    sp("per-ratio  median(data/NF)  width16-84(data/NF)  KS")
    lims = []
    for c, name in enumerate(NAMES):
        lo, hi = np.percentile(np.concatenate([D[:, c], Nf[:, c]]), [0.5, 99.5]); lims.append((lo, hi))
        md, mn = np.median(D[:, c]), np.median(Nf[:, c])
        wd = 0.5 * (np.percentile(D[:, c], 84) - np.percentile(D[:, c], 16))
        wn = 0.5 * (np.percentile(Nf[:, c], 84) - np.percentile(Nf[:, c], 16))
        ks = ks_2samp(D[:, c], Nf[:, c]).statistic
        sp("  %-9s  %.3f / %.3f     %.3f / %.3f     %.3f" % (name, md, mn, wd, wn, ks))
    sp("pairwise Spearman rho (data | NF):")
    for i in range(4):
        for j in range(i):
            rd = spearmanr(D[:, i], D[:, j]).correlation; rn = spearmanr(Nf[:, i], Nf[:, j]).correlation
            sp("  %-9s x %-9s : %+.3f | %+.3f" % (NAMES[i], NAMES[j], rd, rn))

    n = 4; fig, axes = plt.subplots(n, n, figsize=(13.5, 13.5))
    rng = np.random.default_rng(0)
    dsub = D if len(D) <= 4000 else D[rng.choice(len(D), 4000, replace=False)]
    nsub = Nf if len(Nf) <= 8000 else Nf[rng.choice(len(Nf), 8000, replace=False)]
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            if i == j:
                lo, hi = lims[i]; bins = np.linspace(lo, hi, 40)
                ax.hist(D[:, i], bins=bins, density=True, histtype="step", lw=2.2, color=DATA_C)
                ax.hist(Nf[:, i], bins=bins, density=True, histtype="step", lw=2.2, ls="--", color=NF_C)
                ax.set_xlim(lo, hi); ax.set_yticks([])
            else:
                xl, yl = lims[j], lims[i]
                xg, yg = np.mgrid[xl[0]:xl[1]:80j, yl[0]:yl[1]:80j]
                gp = np.vstack([xg.ravel(), yg.ravel()])
                for pts, col, ls in [(dsub[:, [j, i]].T, DATA_C, "-"), (nsub[:, [j, i]].T, NF_C, "--")]:
                    kde = gaussian_kde(pts)
                    zz = kde(gp).reshape(xg.shape)
                    lv = sorted(hpd_thresholds(kde, pts))
                    ax.contour(xg, yg, zz, levels=lv, colors=col, linewidths=1.8, linestyles=ls)
                ax.set_xlim(*xl); ax.set_ylim(*yl)
            if i == n - 1:
                ax.set_xlabel(LAB[NAMES[j]])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(LAB[NAMES[i]])
            else:
                if not (i == j):
                    ax.set_yticklabels([])
    # legend
    from matplotlib.lines import Line2D
    axes[0, n - 1].axis("on"); axes[0, n - 1].set_frame_on(False)
    axes[0, n - 1].set_xticks([]); axes[0, n - 1].set_yticks([])
    axes[0, n - 1].legend(handles=[Line2D([], [], color=DATA_C, lw=2.4, label="real DESI-COSMOS"),
                                   Line2D([], [], color=NF_C, lw=2.4, ls="--", label="NF (DESI-trained)")],
                          frameon=False, loc="center", fontsize=15)
    fig.suptitle(r"Tier 1 joint line-ratio structure: NF vs real DESI-COSMOS ($0.05<z<0.49$, in-box)", fontsize=17, y=0.94)
    out = REPO + "figs_ALTB/hiz_tier1_corner.png"; fig.savefig(out, bbox_inches="tight", dpi=150)
    sp("Saved: " + out)


# =================== FIG 3: Tier 2 ===================
def fig_tier2():
    z2 = A["t2_z"]; ib = A["t2_inbox"]; o = A["t2_obs_o3hb"]; pm = A["t2_pred_o3hb_mean"]; pdr = A["t2_pred_o3hb_draw"]
    m = ib & np.isfinite(o)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    # distribution
    oo = o[m]; pp = pdr[m].ravel()
    lo, hi = np.percentile(np.concatenate([oo, pp]), [0.5, 99.5]); bins = np.linspace(lo, hi, 40)
    ax[0].hist(oo, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=DATA_C, label="DESI-COSMOS")
    ax[0].hist(pp, bins=bins, density=True, histtype="step", lw=2.4, color=NF_C, label="NF prediction")
    ax[0].set_xlabel(r"$\log_{10}(\mathrm{[OIII]}5007/\mathrm{H}\beta)$"); ax[0].set_yticks([])
    ax[0].set_title(r"$0.49<z<1.0$ (in-box)"); ax[0].legend(frameon=False)
    b = np.median(pm[m] - o[m]); rho = spearmanr(pm[m], o[m]).correlation
    ax[0].text(0.03, 0.97, "bias $%+.3f$\n$\\rho=%.2f$" % (b, rho), transform=ax[0].transAxes,
               va="top", fontsize=12, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    # evolution
    zz = z2[m]; oo2 = o[m]; pp2 = pm[m]; pdr2 = pdr[m]
    zb = np.linspace(0.49, 1.0, 8); zc = 0.5 * (zb[:-1] + zb[1:])
    om, ol, oh, PM, PL, PH = ([] for _ in range(6))
    for a2, b2 in zip(zb[:-1], zb[1:]):
        s = (zz >= a2) & (zz < b2)
        if s.sum() < 15:
            for L in (om, ol, oh, PM, PL, PH): L.append(np.nan)
            continue
        om.append(np.median(oo2[s])); ol.append(np.percentile(oo2[s], 16)); oh.append(np.percentile(oo2[s], 84))
        PM.append(np.median(pdr2[s].ravel())); PL.append(np.percentile(pdr2[s].ravel(), 16)); PH.append(np.percentile(pdr2[s].ravel(), 84))
    om, ol, oh, PM, PL, PH = map(np.array, (om, ol, oh, PM, PL, PH))
    ax[1].fill_between(zc, ol, oh, color=DATA_C, alpha=0.2)
    ax[1].plot(zc, om, "-o", color=DATA_C, lw=2, label="DESI-COSMOS (obs)")
    ax[1].plot(zc, PM, "--s", color=NF_C, lw=2, label="NF prediction")
    ax[1].fill_between(zc, PL, PH, color=NF_C, alpha=0.18)
    ax[1].set_xlabel("redshift"); ax[1].set_ylabel(r"$\log_{10}(\mathrm{[OIII]}/\mathrm{H}\beta)$")
    ax[1].set_title(r"median evolution to $z\sim1$"); ax[1].legend(frameon=False)
    fig.suptitle(r"Tier 2 ($0.49<z<1.0$): real DESI-COSMOS spectra, L$_{\mathrm{H}\alpha}$ from H$\beta$ (no lightcone)",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    out = REPO + "figs_ALTB/hiz_tier2.png"; fig.savefig(out, bbox_inches="tight", dpi=150)
    sp("Saved: " + out)


if __name__ == "__main__":
    fig_ratios(); fig_corner(); fig_tier2(); sp("=== DONE ===")
