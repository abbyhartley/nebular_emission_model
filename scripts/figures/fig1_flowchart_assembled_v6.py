"""
Figure 1 flow chart v6  -- ALT-B production model.

Three-stage pipeline mirroring the paper:
   [Inputs / Targets]  ->  [Conditional density estimator]  ->  [Predictive accuracy / Joint structure]

v6 (author feedback):
  - Middle panel restored, now a proper NF-architecture visual: base density
    z~N(0,I) -> stack of invertible block-neural-autoregressive layers f(X;u)
    (conditioned on u) -> learned conditional density p(X|u); sample/evaluate
    directions indicated. (No redundant BPT here.)
  - Pink cards kept but layout tightened: 3 columns, aligned rows, less whitespace.
  - Left = Inputs (logM*, logL_Ha) + Targets (schematic spectrum with 10-90% bands).
  - Right = Predictive accuracy (H-beta, best case) + Joint structure (BPT, data vs NF purple).
"""
from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401
import cmasher as cmr

from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from functools import partial as _pf
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces

# ----------------------------------------------------------------------
BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
SDSS_FITS = Path(BASE + "SDSS_main_training_data_ALTB.fits")
DESI_FITS = Path(BASE + "DESI_BGS_training_data_ALTB.fits")
FLOW_DESI, META_DESI = Path(REPO + "models/nf_desi_ALTB.eqx"), Path(REPO + "models/nf_desi_ALTB_meta.pkl")
OUT = REPO + "figs_ALTB/fig1_flowchart_assembled_v6"

FLUX_SCALE = 1e-17
SEED = 0
N_INSET = 40_000
N_CORR = 40_000
N_MC = 15

_ROBUST_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))

CARD = "#FFDFD9"
CARD_EDGE = "#e9a99e"
TXT = "black"
NF_COL = "#d072d6"      # light pinkish-purple: easy to distinguish from black DESI contours
SPEC_LINE = cmr.bubblegum(0.06)
SPEC_BAND = "#9b59c9"
DENSE = LinearSegmentedColormap.from_list(
    "bubblegum_dense", ["white", cmr.bubblegum(0.85), cmr.bubblegum(0.6),
                        cmr.bubblegum(0.35), cmr.bubblegum(0.0)])
LINE_11_KW = dict(color="black", lw=1.8, ls=":", alpha=0.95)
FIGW, FIGH = 17.0, 8.3
SQ = FIGH / FIGW

SPEC_LINES = {
    "[O II]": (3727.0, ("OII_3726_FLUX", "OII_3729_FLUX")),
    r"H$\gamma$": (4340.0, "HGAMMA_FLUX"),
    r"H$\beta$": (4861.0, "HBETA_FLUX"),
    "[O III]": (5007.0, "OIII_5007_FLUX"),
    r"H$\alpha$": (6563.0, "HALPHA"),
    "[N II]": (6584.0, "NII_6584_FLUX"),
    "[S II]": (6725.0, ("SII_6716_FLUX", "SII_6731_FLUX")),
}


# ----------------------------------------------------------------------
def load_scalar_df(p):
    t = Table.read(p, hdu=1)
    return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()

def thin_df(df, n, seed=0):
    return df.reset_index(drop=True) if len(df) <= n else df.sample(n=n, random_state=seed).reset_index(drop=True)

def log10_lum(z, flux_1e17):
    z = np.asarray(z, float); f = np.asarray(flux_1e17, float) * FLUX_SCALE
    return np.log10(f) + np.log10(4 * np.pi) + 2 * np.log10(cosmo.luminosity_distance(z).to("cm").value)

def add_loglha(df, *, survey):
    df = df.copy()
    if survey == "sdss":
        z = df["Z_1"].to_numpy(float); ha = df["H_ALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    else:
        z = df["Z"].to_numpy(float); ha = df["HALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    m = np.isfinite(z) & (z > 0) & np.isfinite(ha) & (ha > 0)
    loglha = np.full(len(df), np.nan)
    loglha[m] = np.log10(ha[m]) + np.log10(4 * np.pi) + 2 * np.log10(cosmo.luminosity_distance(z[m]).to("cm").value)
    df["LOG_LHA"] = loglha
    return df

def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    template = block_neural_autoregressive_flow(
        key=jr.key(int(meta.get("seed", 0))), base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=2, inverter=_ROBUST_INV)
    return eqx.tree_deserialise_leaves(flow_path, template)

def outcol_index(meta, substrs):
    for i, name in enumerate(meta["resolved"]["out_cols"]):
        if any(s in name for s in substrs):
            return i
    raise KeyError(substrs)

def cond_U(meta, df):
    lm, ll = meta["resolved"]["logmstar_col"], meta["resolved"]["loglha_col"]
    m = np.isfinite(df[lm].to_numpy(float)) & np.isfinite(df[ll].to_numpy(float))
    U = (df.loc[m, [lm, ll]].to_numpy(np.float32) - meta["U_mean"]) / meta["U_std"]
    return m, jnp.asarray(U)

def sample_single(flow, meta, df, *, seed):
    m, U = cond_U(meta, df)
    keys = jr.split(jr.key(seed + 1234), U.shape[0])
    Xn = jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, U)
    return m, np.array(Xn) * meta["X_std"] + meta["X_mean"]

def sample_mcmean(flow, meta, df, *, seed, n_mc=N_MC):
    m, U = cond_U(meta, df)
    key = jr.key(seed + 999)
    acc = np.zeros((U.shape[0], len(meta["resolved"]["out_cols"])), np.float32)
    for _ in range(n_mc):
        key, sk = jr.split(key)
        keys = jr.split(sk, U.shape[0])
        Xn = jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, U)
        acc += np.array(Xn) * meta["X_std"] + meta["X_mean"]
    return m, acc / n_mc

def sq_hexbin(ax, x, y, gridsize=55):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    xy = np.concatenate([x, y]); q1, q3 = np.percentile(xy, [25, 75]); fe = 3.0 * (q3 - q1)
    kp = xy[(xy >= q1 - fe) & (xy <= q3 + fe)]; lo, hi = float(kp.min()), float(kp.max())
    pad = 0.02 * (hi - lo); lo -= pad; hi += pad
    ax.hexbin(x, y, gridsize=gridsize, extent=(lo, hi, lo, hi), bins="log", mincnt=3, cmap=DENSE)
    ax.plot([lo, hi], [lo, hi], **LINE_11_KW)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_facecolor("white")
    return spearmanr(x, y).correlation

def bpt_demarcation(ax):
    xk = np.linspace(-1.9, -0.02, 200)
    ax.plot(xk, 0.61 / (xk - 0.05) + 1.3, ls="--", color="0.4", lw=1.3)
    xk2 = np.linspace(-1.9, 0.30, 200)
    ax.plot(xk2, 0.61 / (xk2 - 0.47) + 1.19, ls=":", color="0.4", lw=1.3)

def bpt_contours(ax, xd, yd, xn, yn, xr, yr):
    for x, y, color in [(xd, yd, "black"), (xn, yn, NF_COL)]:
        m = np.isfinite(x) & np.isfinite(y)
        H, xe, ye = np.histogram2d(x[m], y[m], bins=70, range=[xr, yr])
        H = gaussian_filter(H.T, 1.6)
        Xc = 0.5 * (xe[:-1] + xe[1:]); Yc = 0.5 * (ye[:-1] + ye[1:])
        lv = H.max() * np.array([0.08, 0.25, 0.55, 0.85])
        ax.contour(Xc, Yc, H, levels=lv, colors=[color], linewidths=1.7)
    bpt_demarcation(ax)
    ax.set_xlim(*xr); ax.set_ylim(*yr)

def mini_density(ax, x, y):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    xr = np.percentile(x, [2, 98]); yr = np.percentile(y, [2, 98])
    H, xe, ye = np.histogram2d(x, y, bins=55, range=[xr, yr])
    H = gaussian_filter(H.T, 1.6)
    lv = np.linspace(H.max() * 0.06, H.max(), 8)                 # low tail unfilled -> white bg
    ax.contourf(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), H, levels=lv, cmap=DENSE)
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.6"); s.set_linewidth(0.8)

def ratio_stats(df, col_or_pair, ha, qlo=10, qhi=90):
    if col_or_pair == "HALPHA":
        return 1.0, 1.0, 1.0
    cols = col_or_pair if isinstance(col_or_pair, tuple) else (col_or_pair,)
    f = np.sum([df[c].to_numpy(float) for c in cols], axis=0)
    good = (f > 0) & (ha > 0)
    r = f[good] / ha[good]
    return tuple(np.percentile(r, [qlo, 50, qhi]))

def draw_spectrum(ax, di):
    ha = di["HALPHA_FLUX"].to_numpy(float)
    wl = np.linspace(3650, 6850, 2400)
    cont = 0.05 + 0.003 * (wl - 3650) / 3200
    sig = 7.0
    lo = np.zeros_like(wl); mid = np.zeros_like(wl); hi = np.zeros_like(wl)
    peaks = {}
    for lab, (lam, col) in SPEC_LINES.items():
        p_lo, p50, p_hi = ratio_stats(di, col, ha)
        g = np.exp(-0.5 * ((wl - lam) / sig) ** 2)
        lo += p_lo * g; mid += p50 * g; hi += p_hi * g
        peaks[lab] = (lam, p_hi)
    ax.fill_between(wl, cont + lo, cont + hi, color=SPEC_BAND, alpha=0.55, lw=0, zorder=2)
    ax.plot(wl, cont + hi, color=SPEC_BAND, lw=0.7, alpha=0.8, zorder=2)
    ax.plot(wl, cont + mid, color=SPEC_LINE, lw=1.4, zorder=3)
    ax.plot(wl, cont, color="0.45", lw=0.9, zorder=1)
    off = {"[O II]": (0, 2), r"H$\gamma$": (0, 2), r"H$\beta$": (-8, 2), "[O III]": (10, 2),
           r"H$\alpha$": (0, 3), "[N II]": (-11, 14), "[S II]": (12, 2)}
    for lab, (lam, h) in peaks.items():
        dx, dy = off.get(lab, (0, 3))
        ax.annotate(lab, (lam, cont[np.argmin(abs(wl - lam))] + h), xytext=(dx, dy),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, color=TXT)
    ax.set_xlim(3650, 6850); ax.set_ylim(0, 1.14 * float(np.max(cont + hi)))
    ax.set_yticks([]); ax.set_xlabel(r"rest wavelength [$\AA$]", fontsize=13); ax.tick_params(labelsize=11)


# ----------------------------------------------------------------------
def main():
    Path(REPO + "figs_ALTB").mkdir(exist_ok=True)
    df_d = add_loglha(thin_df(load_scalar_df(DESI_FITS), max(N_INSET, N_CORR), seed=SEED), survey="desi")
    df_s = add_loglha(thin_df(load_scalar_df(SDSS_FITS), N_INSET, seed=SEED + 1), survey="sdss")
    meta_d = pickle.load(open(META_DESI, "rb"))
    flow_d = load_flow(FLOW_DESI, meta_d)
    i_hb = outcol_index(meta_d, ["H_BETA", "HBETA"])
    i_nii = outcol_index(meta_d, ["NII_6584"]); i_oiii = outcol_index(meta_d, ["OIII_5007"])
    i_oii = outcol_index(meta_d, ["OII_3726"]); i_sii = outcol_index(meta_d, ["SII_6716"])

    df_dc = df_d.iloc[:N_CORR].reset_index(drop=True)
    m_bpt, nf_ratios = sample_single(flow_d, meta_d, df_dc, seed=SEED + 10)
    dfb = df_dc.loc[m_bpt].reset_index(drop=True)
    nii = dfb["NII_6584_FLUX"].to_numpy(float); ha = dfb["HALPHA_FLUX"].to_numpy(float)
    oiii = dfb["OIII_5007_FLUX"].to_numpy(float); hb = dfb["HBETA_FLUX"].to_numpy(float)
    gg = (nii > 0) & (ha > 0) & (oiii > 0) & (hb > 0)
    d_nii_ha = np.log10(nii[gg] / ha[gg]); d_oiii_hb = np.log10(oiii[gg] / hb[gg])
    nf_nii_ha = nf_ratios[:, i_nii]; nf_oiii_hb = nf_ratios[:, i_oiii] - nf_ratios[:, i_hb]

    di = df_d.iloc[:N_INSET].reset_index(drop=True)
    m_d, r_dd = sample_mcmean(flow_d, meta_d, di, seed=SEED + 3)
    di = di.loc[m_d].reset_index(drop=True)
    t_dd = log10_lum(di["Z"].to_numpy(float), di["HBETA_FLUX"].to_numpy(float))
    p_dd = di["LOG_LHA"].to_numpy(float) + r_dd[:, i_hb].astype(float)
    m_s, r_ds = sample_mcmean(flow_d, meta_d, df_s, seed=SEED + 4)
    ds = df_s.loc[m_s].reset_index(drop=True)
    t_ds = log10_lum(ds["Z_1"].to_numpy(float), ds["H_BETA_FLUX"].to_numpy(float))
    p_ds = ds["LOG_LHA"].to_numpy(float) + r_ds[:, i_hb].astype(float)

    # ------------------------------------------------------------------
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"axes.labelsize": 16, "axes.titlesize": 16,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "text.color": TXT,
                         "axes.edgecolor": "0.3", "axes.labelcolor": TXT,
                         "xtick.color": TXT, "ytick.color": TXT})
    fig = plt.figure(figsize=(FIGW, FIGH))

    def sqrect(cx, y, h):
        w = SQ * h
        return [cx - w / 2, y, w, h]

    # tightened 3-column layout, wider inter-column gaps (x0, y0, w, h, title)
    cards = {
        "IN": (0.010, 0.550, 0.238, 0.370, "Inputs"),
        "TG": (0.010, 0.080, 0.238, 0.370, "Targets"),
        "MD": (0.320, 0.080, 0.208, 0.840, "Conditional density estimator"),
        "PA": (0.600, 0.550, 0.388, 0.370, "Predictive accuracy"),
        "JS": (0.600, 0.080, 0.388, 0.370, "Joint structure"),
    }
    for key, (x0, y0, w, h, title) in cards.items():
        fig.add_artist(FancyBboxPatch((x0, y0), w, h,
                       boxstyle="round,pad=0.005,rounding_size=0.010",
                       transform=fig.transFigure, facecolor=CARD, edgecolor=CARD_EDGE, lw=1.4, zorder=0))
        fs = 15 if key == "MD" else 18
        fig.text(x0 + w / 2, y0 + h + 0.013, title, ha="center", va="bottom",
                 fontsize=fs, fontweight="bold", color=TXT)
    cx_IN = cards["IN"][0] + cards["IN"][2] / 2
    cx_MD = cards["MD"][0] + cards["MD"][2] / 2
    cx_PA = cards["PA"][0] + cards["PA"][2] / 2
    cx_JS = cards["JS"][0] + cards["JS"][2] / 2

    # ================= Inputs =================
    ax_in = fig.add_axes(sqrect(cx_IN, 0.618, 0.240))
    lm = di[meta_d["resolved"]["logmstar_col"]].to_numpy(float); ll = di["LOG_LHA"].to_numpy(float)
    good = np.isfinite(lm) & np.isfinite(ll)
    ax_in.hexbin(lm[good], ll[good], gridsize=42, bins="log", mincnt=3, cmap=DENSE)
    ax_in.set_facecolor("white")
    ax_in.set_xlabel(r"$\log M_\star\ [M_\odot]$", fontsize=14)
    ax_in.set_ylabel(r"$\log L_{H\alpha}$", fontsize=14); ax_in.tick_params(labelsize=11)
    ax_in.set_title(r"$u=(\log M_\star,\ \log L_{H\alpha})$", fontsize=14)

    # ================= Targets =================
    ax_spec = fig.add_axes([cards["TG"][0] + 0.024, 0.166, cards["TG"][2] - 0.040, 0.205])
    draw_spectrum(ax_spec, di)
    fig.text(cx_IN, 0.385, r"$X=\{\log_{10}(F_{\rm line}/F_{H\alpha})\}$", ha="center", va="bottom", fontsize=14)

    # ================= Conditional density estimator (middle) =================
    # simple base density  ->  flow (text on card)  ->  complex learned line-ratio density
    ax_bz = fig.add_axes([cx_MD - 0.047, 0.630, 0.094, 0.188])
    gx = np.linspace(-3, 3, 140); GX, GY = np.meshgrid(gx, gx); Z = np.exp(-(GX**2 + GY**2) / 2)
    ax_bz.contourf(GX, GY, Z, levels=np.linspace(Z.max() * 0.06, Z.max(), 8), cmap=DENSE)
    ax_bz.set_facecolor("white"); ax_bz.set_xticks([]); ax_bz.set_yticks([])
    for s in ax_bz.spines.values():
        s.set_color("0.6"); s.set_linewidth(0.8)
    fig.text(cx_MD, 0.828, r"base density  $z\sim\mathcal{N}(0,\mathbb{I})$", ha="center", va="bottom", fontsize=15)
    fig.add_artist(FancyArrowPatch((cx_MD, 0.625), (cx_MD, 0.548), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=18, lw=2.2, color=TXT, zorder=6))
    # flow: plain text on the pink card (no box)
    fig.text(cx_MD, 0.518, "block neural autoregressive", ha="center", va="center", fontsize=15, zorder=6)
    fig.text(cx_MD, 0.476, "flow  " + r"$f(X;\,u)$", ha="center", va="center", fontsize=16, zorder=6)
    fig.add_artist(FancyArrowPatch((cx_MD, 0.445), (cx_MD, 0.368), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=18, lw=2.2, color=TXT, zorder=6))
    # learned conditional density (complex)
    ax_tx = fig.add_axes([cx_MD - 0.047, 0.178, 0.094, 0.188]); mini_density(ax_tx, nf_ratios[:, i_oii], nf_ratios[:, i_sii])
    fig.text(cx_MD, 0.170, r"learned density  $p(X\,|\,u)$", ha="center", va="top", fontsize=15)

    # ================= pipeline arrows between stages =================
    fig.add_artist(FancyArrowPatch((0.254, 0.500), (0.316, 0.500), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=26, lw=3.0, color=TXT, zorder=6))
    fig.text(0.285, 0.522, r"$(u,X)$", ha="center", va="bottom", fontsize=16, color=TXT)
    fig.add_artist(FancyArrowPatch((0.534, 0.500), (0.596, 0.500), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=26, lw=3.0, color=TXT, zorder=6))
    fig.text(0.565, 0.520, "sample &\nevaluate", ha="center", va="bottom", fontsize=15, fontweight="bold", color=TXT)

    # ================= Predictive accuracy (H-beta) =================
    ins_h = 0.225
    ax_c = fig.add_axes(sqrect(cx_PA - 0.098, 0.626, ins_h))
    ax_x = fig.add_axes(sqrect(cx_PA + 0.098, 0.626, ins_h))
    rho_dd = sq_hexbin(ax_c, t_dd, p_dd); rho_ds = sq_hexbin(ax_x, t_ds, p_ds)
    for a, ttl, rho in [(ax_c, "In-survey (DESI)", rho_dd), (ax_x, r"Cross-survey (DESI$\to$SDSS)", rho_ds)]:
        a.set_title(ttl, fontsize=13, color=TXT)
        a.set_xlabel(r"$\log L_{{\rm H}\beta,\ \rm true}$", fontsize=13)
        a.text(0.06, 0.93, rf"$\rho={rho:.2f}$", transform=a.transAxes, fontsize=13, va="top",
               bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax_c.set_ylabel(r"$\log L_{{\rm H}\beta,\ \rm pred}$", fontsize=13)

    # ================= Joint structure (BPT) =================
    ax4 = fig.add_axes(sqrect(cx_JS, 0.158, 0.250))
    bpt_contours(ax4, d_nii_ha, d_oiii_hb, nf_nii_ha, nf_oiii_hb, (-1.7, 0.6), (-1.2, 1.4))
    ax4.set_xlabel(r"$\log$([N II]/H$\alpha$)", fontsize=14)
    ax4.set_ylabel(r"$\log$([O III]/H$\beta$)", fontsize=14)
    ax4.legend(handles=[Line2D([], [], color="black", lw=1.8, label="DESI data"),
                        Line2D([], [], color=NF_COL, lw=1.8, label="DESI-trained NF")],
               fontsize=12, loc="upper right", frameon=True, framealpha=0.9)

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, bbox_inches="tight")
        print("Wrote:", f"{OUT}.{ext}", flush=True)
    print(f"rho Hbeta in-survey={rho_dd:.3f}  cross={rho_ds:.3f}", flush=True)


if __name__ == "__main__":
    main()
