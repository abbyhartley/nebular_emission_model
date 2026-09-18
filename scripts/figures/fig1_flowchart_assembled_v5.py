"""
Figure 1 flow chart v5  -- ALT-B production model.

v5 changes (author feedback):
  - REMOVED the "Conditional model" panel (base density / learned joint / flow box).
  - Layout is now 2x2 of SEPARATE cards:
        Inputs (top-left)   Targets (bottom-left)   ->  arrow  ->
        Predictive accuracy (top-right)   Joint structure (bottom-right)
    all UNNUMBERED. The single central arrow is labelled
    "Conditional density estimator  p(X|u)".
  - Joint-structure BPT: legend moved to upper-right (off the data); NF contours
    recoloured to a readable purple.
  - Targets spectrum: removed "joint distribution" label; darkened lines;
    prominent shaded envelope on each emission line (10-90% range) to show how
    the model generalises across galaxy spectra.
  - Predictive-accuracy panel unchanged (H-beta, best case).
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
OUT = REPO + "figs_ALTB/fig1_flowchart_assembled_v5"

FLUX_SCALE = 1e-17
SEED = 0
N_INSET = 60_000
N_CORR = 60_000
N_MC = 20

_ROBUST_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))

CARD = "#FFDFD9"
CARD_EDGE = "#e9a99e"
TXT = "black"
NF_COL = "#7d2fbf"                      # DESI-trained NF contour colour (readable purple)
SPEC_LINE = cmr.bubblegum(0.06)         # dark navy spectrum line
SPEC_BAND = "#9b59c9"                   # purple shaded envelope on emission lines
DENSE = LinearSegmentedColormap.from_list(
    "bubblegum_dense", ["white", cmr.bubblegum(0.85), cmr.bubblegum(0.6),
                        cmr.bubblegum(0.35), cmr.bubblegum(0.0)])
LINE_11_KW = dict(color="black", lw=1.8, ls=":", alpha=0.95)
FIGW, FIGH = 15.5, 8.6
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
    ax.plot(xk, 0.61 / (xk - 0.05) + 1.3, ls="--", color="0.4", lw=1.3)      # Kauffmann 2003
    xk2 = np.linspace(-1.9, 0.30, 200)
    ax.plot(xk2, 0.61 / (xk2 - 0.47) + 1.19, ls=":", color="0.4", lw=1.3)    # Kewley 2001

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
    ax.fill_between(wl, cont + lo, cont + hi, color=SPEC_BAND, alpha=0.55, lw=0, zorder=2)   # 10-90% envelope
    ax.plot(wl, cont + hi, color=SPEC_BAND, lw=0.7, alpha=0.8, zorder=2)
    ax.plot(wl, cont + mid, color=SPEC_LINE, lw=1.4, zorder=3)                                # median spectrum
    ax.plot(wl, cont, color="0.45", lw=0.9, zorder=1)
    off = {"[O II]": (0, 2), r"H$\gamma$": (0, 2), r"H$\beta$": (-9, 2), "[O III]": (11, 2),
           r"H$\alpha$": (0, 3), "[N II]": (-12, 15), "[S II]": (13, 2)}
    for lab, (lam, h) in peaks.items():
        dx, dy = off.get(lab, (0, 3))
        ax.annotate(lab, (lam, cont[np.argmin(abs(wl - lam))] + h), xytext=(dx, dy),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=TXT)
    ymax = 1.14 * float(np.max(cont + hi))
    ax.set_xlim(3650, 6850); ax.set_ylim(0, ymax)
    ax.set_yticks([]); ax.set_xlabel(r"rest wavelength [$\AA$]", fontsize=12); ax.tick_params(labelsize=10)


# ----------------------------------------------------------------------
def main():
    Path(REPO + "figs_ALTB").mkdir(exist_ok=True)
    df_d = add_loglha(thin_df(load_scalar_df(DESI_FITS), max(N_INSET, N_CORR), seed=SEED), survey="desi")
    df_s = add_loglha(thin_df(load_scalar_df(SDSS_FITS), N_INSET, seed=SEED + 1), survey="sdss")
    meta_d = pickle.load(open(META_DESI, "rb"))
    flow_d = load_flow(FLOW_DESI, meta_d)
    i_hb = outcol_index(meta_d, ["H_BETA", "HBETA"])
    i_nii = outcol_index(meta_d, ["NII_6584"]); i_oiii = outcol_index(meta_d, ["OIII_5007"])

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

    # (x0, y0, w, h, title)
    cards = {
        "IN": (0.030, 0.550, 0.320, 0.360, "Inputs"),
        "TG": (0.030, 0.070, 0.320, 0.360, "Targets"),
        "PA": (0.560, 0.550, 0.410, 0.360, "Predictive accuracy"),
        "JS": (0.560, 0.070, 0.410, 0.360, "Joint structure"),
    }
    for x0, y0, w, h, title in cards.values():
        fig.add_artist(FancyBboxPatch((x0, y0), w, h,
                       boxstyle="round,pad=0.006,rounding_size=0.010",
                       transform=fig.transFigure, facecolor=CARD, edgecolor=CARD_EDGE, lw=1.4, zorder=0))
        fig.text(x0 + w / 2, y0 + h + 0.014, title, ha="center", va="bottom",
                 fontsize=21, fontweight="bold", color=TXT)
    cx_IN = cards["IN"][0] + cards["IN"][2] / 2
    cx_PA = cards["PA"][0] + cards["PA"][2] / 2
    cx_JS = cards["JS"][0] + cards["JS"][2] / 2

    # ---- Inputs ----
    ax_in = fig.add_axes(sqrect(cx_IN, 0.610, 0.245))
    lm = di[meta_d["resolved"]["logmstar_col"]].to_numpy(float); ll = di["LOG_LHA"].to_numpy(float)
    good = np.isfinite(lm) & np.isfinite(ll)
    ax_in.hexbin(lm[good], ll[good], gridsize=44, bins="log", mincnt=3, cmap=DENSE)
    ax_in.set_facecolor("white")
    ax_in.set_xlabel(r"$\log M_\star\ [M_\odot]$", fontsize=13)
    ax_in.set_ylabel(r"$\log L_{H\alpha}$", fontsize=13); ax_in.tick_params(labelsize=10)
    ax_in.set_title(r"$u=(\log M_\star,\ \log L_{H\alpha})$", fontsize=13)

    # ---- Targets (spectrum) ----
    ax_spec = fig.add_axes([cards["TG"][0] + 0.026, 0.150, cards["TG"][2] - 0.046, 0.210])
    draw_spectrum(ax_spec, di)
    fig.text(cx_IN, 0.372, r"$X=\{\log_{10}(F_{\rm line}/F_{H\alpha})\}$", ha="center", va="bottom", fontsize=13)

    # ---- central arrow: conditional density estimator ----
    fig.text(0.455, 0.520, "Conditional density estimator", ha="center", va="bottom",
             fontsize=13.5, fontweight="bold", color=TXT)
    fig.add_artist(FancyArrowPatch((0.360, 0.500), (0.548, 0.500), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=32, lw=3.4, color=TXT, zorder=6))
    fig.text(0.455, 0.484, r"$p(X\,|\,u)$", ha="center", va="top", fontsize=15, color=TXT)

    # ---- Predictive accuracy (H-beta) ----
    ins_h = 0.235
    ax_c = fig.add_axes(sqrect(cx_PA - 0.098, 0.605, ins_h))
    ax_x = fig.add_axes(sqrect(cx_PA + 0.098, 0.605, ins_h))
    rho_dd = sq_hexbin(ax_c, t_dd, p_dd); rho_ds = sq_hexbin(ax_x, t_ds, p_ds)
    for a, ttl, rho in [(ax_c, "In-survey (DESI)", rho_dd), (ax_x, r"Cross-survey (DESI$\to$SDSS)", rho_ds)]:
        a.set_title(ttl, fontsize=13, color=TXT)
        a.set_xlabel(r"$\log L_{{\rm H}\beta,\ \rm true}$", fontsize=12)
        a.text(0.06, 0.93, rf"$\rho={rho:.2f}$", transform=a.transAxes, fontsize=13, va="top",
               bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax_c.set_ylabel(r"$\log L_{{\rm H}\beta,\ \rm pred}$", fontsize=12)

    # ---- Joint structure (BPT contour comparison) ----
    ax4 = fig.add_axes(sqrect(cx_JS, 0.120, 0.270))
    bpt_contours(ax4, d_nii_ha, d_oiii_hb, nf_nii_ha, nf_oiii_hb, (-1.7, 0.6), (-1.2, 1.4))
    ax4.set_xlabel(r"$\log$([N II]/H$\alpha$)", fontsize=13)
    ax4.set_ylabel(r"$\log$([O III]/H$\beta$)", fontsize=13)
    ax4.legend(handles=[Line2D([], [], color="black", lw=1.8, label="DESI data"),
                        Line2D([], [], color=NF_COL, lw=1.8, label="DESI-trained NF")],
               fontsize=11, loc="upper right", frameon=True, framealpha=0.9)

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, bbox_inches="tight")
        print("Wrote:", f"{OUT}.{ext}", flush=True)
    print(f"rho Hbeta in-survey={rho_dd:.3f}  cross={rho_ds:.3f}", flush=True)


if __name__ == "__main__":
    main()
