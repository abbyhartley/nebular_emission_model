"""
Figure 1 (redesigned flow chart) v4  -- ALT-B production model.

v4 changes (author feedback):
  - ALT-B flows/data (nf_{desi,sdss}_ALTB) + robust BNAF inverter.
  - all text BLACK (no purple), flow box border BLACK, all arrows BLACK.
  - Panel 1 split vertically: TOP = conditioning inputs (logM* vs logL_Ha),
    BOTTOM = targets shown as a schematic galaxy spectrum with the 8 emission
    lines marked and data-informed shaded bands (16-84% ratio range); downward
    arrow inputs -> spectrum; "joint distribution" label.
  - Panel 2 base density + learned joint: dense = DARK on WHITE bg (inverted cmap);
    learned-joint BPT drawn with classic demarcation curves so the shape reads.
  - Panel 4 "Joint structure": corr matrices REPLACED by a BPT contour comparison
    (data vs NF), matching corner_ratios_NFs_and_data style.
  - Predictive-accuracy insets show H-beta (best case) in- and cross-survey.
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
FLOW_SDSS, META_SDSS = Path(REPO + "models/nf_sdss_ALTB.eqx"), Path(REPO + "models/nf_sdss_ALTB_meta.pkl")
FLOW_DESI, META_DESI = Path(REPO + "models/nf_desi_ALTB.eqx"), Path(REPO + "models/nf_desi_ALTB_meta.pkl")
OUT = REPO + "figs_ALTB/fig1_flowchart_assembled_v4"

FLUX_SCALE = 1e-17
SEED = 0
N_INSET = 60_000
N_CORR = 60_000
N_MC = 20

_ROBUST_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))

CARD = "#FFDFD9"
CARD_EDGE = "#e9a99e"
TXT = "black"                       # all text / arrows / box border
NF_PINK = cmr.bubblegum(0.96)       # NF contour colour (not text)
# white background at low density -> bubblegum inverted (down to its DARK navy end) at high density
DENSE = LinearSegmentedColormap.from_list(
    "bubblegum_dense", ["white", cmr.bubblegum(0.85), cmr.bubblegum(0.6),
                        cmr.bubblegum(0.35), cmr.bubblegum(0.0)])
LINE_11_KW = dict(color="black", lw=1.8, ls=":", alpha=0.95)
LABELS = [r"H$\beta$", r"H$\gamma$", r"[N II]", r"[S II]$_{6716}$",
          r"[S II]$_{6731}$", r"[O II]$_{3726}$", r"[O II]$_{3729}$", r"[O III]"]
FIGW, FIGH = 17.5, 8.6
SQ = FIGH / FIGW

# emission lines for the schematic spectrum: label -> (rest wavelength, DESI flux col or None)
SPEC_LINES = {
    "[O II]": (3727.0, ("OII_3726_FLUX", "OII_3729_FLUX")),
    r"H$\gamma$": (4340.0, "HGAMMA_FLUX"),
    r"H$\beta$": (4861.0, "HBETA_FLUX"),
    "[O III]": (5007.0, "OIII_5007_FLUX"),
    r"H$\alpha$": (6563.0, "HALPHA"),          # denominator, ratio = 1
    "[N II]": (6584.0, "NII_6584_FLUX"),
    "[S II]": (6725.0, ("SII_6716_FLUX", "SII_6731_FLUX")),
}


# ----------------------------------------------------------------------
def load_scalar_df(p):
    t = Table.read(p, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()

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

def data_log_ratios(df, meta):
    raw = [c[6:] if c.startswith("LOG10_") else c for c in meta["resolved"]["target_cols"]]
    ha = next((c for c in ["HALPHA_FLUX", "H_ALPHA_FLUX", "HA_FLUX"] if c in df.columns), None)
    F = np.column_stack([df[c].to_numpy(float) for c in raw]); fha = df[ha].to_numpy(float)
    g = np.all(F > 0, axis=1) & (fha > 0) & np.isfinite(fha) & np.all(np.isfinite(F), axis=1)
    return np.log10(F[g]) - np.log10(fha[g])[:, None]

def sq_hexbin(ax, x, y, gridsize=55):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    xy = np.concatenate([x, y]); q1, q3 = np.percentile(xy, [25, 75]); fe = 3.0 * (q3 - q1)
    kp = xy[(xy >= q1 - fe) & (xy <= q3 + fe)]; lo, hi = float(kp.min()), float(kp.max())
    pad = 0.02 * (hi - lo); lo -= pad; hi += pad
    ax.hexbin(x, y, gridsize=gridsize, extent=(lo, hi, lo, hi), bins="log", mincnt=3, cmap=DENSE)
    ax.plot([lo, hi], [lo, hi], **LINE_11_KW)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    return spearmanr(x, y).correlation

def filled_density(ax, x, y, xr, yr, nbin=80, smooth=2.0, levels=10):
    m = np.isfinite(x) & np.isfinite(y)
    H, xe, ye = np.histogram2d(x[m], y[m], bins=nbin, range=[xr, yr])
    H = gaussian_filter(H.T, smooth)
    Xc = 0.5 * (xe[:-1] + xe[1:]); Yc = 0.5 * (ye[:-1] + ye[1:])
    ax.contourf(Xc, Yc, H, levels=levels, cmap=DENSE)
    ax.set_facecolor("white")

def bpt_demarcation(ax):
    xk = np.linspace(-1.9, -0.02, 200)
    ax.plot(xk, 0.61 / (xk - 0.05) + 1.3, ls="--", color="0.35", lw=1.3)      # Kauffmann 2003
    xk2 = np.linspace(-1.9, 0.30, 200)
    ax.plot(xk2, 0.61 / (xk2 - 0.47) + 1.19, ls=":", color="0.35", lw=1.3)    # Kewley 2001

def bpt_contours(ax, xd, yd, xn, yn, xr, yr):
    for x, y, color in [(xd, yd, "black"), (xn, yn, NF_PINK)]:
        m = np.isfinite(x) & np.isfinite(y)
        H, xe, ye = np.histogram2d(x[m], y[m], bins=70, range=[xr, yr])
        H = gaussian_filter(H.T, 1.6)
        Xc = 0.5 * (xe[:-1] + xe[1:]); Yc = 0.5 * (ye[:-1] + ye[1:])
        lv = H.max() * np.array([0.08, 0.25, 0.55, 0.85])
        ax.contour(Xc, Yc, H, levels=lv, colors=[color], linewidths=1.6)
    bpt_demarcation(ax)
    ax.set_xlim(*xr); ax.set_ylim(*yr)

def ratio_stats(df, col_or_pair, ha):
    if col_or_pair == "HALPHA":
        return 1.0, 1.0, 1.0
    cols = col_or_pair if isinstance(col_or_pair, tuple) else (col_or_pair,)
    f = np.sum([df[c].to_numpy(float) for c in cols], axis=0)
    good = (f > 0) & (ha > 0)
    r = f[good] / ha[good]
    return tuple(np.percentile(r, [16, 50, 84]))

def draw_spectrum(ax, di):
    ha = di["HALPHA_FLUX"].to_numpy(float)
    wl = np.linspace(3650, 6850, 2400)
    cont = 0.06 + 0.004 * (wl - 3650) / 3200      # faint sloped continuum
    sig = 7.0
    lo = np.full_like(wl, 0.0); mid = np.zeros_like(wl); hi = np.zeros_like(wl)
    peaks = {}
    for lab, (lam, col) in SPEC_LINES.items():
        p16, p50, p84 = ratio_stats(di, col, ha)
        g = np.exp(-0.5 * ((wl - lam) / sig) ** 2)
        lo += p16 * g; mid += p50 * g; hi += p84 * g
        peaks[lab] = (lam, p50)
    ax.fill_between(wl, cont + lo, cont + hi, color=NF_PINK, alpha=0.35, lw=0, zorder=2)
    ax.plot(wl, cont + mid, color=cmr.bubblegum(1.0), lw=1.1, zorder=3)
    ax.plot(wl, cont, color="0.5", lw=0.8, zorder=1)
    off = {"[O II]": (0, 2), r"H$\gamma$": (0, 2), r"H$\beta$": (-9, 2), "[O III]": (10, 2),
           r"H$\alpha$": (0, 3), "[N II]": (-12, 14), "[S II]": (12, 2)}
    for lab, (lam, h) in peaks.items():
        dx, dy = off.get(lab, (0, 3))
        ax.annotate(lab, (lam, cont[np.argmin(abs(wl - lam))] + h), xytext=(dx, dy),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=TXT)
    ax.set_xlim(3650, 6850); ax.set_ylim(0, 1.35)
    ax.set_yticks([]); ax.set_xlabel(r"rest wavelength [$\AA$]", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.text(0.03, 0.93, "joint distribution", transform=ax.transAxes, ha="left", va="top",
            fontsize=12, style="italic", color=TXT)


# ----------------------------------------------------------------------
def main():
    Path(REPO + "figs_ALTB").mkdir(exist_ok=True)
    df_d = add_loglha(thin_df(load_scalar_df(DESI_FITS), max(N_INSET, N_CORR), seed=SEED), survey="desi")
    df_s = add_loglha(thin_df(load_scalar_df(SDSS_FITS), N_INSET, seed=SEED + 1), survey="sdss")
    meta_d = pickle.load(open(META_DESI, "rb"))
    flow_d = load_flow(FLOW_DESI, meta_d)   # in- and cross-survey both use the DESI-trained flow
    i_hb = outcol_index(meta_d, ["H_BETA", "HBETA"])
    i_nii = outcol_index(meta_d, ["NII_6584"]); i_oiii = outcol_index(meta_d, ["OIII_5007"])

    # ---- BPT: data vs NF (DESI) ----
    df_dc = df_d.iloc[:N_CORR].reset_index(drop=True)
    m_bpt, nf_ratios = sample_single(flow_d, meta_d, df_dc, seed=SEED + 10)
    dfb = df_dc.loc[m_bpt].reset_index(drop=True)
    nii = dfb["NII_6584_FLUX"].to_numpy(float); ha = dfb["HALPHA_FLUX"].to_numpy(float)
    oiii = dfb["OIII_5007_FLUX"].to_numpy(float); hb = dfb["HBETA_FLUX"].to_numpy(float)
    gg = (nii > 0) & (ha > 0) & (oiii > 0) & (hb > 0)
    d_nii_ha = np.log10(nii[gg] / ha[gg]); d_oiii_hb = np.log10(oiii[gg] / hb[gg])
    nf_nii_ha = nf_ratios[:, i_nii]; nf_oiii_hb = nf_ratios[:, i_oiii] - nf_ratios[:, i_hb]

    # ---- predictive accuracy (H-beta): in-survey + cross-survey ----
    di = df_d.iloc[:N_INSET].reset_index(drop=True)
    m_d, r_dd = sample_mcmean(flow_d, meta_d, di, seed=SEED + 3)
    di = di.loc[m_d].reset_index(drop=True)
    t_dd = log10_lum(di["Z"].to_numpy(float), di["HBETA_FLUX"].to_numpy(float))
    p_dd = di["LOG_LHA"].to_numpy(float) + r_dd[:, i_hb].astype(float)
    m_s, r_ds = sample_mcmean(flow_d, meta_d, df_s, seed=SEED + 4)
    ds = df_s.loc[m_s].reset_index(drop=True)
    t_ds = log10_lum(ds["Z_1"].to_numpy(float), ds["H_BETA_FLUX"].to_numpy(float))  # SDSS uses H_BETA_FLUX
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

    cards = {
        "P1": (0.020, 0.100, 0.310, 0.790, "1.  Data"),
        "P2": (0.400, 0.100, 0.155, 0.790, "2.  Conditional model"),
        "P3": (0.625, 0.545, 0.345, 0.345, "3.  Predictive accuracy"),
        "P4": (0.625, 0.100, 0.345, 0.360, "4.  Joint structure"),
    }
    for x0, y0, w, h, title in cards.values():
        fig.add_artist(FancyBboxPatch((x0, y0), w, h,
                       boxstyle="round,pad=0.006,rounding_size=0.010",
                       transform=fig.transFigure, facecolor=CARD, edgecolor=CARD_EDGE, lw=1.4, zorder=0))
        fig.text(x0 + w / 2, y0 + h + 0.012, title, ha="center", va="bottom",
                 fontsize=20, fontweight="bold", color=TXT)
    c1 = cards["P1"][0] + cards["P1"][2] / 2
    c2 = cards["P2"][0] + cards["P2"][2] / 2

    # ================= Panel 1 (top): conditioning inputs =================
    ax_in = fig.add_axes(sqrect(c1, 0.560, 0.270))
    lm = di[meta_d["resolved"]["logmstar_col"]].to_numpy(float); ll = di["LOG_LHA"].to_numpy(float)
    good = np.isfinite(lm) & np.isfinite(ll)
    ax_in.hexbin(lm[good], ll[good], gridsize=44, bins="log", mincnt=3, cmap=DENSE)
    ax_in.set_facecolor("white")
    ax_in.set_xlabel(r"$\log M_\star\ [M_\odot]$", fontsize=13)
    ax_in.set_ylabel(r"$\log L_{H\alpha}$", fontsize=13); ax_in.tick_params(labelsize=10)
    fig.text(c1, 0.850, r"inputs  $u=(\log M_\star,\ \log L_{H\alpha})$", ha="center", va="bottom", fontsize=14)

    # arrow inputs -> targets
    fig.add_artist(FancyArrowPatch((c1, 0.548), (c1, 0.470), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=20, lw=2.4, color=TXT, zorder=5))

    # ================= Panel 1 (bottom): targets spectrum =================
    ax_spec = fig.add_axes([cards["P1"][0] + 0.028, 0.150, cards["P1"][2] - 0.050, 0.270])
    draw_spectrum(ax_spec, di)
    fig.text(c1, 0.455, r"targets  $X=\{\log_{10}(F_{\rm line}/F_{H\alpha})\}$",
             ha="center", va="bottom", fontsize=14)

    # ================= Panel 2: conditional model =================
    base_h = 0.185; box_w = SQ * base_h
    ax_base = fig.add_axes(sqrect(c2, 0.650, base_h)); ax_base.axis("off")
    gx = np.linspace(-3, 3, 160); GX, GY = np.meshgrid(gx, gx)
    ax_base.contourf(GX, GY, np.exp(-(GX**2 + GY**2) / 2), levels=8, cmap=DENSE)
    ax_base.set_title(r"base density  $\mathcal{N}(0,\mathbb{I})$", fontsize=14, color=TXT)
    ax_learn = fig.add_axes(sqrect(c2, 0.230, base_h))
    filled_density(ax_learn, d_nii_ha, d_oiii_hb, [-1.7, 0.55], [-1.2, 1.35])
    bpt_demarcation(ax_learn)
    ax_learn.set_xlim(-1.7, 0.55); ax_learn.set_ylim(-1.2, 1.35)
    ax_learn.set_xlabel(r"$\log$([N II]/H$\alpha$)", fontsize=12)
    ax_learn.set_ylabel(r"$\log$([O III]/H$\beta$)", fontsize=12); ax_learn.tick_params(labelsize=10)
    fig.text(c2, 0.175, r"learned joint  $p(X\,|\,u)$", ha="center", va="top", fontsize=14)
    bh = 0.115; bx0, by0 = c2 - box_w / 2, 0.462
    fig.add_artist(FancyBboxPatch((bx0, by0), box_w, bh, boxstyle="round,pad=0.004,rounding_size=0.012",
                   transform=fig.transFigure, facecolor="white", edgecolor=TXT, lw=2.2, zorder=3))
    fig.text(c2, by0 + bh / 2, "Conditional\nnormalizing flow\n" + r"$f(X;u)$",
             ha="center", va="center", fontsize=12, color=TXT, zorder=4)
    fig.add_artist(FancyArrowPatch((c2, 0.645), (c2, by0 + bh + 0.004), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=22, lw=2.6, color=TXT, zorder=2))
    fig.add_artist(FancyArrowPatch((c2, by0 - 0.004), (c2, 0.418), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=22, lw=2.6, color=TXT, zorder=2))

    # ================= Panel 3: predictive accuracy (H-beta) =================
    ins_h = 0.240
    ax_c = fig.add_axes(sqrect(0.730, 0.605, ins_h)); ax_dd = fig.add_axes(sqrect(0.865, 0.605, ins_h))
    rho_dd = sq_hexbin(ax_c, t_dd, p_dd); rho_ds = sq_hexbin(ax_dd, t_ds, p_ds)
    for a, ttl, rho in [(ax_c, "In-survey (DESI)", rho_dd), (ax_dd, r"Cross-survey (DESI$\to$SDSS)", rho_ds)]:
        a.set_title(ttl, fontsize=13, color=TXT)
        a.set_xlabel(r"$\log L_{{\rm H}\beta,\ \rm true}$", fontsize=12)
        a.text(0.06, 0.93, rf"$\rho={rho:.2f}$", transform=a.transAxes, fontsize=13, va="top",
               bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax_c.set_ylabel(r"$\log L_{{\rm H}\beta,\ \rm pred}$", fontsize=12)

    # ================= Panel 4: joint structure (BPT contour comparison) =================
    ax4 = fig.add_axes(sqrect(0.797, 0.150, 0.275))
    bpt_contours(ax4, d_nii_ha, d_oiii_hb, nf_nii_ha, nf_oiii_hb, (-1.7, 0.6), (-1.2, 1.4))
    ax4.set_xlabel(r"$\log$([N II]/H$\alpha$)", fontsize=13)
    ax4.set_ylabel(r"$\log$([O III]/H$\beta$)", fontsize=13)
    ax4.legend(handles=[Line2D([], [], color="black", lw=1.8, label="DESI data"),
                        Line2D([], [], color=NF_PINK, lw=1.8, label="DESI-trained NF")],
               fontsize=11, loc="lower left", frameon=True)

    # ================= inter-panel arrows (black, in the white gaps) =================
    for xs, xe, ym, lbl, fs in [(0.335, 0.398, 0.500, r"$\mathbf{p(X\,|\,u)}$", 16),
                                (0.560, 0.620, 0.500, "sample &\nevaluate", 13)]:
        fig.add_artist(FancyArrowPatch((xs, ym), (xe, ym), transform=fig.transFigure,
                       arrowstyle="-|>", mutation_scale=28, lw=3.0, color=TXT, zorder=6))
        fig.text((xs + xe) / 2, ym + 0.028, lbl, ha="center", va="bottom",
                 fontsize=fs, color=TXT, fontweight="bold")

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, bbox_inches="tight")
        print("Wrote:", f"{OUT}.{ext}", flush=True)
    print(f"rho Hbeta in-survey={rho_dd:.3f}  cross={rho_ds:.3f}", flush=True)


if __name__ == "__main__":
    main()
