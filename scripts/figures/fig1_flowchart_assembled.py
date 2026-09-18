"""
Figure 1 (redesigned flow chart) — single assembled figure with three labeled
sections:  (1) INPUTS  ->  (2) CONDITIONAL MODEL  ->  (3) VALIDATION.

Addresses referee feedback:
  - conditioning is explicit: inputs u=(logM*, logLHa) are visually separated from
    the 8 targets X, the section arrow is labeled p(X|u), and u re-enters the flow.
  - three clear sections with header bands / tinted cards.
  - the two pred-vs-true panels are combined into a small accuracy strip (in-survey
    + cross-survey), and a NEW correlation-matrix panel (data vs NF, split triangle)
    foregrounds the JOINT/covariance nature of the model.

Panels: IN = logM*-logLHa density (DESI); TGT = 8-line target list; MOD = flow
schematic; C = in-survey [OIII] pred-vs-true (DESI); D = cross-survey [OIII]
(DESI->SDSS); COR = 8x8 correlation matrix, DESI data (lower) vs NF (upper).
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
import scienceplots  # noqa: F401
import cmasher as cmr

from scipy.stats import spearmanr
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow

# ----------------------------------------------------------------------
BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
SDSS_FITS = Path(BASE + "SDSS_main_training_data.fits")
DESI_FITS = Path(BASE + "DESI_BGS_training_data.fits")
FLOW_SDSS, META_SDSS = Path(REPO + "nf_sdss_main.eqx"), Path(REPO + "nf_sdss_main_meta.pkl")
FLOW_DESI, META_DESI = Path(REPO + "nf_desi_bgs.eqx"), Path(REPO + "nf_desi_bgs_meta.pkl")
OUT = REPO + "figs/fig1_flowchart_assembled"

FLUX_SCALE = 1e-17
SEED = 0
N_INSET = 150_000
N_CORR = 80_000
N_MC = 30

CMAP = cmr.bubblegum
CARD = "#f7ecf3"          # pale bubblegum-pink section card
CARD_EDGE = "#d9b8d0"
ACCENT = "#8e44ad"        # purple accent (matches slides)
LINE_11_KW = dict(color="black", lw=1.8, ls=":", alpha=0.95)
LABELS = [r"H$\beta$", r"H$\gamma$", r"[N II]", r"[S II]$_{6716}$",
          r"[S II]$_{6731}$", r"[O II]$_{3726}$", r"[O II]$_{3729}$", r"[O III]"]
I_OIII = 7

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
        key=jr.key(int(meta.get("seed", 0))), base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    return eqx.tree_deserialise_leaves(flow_path, template)

def cond_U(meta, df):
    lm, ll = meta["resolved"]["logmstar_col"], meta["resolved"]["loglha_col"]
    m = np.isfinite(df[lm].to_numpy(float)) & np.isfinite(df[ll].to_numpy(float))
    U = (df.loc[m, [lm, ll]].to_numpy(np.float32) - meta["U_mean"]) / meta["U_std"]
    return m, jnp.asarray(U)

def sample_single(flow, meta, df, *, seed):
    """One draw per galaxy -> log ratios (N,8). For correlation structure."""
    m, U = cond_U(meta, df)
    keys = jr.split(jr.key(seed + 1234), U.shape[0])
    Xn = jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, U)
    return m, np.array(Xn) * meta["X_std"] + meta["X_mean"]

def sample_mcmean(flow, meta, df, *, seed, n_mc=N_MC):
    """MC-mean log ratios (N,8). For pred-vs-true point estimate."""
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
    lo = np.min(np.concatenate([x, y])); hi = np.max(np.concatenate([x, y]))
    pad = 0.02 * (hi - lo); lo -= pad; hi += pad
    ax.hexbin(x, y, gridsize=gridsize, extent=(lo, hi, lo, hi), bins="log", mincnt=3, cmap=CMAP)
    ax.plot([lo, hi], [lo, hi], **LINE_11_KW)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
    return spearmanr(x, y).correlation

# ----------------------------------------------------------------------
def draw_schematic(ax):
    """Vertical: base Gaussian (top) -> invertible transforms (u enters) -> warped joint (bottom)."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    rng = np.random.default_rng(1)
    g1, g2 = rng.normal(size=2500), rng.normal(size=2500)
    # base blob (top)
    bx, by = 5.0 + 0.9 * g1, 8.05 + 0.9 * g2
    ax.hexbin(bx, by, gridsize=24, cmap=CMAP, mincnt=1, extent=(2.6, 7.4, 6.7, 9.4))
    ax.text(5.0, 9.62, r"base density  $\mathcal{N}(0,\mathbb{I})$", ha="center", va="bottom", fontsize=12.5)
    # warped blob (bottom): banana map
    wx = 5.0 + 1.15 * g1
    wy = 2.85 + 0.72 * (g2 + 0.6 * (g1**2 - 1.0))
    ax.hexbin(wx, wy, gridsize=26, cmap=CMAP, mincnt=1, extent=(2.1, 7.9, 1.4, 4.4))
    ax.text(5.0, 0.92, r"learned joint  $p(X\,|\,u)$", ha="center", va="top", fontsize=12.5)
    # vertical transform arrow (down, base -> warped)
    ax.annotate("", xy=(5.0, 4.55), xytext=(5.0, 6.55),
                arrowprops=dict(arrowstyle="-|>", lw=2.8, color=ACCENT, mutation_scale=24))
    ax.text(5.45, 5.05, r"invertible" + "\n" + r"transforms $f(X;u)$",
            ha="left", va="center", fontsize=11.5, color=ACCENT)
    # conditioning arrow (u enters the transform from the left, near the top)
    ax.annotate("", xy=(4.9, 6.05), xytext=(2.3, 6.05),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color="black", mutation_scale=16))
    ax.text(2.2, 6.3, r"condition on $u$", ha="left", va="bottom", fontsize=11)
    # log-density identity
    ax.text(5.0, 0.2, r"$\log p(X|u)=\log\mathcal{N}(f(X;u))+\log|\det\,\partial f/\partial X|$",
            ha="center", va="center", fontsize=9.8)

# ----------------------------------------------------------------------
def main():
    Path(REPO + "figs").mkdir(exist_ok=True)
    df_d = add_loglha(thin_df(load_scalar_df(DESI_FITS), max(N_INSET, N_CORR), seed=SEED), survey="desi")
    df_s = add_loglha(thin_df(load_scalar_df(SDSS_FITS), N_INSET, seed=SEED + 1), survey="sdss")
    meta_d = pickle.load(open(META_DESI, "rb")); meta_s = pickle.load(open(META_SDSS, "rb"))
    flow_d = load_flow(FLOW_DESI, meta_d); flow_s = load_flow(FLOW_SDSS, meta_s)

    # --- correlation matrices (DESI): data vs single-draw NF ---
    df_dc = df_d.iloc[:N_CORR].reset_index(drop=True)
    C_data = np.corrcoef(data_log_ratios(df_dc, meta_d), rowvar=False)
    m_nf, nf_ratios = sample_single(flow_d, meta_d, df_dc, seed=SEED + 10)
    C_nf = np.corrcoef(nf_ratios, rowvar=False)
    n = len(LABELS)
    M = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i > j:   M[i, j] = C_data[i, j]   # lower triangle = data
            elif i < j: M[i, j] = C_nf[i, j]     # upper triangle = NF

    # --- pred-vs-true [OIII]: in-survey (DESI) + cross (DESI->SDSS) ---
    di = df_d.iloc[:N_INSET].reset_index(drop=True)
    m_d, r_dd = sample_mcmean(flow_d, meta_d, di, seed=SEED + 3)
    di = di.loc[m_d].reset_index(drop=True)
    t_dd = log10_lum(di["Z"].to_numpy(float), di["OIII_5007_FLUX"].to_numpy(float))
    p_dd = di["LOG_LHA"].to_numpy(float) + r_dd[:, I_OIII].astype(float)

    m_s, r_ds = sample_mcmean(flow_d, meta_d, df_s, seed=SEED + 4)
    ds = df_s.loc[m_s].reset_index(drop=True)
    t_ds = log10_lum(ds["Z_1"].to_numpy(float), ds["OIII_5007_FLUX"].to_numpy(float))
    p_ds = ds["LOG_LHA"].to_numpy(float) + r_ds[:, I_OIII].astype(float)

    # ------------------------------------------------------------------
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"axes.labelsize": 14, "axes.titlesize": 14,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    mosaic = [["IN", "MOD", "C", "D"],
              ["IN", "MOD", "COR", "COR"],
              ["TGT", "MOD", "COR", "COR"]]
    fig, ax = plt.subplot_mosaic(
        mosaic, figsize=(18, 8.6),
        gridspec_kw=dict(width_ratios=[1.25, 1.55, 0.95, 0.95],
                         height_ratios=[1.0, 0.72, 0.5],
                         wspace=0.5, hspace=0.55))

    # (IN) logM* - logLHa density
    axin = ax["IN"]
    lm = di[meta_d["resolved"]["logmstar_col"]].to_numpy(float)
    ll = di["LOG_LHA"].to_numpy(float)
    good = np.isfinite(lm) & np.isfinite(ll)
    axin.hexbin(lm[good], ll[good], gridsize=48, bins="log", mincnt=3, cmap=CMAP)
    axin.set_xlabel(r"$\log M_\star\ [M_\odot]$"); axin.set_ylabel(r"$\log L_{H\alpha}\ [\mathrm{erg\,s^{-1}}]$")
    axin.set_title(r"Conditioning inputs $u$", fontsize=13)

    # (TGT) target list
    axt = ax["TGT"]; axt.axis("off")
    axt.text(0.5, 1.02, r"Targets  $X=\{\log_{10}(F_{\rm line}/F_{H\alpha})\}$",
             ha="center", va="top", fontsize=12.5, transform=axt.transAxes)
    lines = ["H$\\beta$", "H$\\gamma$", "[N II] $\\lambda$6584", "[S II] $\\lambda$6716",
             "[S II] $\\lambda$6731", "[O II] $\\lambda$3726", "[O II] $\\lambda$3729", "[O III] $\\lambda$5007"]
    txt = "   ".join([r"$\bullet$ " + s for s in lines[:4]]) + "\n" + \
          "   ".join([r"$\bullet$ " + s for s in lines[4:]])
    axt.text(0.5, 0.62, txt, ha="center", va="top", fontsize=10.8, transform=axt.transAxes)

    # (MOD) schematic
    draw_schematic(ax["MOD"])

    # (C) in-survey OIII
    rho_dd = sq_hexbin(ax["C"], t_dd, p_dd)
    ax["C"].set_title("In-survey (DESI)", fontsize=12)
    ax["C"].set_xlabel(r"$\log L_{\rm[OIII],\ true}$"); ax["C"].set_ylabel(r"$\log L_{\rm[OIII],\ pred}$")
    ax["C"].text(0.05, 0.93, rf"$\rho={rho_dd:.2f}$", transform=ax["C"].transAxes,
                 fontsize=11, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    # (D) cross-survey OIII
    rho_ds = sq_hexbin(ax["D"], t_ds, p_ds)
    ax["D"].set_title(r"Cross-survey (DESI$\to$SDSS)", fontsize=12)
    ax["D"].set_xlabel(r"$\log L_{\rm[OIII],\ true}$"); ax["D"].set_ylabel(r"$\log L_{\rm[OIII],\ pred}$")
    ax["D"].text(0.05, 0.93, rf"$\rho={rho_ds:.2f}$", transform=ax["D"].transAxes,
                 fontsize=11, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    # (COR) split-triangle correlation matrix
    axc = ax["COR"]
    im = axc.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    axc.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="k", lw=1.1)
    axc.set_xticks(range(n)); axc.set_yticks(range(n))
    axc.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=9)
    axc.set_yticklabels(LABELS, fontsize=9)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            axc.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.4,
                     color="white" if abs(M[i, j]) > 0.6 else "black")
    axc.text(0.30, 0.11, "data", transform=axc.transAxes, fontsize=12, style="italic", ha="center")
    axc.text(0.72, 0.90, "NF", transform=axc.transAxes, fontsize=12, style="italic", ha="center")
    axc.set_title(r"Joint structure: NF vs data correlation of the 8 ratios (DESI)", fontsize=12)
    cb = fig.colorbar(im, ax=axc, shrink=0.8, pad=0.02)
    cb.set_label(r"corr$\,[\log(L_{\rm line}/L_{H\alpha})]$", fontsize=10)

    # ------------------------------------------------------------------
    # section cards + headers + inter-section arrows (after layout settles)
    fig.canvas.draw()
    def union(keys):
        bs = [ax[k].get_position() for k in keys]
        x0 = min(b.x0 for b in bs); y0 = min(b.y0 for b in bs)
        x1 = max(b.x1 for b in bs); y1 = max(b.y1 for b in bs)
        return x0, y0, x1, y1
    sections = [(["IN", "TGT"], "1.  INPUTS"),
                (["MOD"], "2.  CONDITIONAL MODEL"),
                (["C", "D", "COR"], "3.  VALIDATION")]
    pad = 0.018
    boxes = []
    for keys, title in sections:
        x0, y0, x1, y1 = union(keys)
        bx0, by0 = x0 - pad - 0.028, y0 - pad - 0.03
        bx1, by1 = x1 + pad + 0.005, y1 + pad
        fig.add_artist(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0,
                       boxstyle="round,pad=0.008,rounding_size=0.012",
                       transform=fig.transFigure, facecolor=CARD, edgecolor=CARD_EDGE,
                       lw=1.3, zorder=0))
        fig.text((bx0 + bx1) / 2, by1 + 0.012, title, ha="center", va="bottom",
                 fontsize=15, fontweight="bold", color=ACCENT)
        boxes.append((bx0, by0, bx1, by1))
    ymid = (boxes[0][1] + boxes[0][3]) / 2
    for (a, b, lbl) in [(0, 1, r"$p(X\,|\,u)$"), (1, 2, "sample &\nevaluate")]:
        x_start, x_end = boxes[a][2], boxes[b][0]
        fig.add_artist(FancyArrowPatch((x_start, ymid), (x_end, ymid),
                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=26,
                       lw=2.6, color=ACCENT, zorder=5))
        fig.text((x_start + x_end) / 2, ymid + 0.03, lbl, ha="center", va="bottom",
                 fontsize=11.5, color=ACCENT, fontweight="bold")

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, bbox_inches="tight")
        print("Wrote:", f"{OUT}.{ext}")
    print(f"rho in-survey={rho_dd:.3f}  cross={rho_ds:.3f}")


if __name__ == "__main__":
    main()
