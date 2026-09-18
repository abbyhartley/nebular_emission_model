"""
Figure 1 (redesigned flow chart) v2 — three clearly separated cards with gaps:
  1. Inputs   ->   2. Conditional model   ->   3. Validation

Changes from v1 (referee + author feedback):
  - three separated cards with real spacing so the inter-panel arrows read clearly
  - larger fonts (>=14-15)
  - square (logM*, logLHa) panel
  - title-case headers (not all caps)
  - "8 lines: a, b, c, ..." comma-separated (no bullets)
  - validation joint-structure panel = two side-by-side DESI matrices (data | NF),
    the bottom row of correlation_matrices.png, so similarity is immediate
  - model schematic redesigned: clean base Gaussian density -> conditional-flow box
    (u enters the BOX) -> real line-ratio density (recognizable BPT-shaped cloud)
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
from scipy.ndimage import gaussian_filter
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
OUT = REPO + "figs/fig1_flowchart_assembled_v2"

FLUX_SCALE = 1e-17
SEED = 0
N_INSET = 150_000
N_CORR = 80_000
N_MC = 30

CMAP = cmr.bubblegum
CARD = "#f7ecf3"
CARD_EDGE = "#d9b8d0"
ACCENT = "#8e44ad"
LINE_11_KW = dict(color="black", lw=1.8, ls=":", alpha=0.95)
LABELS = [r"H$\beta$", r"H$\gamma$", r"[N II]", r"[S II]$_{6716}$",
          r"[S II]$_{6731}$", r"[O II]$_{3726}$", r"[O II]$_{3729}$", r"[O III]"]
I_OIII = 7
FIGW, FIGH = 21.0, 8.4
SQ = FIGH / FIGW    # width_frac = SQ * height_frac  -> display-square axes

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
    lo = np.min(np.concatenate([x, y])); hi = np.max(np.concatenate([x, y]))
    pad = 0.02 * (hi - lo); lo -= pad; hi += pad
    ax.hexbin(x, y, gridsize=gridsize, extent=(lo, hi, lo, hi), bins="log", mincnt=3, cmap=CMAP)
    ax.plot([lo, hi], [lo, hi], **LINE_11_KW)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    return spearmanr(x, y).correlation

def filled_density(ax, x, y, xr, yr, nbin=70, smooth=2.0, levels=10):
    m = np.isfinite(x) & np.isfinite(y)
    H, xe, ye = np.histogram2d(x[m], y[m], bins=nbin, range=[xr, yr])
    H = gaussian_filter(H.T, smooth)
    Xc = 0.5 * (xe[:-1] + xe[1:]); Yc = 0.5 * (ye[:-1] + ye[1:])
    ax.contourf(Xc, Yc, H, levels=levels, cmap=CMAP)

# ----------------------------------------------------------------------
def main():
    Path(REPO + "figs").mkdir(exist_ok=True)
    df_d = add_loglha(thin_df(load_scalar_df(DESI_FITS), max(N_INSET, N_CORR), seed=SEED), survey="desi")
    df_s = add_loglha(thin_df(load_scalar_df(SDSS_FITS), N_INSET, seed=SEED + 1), survey="sdss")
    meta_d = pickle.load(open(META_DESI, "rb")); meta_s = pickle.load(open(META_SDSS, "rb"))
    flow_d = load_flow(FLOW_DESI, meta_d); flow_s = load_flow(FLOW_SDSS, meta_s)

    # correlation matrices (DESI): data vs single-draw NF
    df_dc = df_d.iloc[:N_CORR].reset_index(drop=True)
    C_data = np.corrcoef(data_log_ratios(df_dc, meta_d), rowvar=False)
    _, nf_ratios = sample_single(flow_d, meta_d, df_dc, seed=SEED + 10)
    C_nf = np.corrcoef(nf_ratios, rowvar=False)

    # pred-vs-true [OIII]: in-survey (DESI) + cross (DESI->SDSS)
    di = df_d.iloc[:N_INSET].reset_index(drop=True)
    m_d, r_dd = sample_mcmean(flow_d, meta_d, di, seed=SEED + 3)
    di = di.loc[m_d].reset_index(drop=True)
    t_dd = log10_lum(di["Z"].to_numpy(float), di["OIII_5007_FLUX"].to_numpy(float))
    p_dd = di["LOG_LHA"].to_numpy(float) + r_dd[:, I_OIII].astype(float)
    m_s, r_ds = sample_mcmean(flow_d, meta_d, df_s, seed=SEED + 4)
    ds = df_s.loc[m_s].reset_index(drop=True)
    t_ds = log10_lum(ds["Z_1"].to_numpy(float), ds["OIII_5007_FLUX"].to_numpy(float))
    p_ds = ds["LOG_LHA"].to_numpy(float) + r_ds[:, I_OIII].astype(float)

    # real line-ratio density for the model panel (DESI BPT: [NII]/Ha vs [OIII]/Hb)
    nii = di["NII_6584_FLUX"].to_numpy(float); ha = di["HALPHA_FLUX"].to_numpy(float)
    oiii = di["OIII_5007_FLUX"].to_numpy(float); hb = di["HBETA_FLUX"].to_numpy(float)
    gg = (nii > 0) & (ha > 0) & (oiii > 0) & (hb > 0)
    nii_ha = np.log10(nii[gg] / ha[gg]); oiii_hb = np.log10(oiii[gg] / hb[gg])

    # ------------------------------------------------------------------
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"axes.labelsize": 16, "axes.titlesize": 16,
                         "xtick.labelsize": 12.5, "ytick.labelsize": 12.5})
    fig = plt.figure(figsize=(FIGW, FIGH))

    def sqrect(cx, y, h):
        w = SQ * h
        return [cx - w / 2, y, w, h]

    # --- section cards + headers ---
    cards = {
        "P1": (0.020, 0.135, 0.250, 0.74, "1.  Inputs"),
        "P2": (0.320, 0.135, 0.250, 0.74, "2.  Conditional model"),
        "P3": (0.620, 0.135, 0.365, 0.74, "3.  Validation"),
    }
    for x0, y0, w, h, title in cards.values():
        fig.add_artist(FancyBboxPatch((x0, y0), w, h,
                       boxstyle="round,pad=0.006,rounding_size=0.010",
                       transform=fig.transFigure, facecolor=CARD, edgecolor=CARD_EDGE, lw=1.4, zorder=0))
        fig.text(x0 + w / 2, y0 + h + 0.022, title, ha="center", va="bottom",
                 fontsize=18, fontweight="bold", color=ACCENT)
    c1 = cards["P1"][0] + cards["P1"][2] / 2
    c2 = cards["P2"][0] + cards["P2"][2] / 2
    c3 = cards["P3"][0] + cards["P3"][2] / 2

    # ================= Panel 1: Inputs =================
    ax_in = fig.add_axes(sqrect(c1, 0.42, 0.40))
    lm = di[meta_d["resolved"]["logmstar_col"]].to_numpy(float); ll = di["LOG_LHA"].to_numpy(float)
    good = np.isfinite(lm) & np.isfinite(ll)
    ax_in.hexbin(lm[good], ll[good], gridsize=46, bins="log", mincnt=3, cmap=CMAP)
    ax_in.set_xlabel(r"$\log M_\star\ [M_\odot]$"); ax_in.set_ylabel(r"$\log L_{H\alpha}\ [\mathrm{erg\,s^{-1}}]$")
    ax_in.set_title(r"Conditioning inputs  $u=(\log M_\star,\ \log L_{H\alpha})$", fontsize=14)
    fig.text(c1, 0.315, r"Targets  $X=\{\log_{10}(F_{\rm line}/F_{H\alpha})\}$",
             ha="center", va="top", fontsize=15)
    fig.text(c1, 0.265,
             "8 lines:  H$\\beta$,  H$\\gamma$,  [N II]$\\lambda$6584,  [S II]$\\lambda$6716,\n"
             "[S II]$\\lambda$6731,  [O II]$\\lambda$3726,  [O II]$\\lambda$3729,  [O III]$\\lambda$5007",
             ha="center", va="top", fontsize=13.5)

    # ================= Panel 2: Conditional model =================
    # base Gaussian density (top)
    ax_base = fig.add_axes(sqrect(c2, 0.635, 0.215)); ax_base.axis("off")
    gx = np.linspace(-3, 3, 140); GX, GY = np.meshgrid(gx, gx)
    ax_base.contourf(GX, GY, np.exp(-(GX**2 + GY**2) / 2), levels=8, cmap=CMAP)
    ax_base.set_title(r"base density  $\mathcal{N}(0,\mathbb{I})$", fontsize=14)
    # learned line-ratio density (bottom) — real DESI BPT cloud
    ax_learn = fig.add_axes(sqrect(c2, 0.155, 0.215))
    filled_density(ax_learn, nii_ha, oiii_hb, [-1.5, 0.0], [-1.0, 1.1])
    ax_learn.set_xlabel(r"$\log$([N II]/H$\alpha$)", fontsize=12); ax_learn.set_ylabel(r"$\log$([O III]/H$\beta$)", fontsize=12)
    ax_learn.tick_params(labelsize=9)
    ax_learn.set_title(r"learned joint  $p(X\,|\,u)$", fontsize=14)
    # conditional-flow box (middle)
    bw, bh = 0.150, 0.105
    bx0, by0 = c2 - bw / 2, 0.415
    fig.add_artist(FancyBboxPatch((bx0, by0), bw, bh, boxstyle="round,pad=0.004,rounding_size=0.012",
                   transform=fig.transFigure, facecolor="white", edgecolor=ACCENT, lw=2.2, zorder=3))
    fig.text(c2, by0 + bh / 2, "Conditional\nnormalizing flow\n" + r"$f(X;u)$",
             ha="center", va="center", fontsize=13.5, color=ACCENT, zorder=4)
    # u enters the BOX from the left (conditioning) — kept inside the card
    u_x0 = cards["P2"][0] + 0.014
    fig.add_artist(FancyArrowPatch((u_x0, by0 + bh / 2), (bx0, by0 + bh / 2),
                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=15, lw=2.0, color="black", zorder=4))
    fig.text((u_x0 + bx0) / 2, by0 + bh / 2 + 0.018, r"$u$", ha="center", va="bottom", fontsize=15, fontweight="bold")
    # vertical generative arrows: base -> box -> data
    fig.add_artist(FancyArrowPatch((c2, 0.632), (c2, by0 + bh + 0.004),
                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=20, lw=2.4, color=ACCENT, zorder=2))
    fig.add_artist(FancyArrowPatch((c2, by0 - 0.004), (c2, 0.375),
                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=20, lw=2.4, color=ACCENT, zorder=2))
    fig.text(c2 + 0.085, 0.46, r"invertible", ha="center", va="center", fontsize=11.5, color=ACCENT, rotation=90)

    # ================= Panel 3: Validation =================
    ax_c = fig.add_axes(sqrect(0.705, 0.545, 0.235)); ax_d = fig.add_axes(sqrect(0.885, 0.545, 0.235))
    rho_dd = sq_hexbin(ax_c, t_dd, p_dd); rho_ds = sq_hexbin(ax_d, t_ds, p_ds)
    for a, ttl, rho in [(ax_c, "In-survey (DESI)", rho_dd), (ax_d, r"Cross-survey (DESI$\to$SDSS)", rho_ds)]:
        a.set_title(ttl, fontsize=13)
        a.set_xlabel(r"$\log L_{\rm[OIII],\ true}$", fontsize=13)
        a.text(0.06, 0.93, rf"$\rho={rho:.2f}$", transform=a.transAxes, fontsize=13, va="top",
               bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax_c.set_ylabel(r"$\log L_{\rm[OIII],\ pred}$", fontsize=13)
    fig.text(0.795, 0.845, "Predictive accuracy", ha="center", va="bottom", fontsize=14, style="italic")

    # two side-by-side DESI correlation matrices (data | NF)
    n = len(LABELS); im = None
    mats = [(C_data, "DESI data", 0.700, True), (C_nf, "DESI-trained NF", 0.865, False)]
    for M, ttl, cx, yl in mats:
        axm = fig.add_axes(sqrect(cx, 0.150, 0.255))
        im = axm.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        axm.set_title(ttl, fontsize=13)
        axm.set_xticks(range(n)); axm.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8.5)
        axm.set_yticks(range(n))
        axm.set_yticklabels(LABELS if yl else [""] * n, fontsize=8.5)
        for i in range(n):
            for j in range(n):
                axm.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5.6,
                         color="white" if abs(M[i, j]) > 0.6 else "black")
    cax = fig.add_axes([0.965, 0.150, 0.008, 0.255])
    cb = fig.colorbar(im, cax=cax); cb.set_label(r"corr$\,[\log(L_{\rm line}/L_{H\alpha})]$", fontsize=11)
    fig.text(0.79, 0.44, "Joint structure: NF reproduces the data correlation matrix",
             ha="center", va="bottom", fontsize=14, style="italic")

    # ================= inter-panel arrows =================
    ymid = 0.50
    for xs, xe, lbl in [(0.270 + 0.004, 0.320 - 0.004, r"$p(X\,|\,u)$"),
                        (0.570 + 0.004, 0.620 - 0.004, "sample &\nevaluate")]:
        fig.add_artist(FancyArrowPatch((xs, ymid), (xe, ymid), transform=fig.transFigure,
                       arrowstyle="-|>", mutation_scale=30, lw=3.0, color=ACCENT, zorder=6))
        fig.text((xs + xe) / 2, ymid + 0.035, lbl, ha="center", va="bottom",
                 fontsize=13, color=ACCENT, fontweight="bold")

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, bbox_inches="tight")
        print("Wrote:", f"{OUT}.{ext}")
    print(f"rho in-survey={rho_dd:.3f}  cross={rho_ds:.3f}")


if __name__ == "__main__":
    main()
