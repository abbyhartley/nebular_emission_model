#!/usr/bin/env python3
"""
hiz_lum_1to1 in LUMINOSITY space, but predicted with the METALLICITY-conditioned flow fed the
evolved-MZR metallicity (Zpred(M*,L_Ha) - 0.11*z, Sanders+2021), vs the matched baseline.
Shows how much [OII] improves and whether [OIII] is degraded by adding the metallicity conditioner.
Both flows trained on DESI-BGS ALT-B dropping [NII] (N2 defines Z); 7 targets. Test on DESI-COSMOS.
Each panel: +Z-evolved hexbin (obs vs pred log L), RMSE/scat/rho, and bias base->+Z.
"""
import numpy as np
from pathlib import Path
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
import jax, jax.numpy as jnp, jax.random as jr
import optax, equinox as eqx
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as _LSC
try:
    import scienceplots; plt.style.use(["science", "no-latex"])
except Exception:
    pass
import cmasher as cmr
plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 12, "axes.titlesize": 15, "figure.dpi": 130})

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = Path(BASE + "nebular_emission_model")
TRAIN_FITS = BASE + "DESI_BGS_training_data_ALTB.fits"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
FLUX_SCALE = 1e-17; MASS_ZP = 0.13
SEED = 0; EPOCHS = 200; BATCH = 4096; LR = 3e-4; CLIP = 1.0; NMC = 40
PP04_A, PP04_B = 8.90, 0.57; DZDOH = -0.11
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
LINE_ALIASES = [["HBETA_FLUX", "H_BETA_FLUX"], ["HGAMMA_FLUX", "H_GAMMA_FLUX"], ["NII_6584_FLUX"],
                ["SII_6716_FLUX", "SII_6717_FLUX"], ["SII_6731_FLUX"], ["OII_3726_FLUX"],
                ["OII_3729_FLUX"], ["OIII_5007_FLUX"]]
NII_COL = 2; KEEP = [0, 1, 3, 4, 5, 6, 7]
IDX7 = {"Hbeta": 0, "OII3726": 4, "OII3729": 5, "OIII5007": 6}     # positions in the 7-line array
COLS = ["Hbeta", "OII3726", "OII3729", "OIII5007"]
TITLE = {"Hbeta": r"H$\beta$", "OII3726": r"[OII]3726", "OII3729": r"[OII]3729", "OIII5007": r"[OIII]5007"}
CMAP = _LSC.from_list("bubblegum_dense", ["white", cmr.bubblegum(0.85), cmr.bubblegum(0.6),
                                          cmr.bubblegum(0.35), cmr.bubblegum(0.0)])
GRID = 70; MINCNT = 2


def sp(m): print(m, flush=True)


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def decode(a):
    a = np.asarray(a)
    if a.dtype.kind == "S": a = np.char.decode(a)
    return np.char.strip(a.astype(str))


def resolve(df, aliases):
    for c in aliases:
        if c in df.columns:
            return c
    raise KeyError(aliases)


def _paper_lims(x, y):
    xy = np.concatenate([np.asarray(x, float).ravel(), np.asarray(y, float).ravel()]); xy = xy[np.isfinite(xy)]
    q1, q3 = np.percentile(xy, [25, 75]); fe = 3.0 * (q3 - q1)
    kp = xy[(xy >= q1 - fe) & (xy <= q3 + fe)]
    if kp.size == 0: kp = xy
    loo, hii = float(kp.min()), float(kp.max()); pad = 0.04 * (hii - loo)
    return loo - pad, hii + pad


def _stats(true, pred):
    r = pred - true; p16, p84 = np.percentile(r, [16, 84])
    return dict(rmse=float(np.sqrt(np.mean(r ** 2))), scat=float(0.5 * (p84 - p16)),
                bias=float(np.median(r)), rho=float(spearmanr(true, pred).correlation))


def train_flow(key, Xn, Un, xdim, cond_dim, tag=""):
    flow = block_neural_autoregressive_flow(key=key, base_dist=Normal(jnp.zeros(xdim)), cond_dim=cond_dim)
    opt = optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(LR)); st = opt.init(eqx.filter(flow, eqx.is_inexact_array))

    @eqx.filter_jit
    def loss_fn(fl, x, u):
        return -jnp.mean(fl.log_prob(x, condition=u))

    @eqx.filter_jit
    def step(fl, s, x, u):
        loss, g = eqx.filter_value_and_grad(loss_fn)(fl, x, u)
        upd, s = opt.update(eqx.filter(g, eqx.is_inexact_array), s, params=eqx.filter(fl, eqx.is_inexact_array))
        return eqx.apply_updates(fl, upd), s, loss
    rng = np.random.default_rng(SEED + 7); n = Xn.shape[0]
    for ep in range(1, EPOCHS + 1):
        order = rng.permutation(n); ls = []
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            flow, st, loss = step(flow, st, jnp.asarray(Xn[idx]), jnp.asarray(Un[idx])); ls.append(float(loss))
        if ep % 50 == 0 or ep == 1:
            sp(f"  [{tag}] ep{ep:3d} loss={np.mean(ls):.4f}")
    return flow


def sample_mean(flow, U, Um, Us, Xm, Xs, xdim, key, n_mc=NMC, batch=40000):
    Un_all = jnp.asarray((U - Um) / Us); out = np.zeros((len(U), xdim))
    for lo in range(0, len(U), batch):
        hi = min(len(U), lo + batch); Un = Un_all[lo:hi]; acc = np.zeros((hi - lo, xdim))
        for _ in range(n_mc):
            key, sk = jr.split(key); keys = jr.split(sk, hi - lo)
            Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un)); acc += Xn
        out[lo:hi] = acc / n_mc
    return out * Xs + Xm


def build_training():
    t = Table.read(TRAIN_FITS, hdu=1); df = t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()
    cols = [resolve(df, a) for a in LINE_ALIASES]; F = np.column_stack([df[c].to_numpy(float) for c in cols])
    fha = df[resolve(df, ["HALPHA_FLUX", "H_ALPHA_FLUX"])].to_numpy(float)
    logm = df["LOGM_COLOR"].to_numpy(float); z = df["Z"].to_numpy(float)
    good = (np.all(F > 0, axis=1) & np.all(np.isfinite(F), axis=1) & (fha > 0) & np.isfinite(fha)
            & np.isfinite(logm) & np.isfinite(z) & (z > 0))
    X8 = np.log10(F[good]) - np.log10(fha[good])[:, None]
    return logm[good], log10_lum(z[good], fha[good]), PP04_A + PP04_B * X8[:, NII_COL], X8[:, KEEP]


def main():
    logm_tr, loglha_tr, logOH_tr, X7 = build_training()
    sp("train N=%d" % len(logm_tr))

    def _design(m, l):
        m = np.asarray(m, float); l = np.asarray(l, float)
        return np.column_stack([np.ones_like(m), m, l, m * m, l * l, m * l])
    coef, *_ = np.linalg.lstsq(_design(logm_tr, loglha_tr), logOH_tr, rcond=None)
    ZPRED = lambda m, l: _design(m, l) @ coef

    U2 = np.column_stack([logm_tr, loglha_tr]); U3 = np.column_stack([logm_tr, loglha_tr, logOH_tr])
    U2m, U2s = U2.mean(0), np.where(U2.std(0) == 0, 1, U2.std(0))
    U3m, U3s = U3.mean(0), np.where(U3.std(0) == 0, 1, U3.std(0))
    Xm, Xs = X7.mean(0), np.where(X7.std(0) == 0, 1, X7.std(0))
    k = jr.key(SEED); k1, k2, ks = jr.split(k, 3)
    sp("train baseline..."); f_base = train_flow(k1, ((X7 - Xm) / Xs).astype(np.float32), ((U2 - U2m) / U2s).astype(np.float32), 7, 2, "base")
    sp("train +Z...");       f_Z = train_flow(k2, ((X7 - Xm) / Xs).astype(np.float32), ((U3 - U3m) / U3s).astype(np.float32), 7, 3, "+Z")

    # ---- COSMOS ----
    h = fits.open(COSMOS); d1, d6 = h[1].data, h[6].data
    z = np.asarray(d1["Z"], float); zw = np.asarray(d1["ZWARN"], float)
    st = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float)

    def cf(nm):
        raw = np.asarray(d1[nm + "_FLUX"], float); iv = np.asarray(d1[nm + "_FLUX_IVAR"], float)
        return raw * 10 ** CAL[nm], raw * np.sqrt(np.clip(iv, 0, None))
    Fha, Sha = cf("HALPHA"); Fhb, Shb = cf("HBETA"); Foa, Soa = cf("OII_3726"); Fob, Sob = cf("OII_3729"); Fo3, So3 = cf("OIII_5007")
    base = (zw == 0) & (st == "GALAXY") & (z > 0.05) & np.isfinite(lpm) & (lpm > 6) & (lpm < 13); logm = lpm + MASS_ZP
    calm = base & (z < 0.49) & (Sha > 3) & (Shb > 3) & (Fha > 0) & (Fhb > 0)
    dec = (log10_lum(z, Fha) - log10_lum(z, Fhb))[calm]; mm = logm[calm]
    mb = np.percentile(mm, np.linspace(0, 100, 9)); mc = 0.5 * (mb[:-1] + mb[1:])
    Rm = np.array([np.median(dec[(mm >= a) & (mm < b)]) for a, b in zip(mb[:-1], mb[1:])]); Rof = lambda x: np.interp(x, mc, Rm, left=Rm[0], right=Rm[-1])

    def tier(mask, anchor_line):
        i = np.where(mask)[0]
        m = logm[i]; zz = z[i]
        lha = log10_lum(zz, Fha[i]) if anchor_line == "Ha" else log10_lum(zz, Fhb[i]) + Rof(m)  # conditioning L_Ha
        anchor = log10_lum(zz, Fha[i]) if anchor_line == "Ha" else log10_lum(zz, Fhb[i])
        inbox = (m >= BOX_M[0]) & (m <= BOX_M[1]) & (lha >= BOX_L[0]) & (lha <= BOX_L[1])
        Lobs = {"Hbeta": log10_lum(zz, Fhb[i]), "OII3726": log10_lum(zz, Foa[i]),
                "OII3729": log10_lum(zz, Fob[i]), "OIII5007": log10_lum(zz, Fo3[i])}
        Zevo = ZPRED(m, lha) + DZDOH * zz
        Pb = sample_mean(f_base, np.column_stack([m, lha]), U2m, U2s, Xm, Xs, 7, ks)
        Pz = sample_mean(f_Z, np.column_stack([m, lha, Zevo]), U3m, U3s, Xm, Xs, 7, ks)
        return dict(anchor=anchor, inbox=inbox, Lobs=Lobs, Pb=Pb, Pz=Pz, relHa=(anchor_line == "Ha"))
    T1 = tier(base & (z >= 0.05) & (z < 0.49) & (Sha > 3) & (Shb > 3) & (Soa > 3) & (Sob > 3) & (So3 > 3) & (Fhb > 0), "Ha")
    T2 = tier(base & (z >= 0.49) & (z < 1.00) & (Shb > 3) & (Soa > 3) & (Sob > 3) & (So3 > 3) & (Fhb > 0) & (Foa > 0) & (Fob > 0), "Hb")
    sp("COSMOS in-box: Tier1=%d Tier2=%d" % (T1["inbox"].sum(), T2["inbox"].sum()))

    def Lpred(T, P, col):
        if T["relHa"]:
            return T["anchor"] + P[:, IDX7[col]]
        return T["anchor"] + (P[:, IDX7[col]] - P[:, IDX7["Hbeta"]])

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.7)); hb_last = None
    XL = r"observed $\log L$ [erg s$^{-1}$]"
    for row, (T, tlab) in enumerate([(T1, "Tier 1\n$0.05<z<0.49$"), (T2, "Tier 2\n$0.49<z<1.0$")]):
        g = T["inbox"]
        for jc, col in enumerate(COLS):
            ax = axes[row, jc]
            if row == 1 and col == "Hbeta":
                ax.axis("off"); ax.text(0.5, 0.5, r"H$\beta$ = Tier-2" + "\n" + r"anchor (H$\alpha$" + "\n" + r"out of window)",
                                        ha="center", va="center", transform=ax.transAxes, fontsize=11.5, color="0.4"); continue
            Lo = T["Lobs"][col][g]; Lz = Lpred(T, T["Pz"], col)[g]; Lb = Lpred(T, T["Pb"], col)[g]
            lo, hi = _paper_lims(Lo, Lz); sz = _stats(Lo, Lz); sb = _stats(Lo, Lb)
            hb_last = ax.hexbin(Lo, Lz, gridsize=GRID, extent=(lo, hi, lo, hi), bins="log", mincnt=MINCNT, cmap=CMAP)
            ax.plot([lo, hi], [lo, hi], color="black", lw=2.6, ls=":", alpha=0.95)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
            if row == 0: ax.set_title(TITLE[col])
            ax.set_xlabel(XL)
            ax.text(0.04, 0.96, "RMSE=%.3f scat=%.3f\n$\\rho$=%.3f\nbias %+.3f$\\to$%+.3f" %
                    (sz["rmse"], sz["scat"], sz["rho"], sb["bias"], sz["bias"]),
                    transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                    bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round,pad=0.2"))
            sp("  %-6s %-9s  bias base %+.3f -> +Z %+.3f   RMSE %.3f->%.3f  scat %.3f->%.3f  rho %.3f->%.3f" %
               ("T%d" % (row + 1), col, sb["bias"], sz["bias"], sb["rmse"], sz["rmse"], sb["scat"], sz["scat"], sb["rho"], sz["rho"]))
        axes[row, 0 if row == 0 else 1].set_ylabel(r"predicted $\log L$  (+Z evolved-MZR)")
    fig.text(0.006, 0.75, "Tier 1\n$0.05<z<0.49$", rotation=90, va="center", ha="left", fontsize=13)
    fig.text(0.006, 0.29, "Tier 2\n$0.49<z<1.0$", rotation=90, va="center", ha="left", fontsize=13)
    fig.suptitle(r"Metallicity-conditioned flow (+ evolved MZR): observed vs predicted line luminosities", fontsize=15, y=0.995)
    fig.text(0.5, 0.945, r"box shows bias baseline$\to$+Z (evolved-MZR); NF conditioned on $(\log M_\star,\ \log L_{\mathrm{H}\alpha},\ 12+\log\mathrm{O/H})$", ha="center", fontsize=10.5, color="0.3")
    fig.tight_layout(rect=[0.035, 0, 0.93, 0.925])
    cax = fig.add_axes([0.945, 0.13, 0.014, 0.72]); cb = fig.colorbar(hb_last, cax=cax); cb.set_label(r"$\log_{10}(N)$ per hexbin", fontsize=12)
    out = REPO / "figs_ALTB" / "hiz_lum_1to1_metZ.png"; fig.savefig(out, bbox_inches="tight", dpi=200); sp("Saved: " + str(out))
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
