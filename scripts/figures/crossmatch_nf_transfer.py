"""
Step 1: rerun the NF cross-survey transfer test on the CROSS-MATCHED (same-galaxy)
SDSS<->DESI sample, to isolate the survey measurement effect from population/selection
differences.

  - build (u=logM*, logL_Ha ; X = 8 log line ratios to Ha) from each survey's OWN
    measurement of the matched galaxies (docs/crossmatch_sdss_desi_fluxes.csv)
  - train a BNAF on DESI-xm and another on SDSS-xm (same architecture as the real flows)
  - cross-test: DESI-flow -> predict SDSS-xm, SDSS-flow -> predict DESI-xm (MC-mean)
  - plot predicted vs true log L for Hb / [NII] / [OII] / [OIII], print per-line stats
Question: does [O III] still scatter on the matched sample?  (if yes -> AGN diagnosis)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import jax, jax.numpy as jnp, jax.random as jr
import optax, equinox as eqx
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa
try:
    import cmasher as cmr; CMAP = cmr.bubblegum
except Exception:
    CMAP = "magma"

REPO = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/"
CSV = REPO + "docs/crossmatch_sdss_desi_fluxes.csv"
OUTFIG = REPO + "figs/crossmatch_transfer"
FLUX_SCALE = 1e-17
SEED = 0
N_MC = 50
LINES = ["Hbeta", "Hgamma", "NII6584", "SII6717", "SII6731", "OII3726", "OII3729", "OIII5007"]
LABELS = [r"H$\beta$", r"H$\gamma$", "[N II]6584", "[S II]6717", "[S II]6731",
          "[O II]3726", "[O II]3729", "[O III]5007"]
SHOW = [("Hbeta", r"H$\beta$"), ("NII6584", r"[N II]$\lambda$6584"),
        ("OII3726", r"[O II]$\lambda$3726"), ("OIII5007", r"[O III]$\lambda$5007")]
I = {L: k for k, L in enumerate(LINES)}
CLASS_COL = {"SF": "#0072B2", "composite": "#E69F00", "AGN": "#CC79A7"}
CLASSES = ["SF", "composite", "AGN"]


def classify(nii6584, ha, oiii, hb):
    nii6584, ha, oiii, hb = map(lambda a: np.asarray(a, float), (nii6584, ha, oiii, hb))
    v = (nii6584 > 0) & (ha > 0) & (oiii > 0) & (hb > 0)
    v &= np.isfinite(nii6584) & np.isfinite(ha) & np.isfinite(oiii) & np.isfinite(hb)
    x = np.full(len(ha), np.nan); y = np.full(len(ha), np.nan)
    x[v] = np.log10(nii6584[v] / ha[v]); y[v] = np.log10(oiii[v] / hb[v])
    ka = np.where(x < 0.05, 0.61 / (x - 0.05) + 1.30, -np.inf)
    ke = np.where(x < 0.47, 0.61 / (x - 0.47) + 1.19, -np.inf)
    sf = v & (y < ka); agn = v & ((y > ke) | (x >= 0.47)); comp = v & ~sf & ~agn
    lab = np.full(len(ha), -1, int); lab[sf] = 0; lab[comp] = 1; lab[agn] = 2
    return lab, x, y


def sigma(r):
    r = r[np.isfinite(r)]
    p16, p84 = np.percentile(r, [16, 84]); return 0.5 * (p84 - p16)


def build(survey, df):
    ha = df[f"{survey}_Halpha"].to_numpy(float)
    z = df[f"z_{survey}"].to_numpy(float)
    logm = df[f"logm_{survey}"].to_numpy(float)
    fl = {L: df[f"{survey}_{L}"].to_numpy(float) for L in LINES}
    good = (ha > 0) & np.isfinite(ha) & np.isfinite(z) & (z > 0) & np.isfinite(logm)
    for L in LINES:
        good &= (fl[L] > 0) & np.isfinite(fl[L])
    loglha = np.full(len(df), np.nan)
    loglha[good] = (np.log10(ha[good] * FLUX_SCALE) + np.log10(4 * np.pi)
                    + 2 * np.log10(cosmo.luminosity_distance(z[good]).to("cm").value))
    X = np.column_stack([np.log10(fl[L]) - np.log10(ha) for L in LINES]).astype(np.float32)
    U = np.column_stack([logm, loglha]).astype(np.float32)
    return good, X, U, loglha


def train_flow(X, U, seed=0, epochs=400, batch=512, lr=3e-4, clip=1.0):
    Xm, Xs = X.mean(0), X.std(0); Xs[Xs == 0] = 1.0
    Um, Us = U.mean(0), U.std(0); Us[Us == 0] = 1.0
    Xn = jnp.asarray((X - Xm) / Xs); Un = jnp.asarray((U - Um) / Us)
    flow = block_neural_autoregressive_flow(
        key=jr.key(seed), base_dist=Normal(jnp.zeros(Xn.shape[1])), cond_dim=Un.shape[1])
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))
    opt_state = opt.init(eqx.filter(flow, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(flow, opt_state, x, u):
        def loss_fn(f): return -jnp.mean(f.log_prob(x, condition=u))
        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
        updates, opt_state = opt.update(eqx.filter(grads, eqx.is_inexact_array), opt_state,
                                        params=eqx.filter(flow, eqx.is_inexact_array))
        return eqx.apply_updates(flow, updates), opt_state, loss

    rng = np.random.default_rng(seed + 123); n = Xn.shape[0]
    Xn_np, Un_np = np.array(Xn), np.array(Un)
    for ep in range(1, epochs + 1):
        order = rng.permutation(n); losses = []
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            flow, opt_state, loss = step(flow, opt_state, jnp.asarray(Xn_np[idx]), jnp.asarray(Un_np[idx]))
            losses.append(float(loss))
        if ep % 50 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  loss={np.mean(losses):.4f}", flush=True)
    return flow, (Xm, Xs, Um, Us)


def predict_ratios(flow, U, norm, n_mc=N_MC, seed=7):
    Xm, Xs, Um, Us = norm
    Un = jnp.asarray(((U - Um) / Us).astype(np.float32))
    acc = np.zeros((len(U), len(LINES))); key = jr.key(seed)
    for _ in range(n_mc):
        key, sk = jr.split(key); kk = jr.split(sk, len(U))
        Xn = np.array(jax.vmap(lambda a, u: flow.sample(a, sample_shape=(), condition=u))(kk, Un))
        acc += Xn * Xs + Xm
    return acc / n_mc


def stats(pred, true):
    r = pred - true
    p16, p84 = np.percentile(r, [16, 84])
    return dict(rmse=np.sqrt(np.mean(r**2)), scat=0.5 * (p84 - p16),
                bias=np.median(r), rho=spearmanr(pred, true).correlation)


def main():
    df = pd.read_csv(CSV)
    print(f"matched: {len(df)}")
    gS, XS, US, llS = build("sdss", df)
    gD, XD, UD, llD = build("desi", df)
    print(f"valid (all 8 lines+Ha>0):  SDSS {gS.sum()}   DESI {gD.sum()}")
    # BPT class per galaxy in each survey's OWN measurement (for the diagnosis figure)
    labS, xbS, ybS = classify(df.sdss_NII6584, df.sdss_Halpha, df.sdss_OIII5007, df.sdss_Hbeta)
    labD, xbD, ybD = classify(df.desi_NII6584, df.desi_Halpha, df.desi_OIII5007, df.desi_Hbeta)
    BPT = {"SDSS$\\to$DESI": (labD, xbD, ybD), "DESI$\\to$SDSS": (labS, xbS, ybS)}

    print("\nTraining SDSS-xm flow..."); fS, nS = train_flow(XS[gS], US[gS], seed=SEED)
    print("Training DESI-xm flow..."); fD, nD = train_flow(XD[gD], UD[gD], seed=SEED + 1)

    # cross-survey: (train -> test)
    #  SDSS->DESI : SDSS-flow predicts DESI-xm galaxies
    #  DESI->SDSS : DESI-flow predicts SDSS-xm galaxies
    res = {}
    #  tag, train-flow, train-norm, test-mask, test-U, test-X(ratios), test-logLHa
    for tag, flow, norm, gte, Ute, Xte, llte in [("SDSS$\\to$DESI", fS, nS, gD, UD, XD, llD),
                                                 ("DESI$\\to$SDSS", fD, nD, gS, US, XS, llS)]:
        pr = predict_ratios(flow, Ute[gte], norm, seed=SEED + 11)
        # predicted / true log L_line = logL_Ha(test) + ratio
        ll = llte[gte]
        predL = ll[:, None] + pr
        trueL = ll[:, None] + Xte[gte]
        labte, xbte, ybte = BPT[tag]
        res[tag] = (predL, trueL, labte[gte], xbte[gte], ybte[gte])
        print(f"\n{tag}  per-line [logL] stats:")
        for L in LINES:
            s = stats(predL[:, I[L]], trueL[:, I[L]])
            print(f"  {L:9s} rmse={s['rmse']:.3f} scat={s['scat']:.3f} bias={s['bias']:+.3f} rho={s['rho']:.3f}")
        # [OIII] residual by BPT class
        roiii = predL[:, I["OIII5007"]] - trueL[:, I["OIII5007"]]
        print(f"  --> [OIII] residual by (test-survey) BPT class:")
        for k, cl in enumerate(CLASSES):
            mk = res[tag][2] == k
            print(f"      {cl:9s} N={int(mk.sum()):5d} ({mk.mean():4.0%})  bias={np.median(roiii[mk]):+.3f}  sigma={sigma(roiii[mk]):.3f}")

    # ---------------- figure: 2 dirs x 4 lines ----------------
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"axes.labelsize": 12, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.titlesize": 12})
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.6), constrained_layout=True)
    for row, tag in enumerate(["SDSS$\\to$DESI", "DESI$\\to$SDSS"]):
        predL, trueL = res[tag][0], res[tag][1]
        for col, (L, lab) in enumerate(SHOW):
            a = ax[row, col]; k = I[L]
            x, y = trueL[:, k], predL[:, k]
            m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
            lo = np.percentile(np.concatenate([x, y]), 0.5); hi = np.percentile(np.concatenate([x, y]), 99.5)
            a.hexbin(x, y, gridsize=45, extent=(lo, hi, lo, hi), bins="log", mincnt=3, cmap=CMAP)
            a.plot([lo, hi], [lo, hi], "k:", lw=1.5)
            a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal", adjustable="box")
            s = stats(y, x)
            a.text(0.05, 0.95, f"scat={s['scat']:.2f}\n$\\rho$={s['rho']:.2f}", transform=a.transAxes,
                   va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
            if row == 0:
                a.set_title(lab)
            if col == 0:
                a.set_ylabel(tag + "\n" + r"$\log L_{\rm pred}$")
            if row == 1:
                a.set_xlabel(r"$\log L_{\rm true}$")
    fig.suptitle("Cross-matched (same-galaxy) NF cross-survey transfer — does [O III] still scatter?", fontsize=13)
    for e in ("png", "pdf"):
        fig.savefig(f"{OUTFIG}.{e}", dpi=170, bbox_inches="tight")
    print("\nWrote", OUTFIG + ".png")

    # ---------------- diagnosis: [OIII] residual vs BPT class ----------------
    fig2, ax2 = plt.subplots(2, 2, figsize=(12.5, 10.5), constrained_layout=True)
    xn = np.linspace(-1.6, 0.3, 200); ka_c = 0.61 / (xn - 0.05) + 1.30; ke_c = 0.61 / (xn - 0.47) + 1.19
    for j, tag in enumerate(["SDSS$\\to$DESI", "DESI$\\to$SDSS"]):
        predL, trueL, lab, xb, yb = res[tag]
        r = predL[:, I["OIII5007"]] - trueL[:, I["OIII5007"]]
        a = ax2[0, j]
        for k, cl in enumerate(CLASSES):
            mk = lab == k
            a.hist(r[mk], bins=np.linspace(-1.2, 1.2, 61), density=True, histtype="step", lw=2,
                   color=CLASS_COL[cl], label=f"{cl} ({mk.mean():.0%}, $\\sigma$={sigma(r[mk]):.2f})")
        a.axvline(0, color="k", ls=":", lw=1)
        a.set_xlabel(r"[O III] residual $\log L_{\rm pred}-\log L_{\rm true}$ [dex]")
        a.set_ylabel("normalized density"); a.set_title(tag + "  (test-survey BPT)"); a.legend()
        b = ax2[1, j]
        g = np.isfinite(xb) & np.isfinite(yb)
        hb = b.hexbin(xb[g], yb[g], C=np.abs(r[g]), gridsize=45, reduce_C_function=np.median,
                      cmap=CMAP, extent=(-1.6, 0.5, -1.2, 1.3), mincnt=1)
        m1 = xn < 0.05; m2 = xn < 0.47
        b.plot(xn[m1], ka_c[m1], "k--", lw=1.5, label="Kauffmann03")
        b.plot(xn[m2], ke_c[m2], "k-.", lw=1.5, label="Kewley01")
        b.set_xlabel(r"$\log$([N II]/H$\alpha$)"); b.set_ylabel(r"$\log$([O III]/H$\beta$)")
        b.set_xlim(-1.6, 0.5); b.set_ylim(-1.2, 1.3); b.set_title(tag + " BPT plane"); b.legend(loc="lower left")
        cb = fig2.colorbar(hb, ax=b, shrink=0.9); cb.set_label(r"median $|$[O III] residual$|$ [dex]")
    fig2.suptitle("Matched-sample NF: largest [O III] transfer residuals are AGN/composite excitation", fontsize=13)
    for e in ("png", "pdf"):
        fig2.savefig(REPO + f"figs/crossmatch_oiii_agn_scatter.{e}", dpi=170, bbox_inches="tight")
    print("Wrote figs/crossmatch_oiii_agn_scatter.png")


if __name__ == "__main__":
    main()
