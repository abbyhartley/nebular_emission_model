"""
Appendix stress test: does conditioning the flow on the Balmer decrement R=Ha/Hb
tighten the predicted-vs-observed line ratios?

In-survey (SDSS), fixed 80/20 train/test split. We drop Hbeta from the targets
(since R fixes Hbeta/Ha) and model the OTHER 7 log line ratios. Three configs,
same split, same targets:
  baseline : condition on (logM*, logL_Ha)                [2D]
  +R oracle: condition on (logM*, logL_Ha, logR_true)     [3D, true R]  -> upper bound
  +R 2stage: same +R flow, but fed R_hat predicted from (logM*, logL_Ha)  -> realistic
We report per-line RMSE / scatter / Spearman rho on the test set, plus the gain
over baseline. Physical expectation: the blue lines (Hgamma, [OII], [OIII])
tighten, the red lines ([NII],[SII]) barely change.
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = Path(BASE + "nebular_emission_model")
FITS = BASE + "SDSS_main_training_data.fits"
META = REPO / "nf_sdss_main_meta.pkl"
FLUX_SCALE = 1e-17
SEED = 0
EPOCHS = 200
BATCH = 2048
LR = 3e-4
CLIP = 1.0
NMC = 50
TEST_FRAC = 0.2

LABELS8 = [r"H$\beta$", r"H$\gamma$", r"[N II]", r"[S II]$_{6717}$", r"[S II]$_{6731}$",
           r"[O II]$_{3726}$", r"[O II]$_{3729}$", r"[O III]"]
LABELS7 = LABELS8[1:]          # drop Hbeta (col 0)


def load_df(p):
    t = Table.read(p, hdu=1)
    return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()


def build_arrays():
    meta = pickle.load(open(META, "rb"))
    df = load_df(FITS)
    raw = [c[6:] if c.startswith("LOG10_") else c for c in meta["resolved"]["target_cols"]]
    ha = next(c for c in ["H_ALPHA_FLUX", "HALPHA_FLUX"] if c in df.columns)
    F = np.column_stack([df[c].to_numpy(float) for c in raw])       # (N,8) fluxes, Hbeta=col0
    fha = df[ha].to_numpy(float)
    logm = df["LOGM_COLOR"].to_numpy(float)
    z = df["Z_1"].to_numpy(float)
    okz = np.isfinite(z) & (z > 0) & np.isfinite(fha) & (fha > 0)
    loglha = np.full(len(df), np.nan)
    loglha[okz] = (np.log10(fha[okz] * FLUX_SCALE) + np.log10(4 * np.pi)
                   + 2 * np.log10(cosmo.luminosity_distance(z[okz]).to("cm").value))
    good = (np.all(F > 0, axis=1) & (fha > 0) & np.isfinite(fha) & np.all(np.isfinite(F), axis=1)
            & np.isfinite(logm) & np.isfinite(loglha))
    X8 = np.log10(F[good]) - np.log10(fha[good])[:, None]           # (Ng,8) log(line/Ha)
    logR = (np.log10(fha[good]) - np.log10(F[good, 0]))             # log10(Ha/Hbeta)
    return logm[good], loglha[good], logR, X8


def train_flow(key, Xn, Un, xdim, cond_dim, epochs=EPOCHS, tag=""):
    flow = block_neural_autoregressive_flow(key=key, base_dist=Normal(jnp.zeros(xdim)), cond_dim=cond_dim)
    opt = optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(LR))
    opt_state = opt.init(eqx.filter(flow, eqx.is_inexact_array))

    @eqx.filter_jit
    def loss_fn(fl, x, u):
        return -jnp.mean(fl.log_prob(x, condition=u))

    @eqx.filter_jit
    def step(fl, st, x, u):
        loss, g = eqx.filter_value_and_grad(loss_fn)(fl, x, u)
        upd, st = opt.update(eqx.filter(g, eqx.is_inexact_array), st, params=eqx.filter(fl, eqx.is_inexact_array))
        return eqx.apply_updates(fl, upd), st, loss

    rng = np.random.default_rng(SEED + 7)
    n = Xn.shape[0]
    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        losses = []
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            fl_x = jnp.asarray(Xn[idx]); fl_u = jnp.asarray(Un[idx])
            flow, opt_state, loss = step(flow, opt_state, fl_x, fl_u)
            losses.append(float(loss))
        if ep % 40 == 0 or ep == 1:
            print(f"  [{tag}] epoch {ep:3d}  loss={np.mean(losses):.4f}", flush=True)
    return flow


def mc_mean(flow, U, U_mean, U_std, X_mean, X_std, xdim, key, n=NMC):
    Un = jnp.asarray((U - U_mean) / U_std)
    acc = jnp.zeros((U.shape[0], xdim))
    keys = jr.split(key, n)
    for k in keys:
        kk = jr.split(k, U.shape[0])
        s = jax.vmap(lambda a, u: flow.sample(a, sample_shape=(), condition=u))(kk, Un)
        acc = acc + s
    Xn = np.array(acc) / n
    return Xn * X_std + X_mean


def line_metrics(pred, true):
    r = pred - true
    p16, p84 = np.percentile(r, [16, 84])
    return dict(rmse=float(np.sqrt(np.mean(r**2))),
                scatter=float(0.5 * (p84 - p16)),
                nmad=float(1.4826 * np.median(np.abs(r - np.median(r)))),
                rho=float(spearmanr(pred, true).correlation))


def main():
    logm, loglha, logR, X8 = build_arrays()
    N = len(logm)
    print(f"N good = {N}", flush=True)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    ntest = int(TEST_FRAC * N)
    te, tr = perm[:ntest], perm[ntest:]

    X7 = X8[:, 1:]                                   # drop Hbeta
    U2 = np.column_stack([logm, loglha])
    U3 = np.column_stack([logm, loglha, logR])

    # standardization from TRAIN
    def stats(a, idx):
        m = a[idx].mean(0); s = a[idx].std(0); s = np.where(s == 0, 1.0, s); return m, s
    X7m, X7s = stats(X7, tr)
    U2m, U2s = stats(U2, tr)
    U3m, U3s = stats(U3, tr)
    lRm, lRs = logR[tr].mean(), logR[tr].std()

    Xn7 = ((X7 - X7m) / X7s).astype(np.float32)
    Un2 = ((U2 - U2m) / U2s).astype(np.float32)
    Un3 = ((U3 - U3m) / U3s).astype(np.float32)
    lRn = ((logR - lRm) / lRs).astype(np.float32)[:, None]

    k = jr.key(SEED)
    k1, k2, k3, ks = jr.split(k, 4)
    print("Training baseline (2D->7)...", flush=True)
    f_base = train_flow(k1, Xn7[tr], Un2[tr], 7, 2, tag="base")
    print("Training +R (3D->7)...", flush=True)
    f_R = train_flow(k2, Xn7[tr], Un3[tr], 7, 3, tag="+R")
    print("Training R-predictor (2D->1)...", flush=True)
    f_Rp = train_flow(k3, lRn[tr], Un2[tr], 1, 2, tag="Rpred")

    Xtrue = X7[te]
    # baseline
    Pb = mc_mean(f_base, U2[te], U2m, U2s, X7m, X7s, 7, ks)
    # +R oracle (true logR)
    Po = mc_mean(f_R, U3[te], U3m, U3s, X7m, X7s, 7, ks)
    # R-predictor -> Rhat, then two-stage
    Rhat = mc_mean(f_Rp, U2[te], U2m, U2s, np.array([lRm]), np.array([lRs]), 1, ks)[:, 0]
    U3_hat = np.column_stack([logm[te], loglha[te], Rhat])
    Pt = mc_mean(f_R, U3_hat, U3m, U3s, X7m, X7s, 7, ks)

    rR = line_metrics(Rhat, logR[te])
    print(f"\nR-predictor (logR from M*,L_Ha): RMSE={rR['rmse']:.3f} dex, rho={rR['rho']:.3f}", flush=True)

    rows = []
    for j, lab in enumerate(LABELS7):
        mb = line_metrics(Pb[:, j], Xtrue[:, j])
        mo = line_metrics(Po[:, j], Xtrue[:, j])
        mt = line_metrics(Pt[:, j], Xtrue[:, j])
        rows.append(dict(line=lab.replace("$", "").replace("\\", ""),
                         base_scatter=mb["scatter"], oracle_scatter=mo["scatter"], twostage_scatter=mt["scatter"],
                         base_rmse=mb["rmse"], oracle_rmse=mo["rmse"], twostage_rmse=mt["rmse"],
                         base_rho=mb["rho"], oracle_rho=mo["rho"], twostage_rho=mt["rho"],
                         oracle_scatter_redpct=100 * (1 - mo["scatter"] / mb["scatter"]),
                         twostage_scatter_redpct=100 * (1 - mt["scatter"] / mb["scatter"])))
    tab = pd.DataFrame(rows)
    # pooled
    pooled = dict(line="ALL-7 pooled",
                  base_scatter=line_metrics(Pb.ravel(), Xtrue.ravel())["scatter"],
                  oracle_scatter=line_metrics(Po.ravel(), Xtrue.ravel())["scatter"],
                  twostage_scatter=line_metrics(Pt.ravel(), Xtrue.ravel())["scatter"],
                  base_rho=line_metrics(Pb.ravel(), Xtrue.ravel())["rho"],
                  oracle_rho=line_metrics(Po.ravel(), Xtrue.ravel())["rho"],
                  twostage_rho=line_metrics(Pt.ravel(), Xtrue.ravel())["rho"])
    tab = pd.concat([tab, pd.DataFrame([pooled])], ignore_index=True)
    tab.to_csv(REPO / "docs" / "balmer_conditioning_results.csv", index=False)
    print("\n" + tab.to_string(index=False), flush=True)
    print("\nWrote docs/balmer_conditioning_results.csv", flush=True)

    # figure: per-line scatter, 3 configs
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 11, "ytick.labelsize": 12, "legend.fontsize": 12})
    per = tab[tab.line != "ALL-7 pooled"]
    x = np.arange(len(per)); w = 0.27
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x - w, per.base_scatter, w, label="baseline (M*, L_Ha)", color="#0072B2")
    ax.bar(x, per.twostage_scatter, w, label="+R predicted (two-stage)", color="#009E73")
    ax.bar(x + w, per.oracle_scatter, w, label="+R true (oracle)", color="#CC79A7")
    ax.set_xticks(x); ax.set_xticklabels(LABELS7, rotation=30, ha="right")
    ax.set_ylabel(r"scatter of $\log_{10}(L_{\rm line}/L_{\mathrm{H}\alpha})$  [dex]")
    ax.set_title("Effect of conditioning on the Balmer decrement (SDSS in-survey)")
    ax.legend(frameon=True, framealpha=0.9)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(REPO / "figs" / f"balmer_conditioning_scatter.{e}", dpi=180, bbox_inches="tight")
    print("Wrote figs/balmer_conditioning_scatter.png", flush=True)


if __name__ == "__main__":
    main()
