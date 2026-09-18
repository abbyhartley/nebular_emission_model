"""
Cross-survey Balmer-decrement conditioning test.

Does giving the flow the TARGET survey's true Balmer decrement R=Ha/Hb reduce the
cross-survey offset -- particularly for the blue, dust-sensitive [O III]?

Train baseline (M*,L_Ha->7 lines, Hbeta dropped) and +R (M*,L_Ha,logR->7) flows on
each FULL survey, then apply across surveys feeding the target survey's TRUE logR
(the "external R" oracle). Report per-line bias (median residual), scatter, and rho.
"""
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
FLUX_SCALE = 1e-17
SEED = 0
EPOCHS = 200
BATCH = 2048
LR = 3e-4
CLIP = 1.0
NMC = 50

CFG = {
    "sdss": dict(fits=BASE + "SDSS_main_training_data.fits", meta=REPO / "nf_sdss_main_meta.pkl", z="Z_1"),
    "desi": dict(fits=BASE + "DESI_BGS_training_data.fits", meta=REPO / "nf_desi_bgs_meta.pkl", z="Z"),
}
LABELS7 = [r"H$\gamma$", r"[N II]", r"[S II]$_a$", r"[S II]$_b$",
           r"[O II]$_a$", r"[O II]$_b$", r"[O III]"]


def load_df(p):
    t = Table.read(p, hdu=1)
    return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()


def build(survey):
    c = CFG[survey]
    meta = pickle.load(open(c["meta"], "rb"))
    df = load_df(c["fits"])
    raw = [x[6:] if x.startswith("LOG10_") else x for x in meta["resolved"]["target_cols"]]
    ha = next(x for x in ["H_ALPHA_FLUX", "HALPHA_FLUX"] if x in df.columns)
    F = np.column_stack([df[x].to_numpy(float) for x in raw])
    fha = df[ha].to_numpy(float)
    logm = df["LOGM_COLOR"].to_numpy(float)
    z = df[c["z"]].to_numpy(float)
    okz = np.isfinite(z) & (z > 0) & np.isfinite(fha) & (fha > 0)
    loglha = np.full(len(df), np.nan)
    loglha[okz] = (np.log10(fha[okz] * FLUX_SCALE) + np.log10(4 * np.pi)
                   + 2 * np.log10(cosmo.luminosity_distance(z[okz]).to("cm").value))
    good = (np.all(F > 0, axis=1) & (fha > 0) & np.isfinite(fha) & np.all(np.isfinite(F), axis=1)
            & np.isfinite(logm) & np.isfinite(loglha))
    X8 = np.log10(F[good]) - np.log10(fha[good])[:, None]
    logR = np.log10(fha[good]) - np.log10(F[good, 0])
    return dict(logm=logm[good], loglha=loglha[good], logR=logR, X7=X8[:, 1:])


def train_flow(key, Xn, Un, xdim, cond_dim, tag=""):
    flow = block_neural_autoregressive_flow(key=key, base_dist=Normal(jnp.zeros(xdim)), cond_dim=cond_dim)
    opt = optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(LR))
    st = opt.init(eqx.filter(flow, eqx.is_inexact_array))

    @eqx.filter_jit
    def loss_fn(fl, x, u):
        return -jnp.mean(fl.log_prob(x, condition=u))

    @eqx.filter_jit
    def step(fl, s, x, u):
        loss, g = eqx.filter_value_and_grad(loss_fn)(fl, x, u)
        upd, s = opt.update(eqx.filter(g, eqx.is_inexact_array), s, params=eqx.filter(fl, eqx.is_inexact_array))
        return eqx.apply_updates(fl, upd), s, loss

    rng = np.random.default_rng(SEED + 7)
    n = Xn.shape[0]
    for ep in range(1, EPOCHS + 1):
        order = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            flow, st, loss = step(flow, st, jnp.asarray(Xn[idx]), jnp.asarray(Un[idx]))
        if ep % 50 == 0 or ep == 1:
            print(f"  [{tag}] ep {ep} loss {float(loss):.3f}", flush=True)
    return flow


def std_fit(a):
    m = a.mean(0); s = a.std(0); s = np.where(s == 0, 1.0, s); return m, s


def mc_mean(flow, U, Um, Us, Xm, Xs, xdim, key, n=NMC):
    Un = jnp.asarray((U - Um) / Us)
    acc = jnp.zeros((U.shape[0], xdim))
    for k in jr.split(key, n):
        kk = jr.split(k, U.shape[0])
        acc = acc + jax.vmap(lambda a, u: flow.sample(a, sample_shape=(), condition=u))(kk, Un)
    return (np.array(acc) / n) * Xs + Xm


def metrics(pred, true):
    r = pred - true
    p16, p84 = np.percentile(r, [16, 84])
    return dict(bias=float(np.median(r)), scatter=float(0.5 * (p84 - p16)),
                rmse=float(np.sqrt(np.mean(r**2))), rho=float(spearmanr(pred, true).correlation))


def main():
    D = {s: build(s) for s in CFG}
    for s in D:
        print(s, "N=", len(D[s]["logm"]), flush=True)

    flows = {}
    k = jr.key(SEED)
    for s in ("sdss", "desi"):
        d = D[s]
        U2 = np.column_stack([d["logm"], d["loglha"]])
        U3 = np.column_stack([d["logm"], d["loglha"], d["logR"]])
        X7 = d["X7"]
        U2m, U2s = std_fit(U2); U3m, U3s = std_fit(U3); X7m, X7s = std_fit(X7)
        k, ka, kb = jr.split(k, 3)
        print(f"Training {s} baseline...", flush=True)
        fb = train_flow(ka, ((X7 - X7m) / X7s).astype(np.float32), ((U2 - U2m) / U2s).astype(np.float32), 7, 2, tag=f"{s}-base")
        print(f"Training {s} +R...", flush=True)
        fr = train_flow(kb, ((X7 - X7m) / X7s).astype(np.float32), ((U3 - U3m) / U3s).astype(np.float32), 7, 3, tag=f"{s}-R")
        flows[s] = dict(base=fb, R=fr, U2m=U2m, U2s=U2s, U3m=U3m, U3s=U3s, X7m=X7m, X7s=X7s)

    rows = []
    ks = jr.key(SEED + 99)
    for tr, te in [("sdss", "desi"), ("desi", "sdss")]:
        d = D[te]
        Xtrue = d["X7"]
        U2 = np.column_stack([d["logm"], d["loglha"]])
        U3 = np.column_stack([d["logm"], d["loglha"], d["logR"]])   # target survey's TRUE logR
        f = flows[tr]
        Pb = mc_mean(f["base"], U2, f["U2m"], f["U2s"], f["X7m"], f["X7s"], 7, ks)
        Po = mc_mean(f["R"], U3, f["U3m"], f["U3s"], f["X7m"], f["X7s"], 7, ks)
        for j, lab in enumerate(LABELS7):
            mb = metrics(Pb[:, j], Xtrue[:, j]); mo = metrics(Po[:, j], Xtrue[:, j])
            rows.append(dict(direction=f"{tr}->{te}", line=lab.replace("$", "").replace("\\", ""),
                             base_bias=mb["bias"], oracle_bias=mo["bias"],
                             base_scatter=mb["scatter"], oracle_scatter=mo["scatter"],
                             base_rho=mb["rho"], oracle_rho=mo["rho"]))
    tab = pd.DataFrame(rows)
    tab.to_csv(REPO / "docs" / "balmer_conditioning_crosssurvey.csv", index=False)
    print("\n" + tab.to_string(index=False), flush=True)

    # figure: per-line |bias| and scatter, baseline vs oracle, 2 directions x 2 metrics
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"axes.labelsize": 13, "axes.titlesize": 12, "xtick.labelsize": 9,
                         "ytick.labelsize": 10, "legend.fontsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    dirs = ["sdss->desi", "desi->sdss"]
    x = np.arange(len(LABELS7)); w = 0.38
    for r_i, direction in enumerate(dirs):
        sub = tab[tab.direction == direction].reset_index(drop=True)
        for c_i, (col, ylab) in enumerate([("bias", "median residual (bias) [dex]"),
                                           ("scatter", "scatter [dex]")]):
            ax = axes[r_i, c_i]
            ax.bar(x - w / 2, sub[f"base_{col}"], w, label="baseline", color="#0072B2")
            ax.bar(x + w / 2, sub[f"oracle_{col}"], w, label="+R (true, external)", color="#CC79A7")
            if col == "bias":
                ax.axhline(0, color="0.4", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels(LABELS7, rotation=30, ha="right")
            ax.set_ylabel(ylab)
            ttl = direction.upper().replace("->", r" $\rightarrow$ ")
            ax.set_title(ttl)
            if r_i == 0 and c_i == 0:
                ax.legend(frameon=True, framealpha=0.9)
    fig.suptitle("Cross-survey: effect of conditioning on the (external) Balmer decrement", fontsize=13)
    for e in ("png", "pdf"):
        fig.savefig(REPO / "figs" / f"balmer_conditioning_crosssurvey.{e}", dpi=170, bbox_inches="tight")
    print("Wrote figs/balmer_conditioning_crosssurvey.png", flush=True)


if __name__ == "__main__":
    main()
