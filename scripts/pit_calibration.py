"""
A1: Calibration / coverage test for the conditional flows.
For each in-survey flow we draw N_SAMP samples per galaxy and compute, per line,
the PIT = rank of the true value among samples (empirical CDF). A calibrated
conditional density gives PIT ~ Uniform(0,1). We also compute empirical central-
interval coverage vs nominal. Outputs a per-line PIT-histogram figure (2 surveys)
and a coverage figure, plus a summary CSV.
"""
from pathlib import Path
import pickle
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
FLUX_SCALE = 1e-17
N_SAMP = 500
N_SUB = 15000
SEED = 0
LABELS = [r"H$\beta$", r"H$\gamma$", r"[N II]", r"[S II]$_{6716}$",
          r"[S II]$_{6731}$", r"[O II]$_{3726}$", r"[O II]$_{3729}$", r"[O III]"]
SURVEYS = [("SDSS", BASE+"SDSS_main_training_data.fits", REPO+"nf_sdss_main.eqx", REPO+"nf_sdss_main_meta.pkl", "sdss"),
           ("DESI", BASE+"DESI_BGS_training_data.fits", REPO+"nf_desi_bgs.eqx", REPO+"nf_desi_bgs_meta.pkl", "desi")]


def load_scalar_df(p):
    t = Table.read(p, hdu=1); return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()

def add_loglha(df, survey):
    if survey == "sdss":
        z = df["Z_1"].to_numpy(float); ha = df["H_ALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    else:
        z = df["Z"].to_numpy(float); ha = df["HALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    m = np.isfinite(z)&(z>0)&np.isfinite(ha)&(ha>0)
    out = np.full(len(df), np.nan)
    out[m] = np.log10(ha[m])+np.log10(4*np.pi)+2*np.log10(cosmo.luminosity_distance(z[m]).to("cm").value)
    df = df.copy(); df["LOG_LHA"] = out; return df

def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))), base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    return eqx.tree_deserialise_leaves(flow_path, tmpl)

def get_XU(df, meta):
    tcols = meta["resolved"]["target_cols"]
    raw = [c[6:] if c.startswith("LOG10_") else c for c in tcols]
    ha = next(c for c in ["H_ALPHA_FLUX","HALPHA_FLUX"] if c in df.columns)
    logm = meta["resolved"]["logmstar_col"]
    F = np.column_stack([df[c].to_numpy(float) for c in raw]); fha = df[ha].to_numpy(float)
    u1 = df[logm].to_numpy(float); u2 = df["LOG_LHA"].to_numpy(float)
    good = np.all(F>0,axis=1)&(fha>0)&np.isfinite(fha)&np.all(np.isfinite(F),axis=1)&np.isfinite(u1)&np.isfinite(u2)
    X = np.log10(F[good])-np.log10(fha[good])[:,None]
    U = np.column_stack([u1[good], u2[good]])
    return X, U

def pit_for_survey(name, fits, flowp, metap, key):
    meta = pickle.load(open(metap,"rb")); flow = load_flow(flowp, meta)
    df = add_loglha(load_scalar_df(fits), key)
    X, U = get_XU(df, meta)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(N_SUB, len(X)), replace=False)
    X, U = X[idx], U[idx]
    Un = ((U - meta["U_mean"]) / meta["U_std"]).astype(np.float32)
    xdim = X.shape[1]
    below = np.zeros((len(X), xdim))
    for i in range(N_SAMP):
        keys = jr.split(jr.key(SEED + 1000 + i), len(X))
        s = jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, jnp.asarray(Un))
        Xs = np.array(s) * meta["X_std"] + meta["X_mean"]     # raw log-ratio samples
        below += (Xs < X)
    pit = below / N_SAMP                                       # (N, 8)
    print(f"{name}: PIT computed on N={len(X)} x {N_SAMP} samples")
    return pit

def main():
    pits = {}
    for name, fits, flowp, metap, key in SURVEYS:
        pits[name] = pit_for_survey(name, fits, flowp, metap, key)

    plt.style.use(["science","no-latex"])
    # PIT histograms: 2 surveys x 8 lines
    fig, axes = plt.subplots(2, 8, figsize=(20, 5.2), constrained_layout=True)
    nb = 20
    for r, (name, *_ ) in enumerate(SURVEYS):
        P = pits[name]
        for c in range(8):
            ax = axes[r, c]
            ax.hist(P[:, c], bins=np.linspace(0,1,nb+1), density=True, color="#0072B2", alpha=0.85)
            ax.axhline(1.0, color="k", ls=":", lw=1)
            ax.set_ylim(0, 2.2); ax.set_xticks([0,0.5,1])
            if r == 0: ax.set_title(LABELS[c], fontsize=11)
            if c == 0: ax.set_ylabel(name+"\nPIT density", fontsize=11)
            if r == 1: ax.set_xlabel("PIT", fontsize=10)
    fig.suptitle("Probability integral transform (calibration): uniform = calibrated", fontsize=13)
    for e in ("png","pdf"): fig.savefig(REPO+f"figs/pit_histograms.{e}", dpi=180, bbox_inches="tight")
    print("Wrote figs/pit_histograms.png")

    # coverage curves + summary
    qs = np.linspace(0.05, 0.95, 19)
    rows = []
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for a,(name,*_ ) in zip(axes2, SURVEYS):
        P = pits[name]
        a.plot([0,1],[0,1],"k:",lw=1.2, label="ideal")
        for c in range(8):
            cov = [np.mean(np.abs(P[:,c]-0.5) < q/2) for q in qs]
            a.plot(qs, cov, lw=1.4, label=LABELS[c])
            rows.append(dict(survey=name, line=LABELS[c],
                             pit_mean=float(P[:,c].mean()), pit_std=float(P[:,c].std()),
                             cov68=float(np.mean(np.abs(P[:,c]-0.5)<0.34)),
                             cov90=float(np.mean(np.abs(P[:,c]-0.5)<0.45))))
        a.set_title(name); a.set_xlabel("nominal central interval"); a.set_ylabel("empirical coverage")
        a.legend(fontsize=7, ncol=2)
    for e in ("png","pdf"): fig2.savefig(REPO+f"figs/pit_coverage.{e}", dpi=180, bbox_inches="tight")
    print("Wrote figs/pit_coverage.png")

    import pandas as pd
    pd.DataFrame(rows).to_csv(REPO+"docs/pit_coverage_summary.csv", index=False)
    print("Wrote docs/pit_coverage_summary.csv")
    for name in pits:
        P = pits[name]
        print(f"{name}: mean|cov68-0.68| = {np.mean([abs(np.mean(np.abs(P[:,c]-0.5)<0.34)-0.68) for c in range(8)]):.3f}")

if __name__ == "__main__":
    main()
