"""
Tier-1 covariance/correlation benchmark: does the NF reproduce the joint structure
of the 8 log line ratios better than simpler models?

For each survey we fit, on the SAME galaxies, four comparison models for the
8-D target x = log10(L_line/L_Ha) given u = (logM*, logL_Ha):
  - NF            : samples from the trained conditional flow (this work)
  - CondGauss     : linear regression mean + FULL residual covariance, Gaussian draws
  - IndepNoise    : per-line linear regression mean + INDEPENDENT (diagonal) Gaussian noise
                    (emulates a point-estimate-plus-independent-noise approach)
  - Shuffle       : observed x with each line permuted independently (destroys all correlation)
We compare each model's correlation matrix to the data's and report the Frobenius
distance ||C_model - C_data||_F (the diagonal cancels, so this measures off-diagonal
structure). Expectation: CondGauss matches by construction (2nd moments), IndepNoise
keeps only the conditioning-driven correlation and misses the residual structure,
Shuffle fails entirely, NF matches.
"""
from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
SDSS_FITS = Path(BASE + "SDSS_main_training_data.fits")
DESI_FITS = Path(BASE + "DESI_BGS_training_data.fits")
FLOW_SDSS, META_SDSS = Path(REPO + "nf_sdss_main.eqx"), Path(REPO + "nf_sdss_main_meta.pkl")
FLOW_DESI, META_DESI = Path(REPO + "nf_desi_bgs.eqx"), Path(REPO + "nf_desi_bgs_meta.pkl")

FLUX_SCALE = 1e-17
SEED = 0
LABELS = [r"H$\beta$", r"H$\gamma$", r"[N\,II]", r"[S\,II]$_{6716}$",
          r"[S\,II]$_{6731}$", r"[O\,II]$_{3726}$", r"[O\,II]$_{3729}$", r"[O\,III]"]


def load_scalar_df(p):
    t = Table.read(p, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()


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


def get_data_XU(df, meta, *, survey):
    """Return X (N,8 log ratios) and U (N,2 conditioning) for the same good rows."""
    tcols = meta["resolved"]["target_cols"]
    raw = [c[6:] if c.startswith("LOG10_") else c for c in tcols]
    hacol = meta["resolved"].get("logha_col") or ""
    ha = next((c for c in [hacol[6:] if hacol.startswith("LOG10_") else hacol,
                           "H_ALPHA_FLUX", "HALPHA_FLUX", "HA_FLUX"] if c in df.columns), None)
    logm = meta["resolved"]["logmstar_col"]
    F = np.column_stack([df[c].to_numpy(float) for c in raw])
    fha = df[ha].to_numpy(float)
    u1 = df[logm].to_numpy(float); u2 = df["LOG_LHA"].to_numpy(float)
    good = (np.all(F > 0, axis=1) & (fha > 0) & np.isfinite(fha) & np.all(np.isfinite(F), axis=1)
            & np.isfinite(u1) & np.isfinite(u2))
    X = np.log10(F[good]) - np.log10(fha[good])[:, None]
    U = np.column_stack([u1[good], u2[good]])
    return X, U


def sample_from_U(flow, meta, U, *, seed):
    Un = (U.astype(np.float32) - meta["U_mean"]) / meta["U_std"]
    keys = jr.split(jr.key(seed + 1234), len(U))
    Xn = jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, jnp.asarray(Un))
    return np.array(Xn) * meta["X_std"] + meta["X_mean"]


def ols(X, U):
    A = np.column_stack([np.ones(len(U)), U])          # [1, logM*, logLHa]
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    Xhat = A @ coef
    return Xhat, X - Xhat


def build_models(X, U, flow, meta, *, seed):
    rng = np.random.default_rng(seed)
    Xhat, E = ols(X, U)
    Sigma = np.cov(E, rowvar=False)
    sig = E.std(0)
    models = {
        "Data": X,
        "NF (this work)": sample_from_U(flow, meta, U, seed=seed + 5),
        "Cond. Gaussian": Xhat + rng.multivariate_normal(np.zeros(X.shape[1]), Sigma, size=len(X)),
        "Indep. + noise": Xhat + rng.normal(size=X.shape) * sig,
        "Shuffle": np.column_stack([rng.permutation(X[:, k]) for k in range(X.shape[1])]),
    }
    return {k: np.corrcoef(v, rowvar=False) for k, v in models.items()}


def main():
    surveys = [
        ("SDSS", SDSS_FITS, FLOW_SDSS, META_SDSS, "sdss"),
        ("DESI", DESI_FITS, FLOW_DESI, META_DESI, "desi"),
    ]
    order = ["Data", "NF (this work)", "Cond. Gaussian", "Indep. + noise", "Shuffle"]
    results = {}
    for name, fits, flowp, metap, key in surveys:
        df = add_loglha(load_scalar_df(fits), survey=key)
        meta = pickle.load(open(metap, "rb"))
        flow = load_flow(flowp, meta)
        X, U = get_data_XU(df, meta, survey=key)
        print(f"{name}: N={len(X)}")
        results[name] = build_models(X, U, flow, meta, seed=SEED)

    # Frobenius distances to data
    print("\n=== Frobenius distance ||C_model - C_data||_F (off-diagonal structure) ===")
    frob = {s: {} for s in results}
    for s, mats in results.items():
        Cd = mats["Data"]
        for m in order[1:]:
            frob[s][m] = float(np.linalg.norm(mats[m] - Cd))
            print(f"  {s:5s} {m:16s} {frob[s][m]:.3f}")

    # figure: 2 rows (surveys) x 5 cols (models), correlation matrices
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"axes.titlesize": 11, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    n = len(LABELS)
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 7.2), constrained_layout=True)
    for r, (sname, *_rest) in enumerate(surveys):
        mats = results[sname]
        for c, m in enumerate(order):
            ax = axes[r, c]
            im = ax.imshow(mats[m], cmap="RdBu_r", vmin=-1, vmax=1)
            ttl = m if m == "Data" else f"{m}\n" + r"$\|\Delta C\|_F=$" + f"{frob[sname][m]:.2f}"
            ax.set_title(ttl, fontsize=10)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            if r == 1:
                ax.set_xticklabels(LABELS, rotation=45, ha="right")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_yticklabels(LABELS)
            else:
                ax.set_yticklabels([])
        axes[r, 0].set_ylabel(sname, fontsize=14, labelpad=28, rotation=90)
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, location="right", pad=0.015)
    cbar.set_label(r"Correlation of $\log_{10}(L_{\rm line}/L_{\mathrm{H}\alpha})$", fontsize=12)
    fig.suptitle("Tier-1 benchmark: correlation structure of the 8 target log line ratios", fontsize=14)
    for ext in ("png", "pdf"):
        Path(REPO + "figs").mkdir(exist_ok=True)
        out = REPO + f"figs/tier1_corr_benchmark.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("Wrote:", out)

    # save frobenius table
    import csv
    with open(REPO + "docs/tier1_frobenius.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["survey", "model", "frobenius_corr_to_data"])
        for s in frob:
            for m, v in frob[s].items():
                w.writerow([s, m, f"{v:.4f}"])
    print("Wrote:", REPO + "docs/tier1_frobenius.csv")


if __name__ == "__main__":
    main()
