# um_apply_desi_nf_make_oii_lf.py
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow


# -----------------------
# Inputs
# -----------------------
UM_PARQUET = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/UM_comp/um_a0.911185_z0p1_conditions_logMge8.3.parquet")

FLOW_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs.eqx")
META_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs_meta.pkl")

OUT_PNG = "um_oii_lf_from_desi_nf.png"
OUT_CSV = "um_oii_lf_from_desi_nf.csv"

# Sampling config
N_MC = 20            # MC mean samples per galaxy (20 is a good start)
BATCH_SIZE = 50_000  # adjust for memory/speed
SEED = 0

# LF bins
BINS = np.arange(38.0, 44.6, 0.2)  # log10 L[erg/s]
# If you know the UM effective comoving volume in Mpc^3, put it here.
# If None, the script reports counts per dex (unnormalized).
VOLUME_MPC3 = 2.8e6


# -----------------------
# Helpers
# -----------------------
def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    key = jr.key(int(meta.get("seed", 0)))
    template = block_neural_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=2,
    )
    return eqx.tree_deserialise_leaves(flow_path, template)

def find_outcol_index(out_cols, substrs):
    for i, name in enumerate(out_cols):
        if any(s in name for s in substrs):
            return i
    raise KeyError(f"Could not find any of {substrs} in out_cols.")

def sample_ratios_mcmean(flow, meta, df, *, seed=0, n_mc=20, batch_size=50_000):
    """
    Returns MC-mean of ratios in *original* (unnormalized) space:
      ratios[j] = log10(F_line / F_Ha) for each out_col.
    Shape: (N, xdim)
    """
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]

    U_all = df[[logm_col, loglha_col]].to_numpy(np.float32)
    Un_all = (U_all - meta["U_mean"]) / meta["U_std"]
    Un_all = jnp.asarray(Un_all)

    X_mean, X_std = meta["X_mean"], meta["X_std"]
    n = len(df)
    xdim = len(meta["resolved"]["out_cols"])
    out = np.empty((n, xdim), dtype=np.float32)

    def sample_one(k, u):
        return flow.sample(k, sample_shape=(), condition=u)

    key = jr.key(seed + 999)

    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        Un = Un_all[lo:hi]
        nb = hi - lo

        rs = []
        for _ in range(int(n_mc)):
            key, subkey = jr.split(key)
            keys = jr.split(subkey, nb)
            Xn = jax.vmap(sample_one, in_axes=(0, 0))(keys, Un)
            Xn = np.array(Xn)
            rs.append(Xn * X_std + X_mean)

        out[lo:hi] = np.mean(np.stack(rs, axis=0), axis=0)

        print(f"Sampled {hi:,}/{n:,} galaxies")

    return out

def make_lf(logL, bins, volume_mpc3=None):
    logL = np.asarray(logL, float)
    logL = logL[np.isfinite(logL)]
    hist, edges = np.histogram(logL, bins=bins)
    dlogL = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    if volume_mpc3 is None:
        phi = hist / dlogL
        ylab = r"$N\,\mathrm{dex^{-1}}$"
    else:
        phi = hist / (volume_mpc3 * dlogL)
        ylab = r"$\phi(L)\;[\mathrm{Mpc^{-3}\,dex^{-1}}]$"
    return centers, phi, hist, ylab


# -----------------------
# Main
# -----------------------
def main():
    # Load UM conditioning set
    um = pd.read_parquet(UM_PARQUET)
    for c in ["LOGM_COLOR", "LOG_LHA"]:
        if c not in um.columns:
            raise KeyError(f"UM parquet missing required column {c}")

    # Load DESI flow + meta
    with open(META_DESI, "rb") as f:
        meta = pickle.load(f)
    flow = load_flow(FLOW_DESI, meta)

    # Rename columns if meta expects different names (usually it expects LOGM_COLOR/LOG_LHA already)
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]
    if logm_col not in um.columns or loglha_col not in um.columns:
        raise KeyError(f"UM df must contain columns {logm_col} and {loglha_col} for conditioning.")

    # Sample ratios and compute OII total
    out_cols = meta["resolved"]["out_cols"]
    i26 = find_outcol_index(out_cols, ["OII_3726"])
    i29 = find_outcol_index(out_cols, ["OII_3729"])

    ratios = sample_ratios_mcmean(flow, meta, um, seed=SEED, n_mc=N_MC, batch_size=BATCH_SIZE)
    r26 = ratios[:, i26].astype(float)
    r29 = ratios[:, i29].astype(float)

    logLHa = um[loglha_col].to_numpy(float)
    logOII_Ha = np.log10(np.power(10.0, r26) + np.power(10.0, r29))
    logLoii = logLHa + logOII_Ha

    # LF
    centers, phi, hist, ylab = make_lf(logLoii, BINS, volume_mpc3=VOLUME_MPC3)

    # Save LF data
    df_lf = pd.DataFrame({
        "logL_center": centers,
        "count": hist,
        "phi_or_counts_per_dex": phi,
    })
    df_lf.to_csv(OUT_CSV, index=False)
    print("Wrote:", OUT_CSV)

    # Plot
    plt.style.use(["science", "no-latex"])
    plt.figure(figsize=(6.2, 4.6))
    m = hist > 0
    plt.semilogy(centers[m], phi[m], drawstyle="steps-mid", lw=2.0)
    plt.xlabel(r"$\log_{10} L_{\mathrm{[OII]}}\, [\mathrm{erg\,s^{-1}}]$")
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=250)
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
