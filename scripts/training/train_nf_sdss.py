# train_nf_sdss_loglum.py
from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
import sys

# -----------------------------
# Repo import (matches your pattern)
# -----------------------------
REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
sys.path.insert(0, str(REPO / "src"))

from normflow.train_NF import train_line_ratio_flow

# -----------------------------
# I/O paths
# -----------------------------
infile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")

out_flow = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main.eqx")
out_meta = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main_meta.pkl")

# -----------------------------
# Columns
# -----------------------------
z_col = "Z_1"
logm_col = "LOGM_COLOR"

# SDSS/MPA-JHU emission-line fluxes are LINEAR and (in your case) in units of 1e-17 erg/s/cm^2
FLUX_SCALE = 1e-17

ha_flux_col = "H_ALPHA_FLUX"
line_flux_cols = [
    "H_BETA_FLUX",
    "H_GAMMA_FLUX",
    "NII_6584_FLUX",
    "SII_6717_FLUX",   # SDSS has 6717 not 6716
    "SII_6731_FLUX",
    "OII_3726_FLUX",
    "OII_3729_FLUX",
    "OIII_5007_FLUX",
]

# Training hyperparams
SEED = 0
EPOCHS = 200
BATCH = 2048
LR = 3e-4
CLIP = 1.0


def log10_luminosity_from_flux(z, flux_cgs):
    """flux_cgs in erg/s/cm^2 (linear). Returns log10 L in erg/s."""
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    return np.log10(flux_cgs) + np.log10(4.0 * np.pi) + 2.0 * np.log10(dl_cm)


def quick_sample_sanity(flow, meta, df_train, *, n_cond=3000, seed=12345):
    X_mean = meta["X_mean"]; X_std = meta["X_std"]
    U_mean = meta["U_mean"]; U_std = meta["U_std"]

    logm = df_train[meta["resolved"]["logmstar_col"]].to_numpy(dtype=np.float32)
    loglha = df_train[meta["resolved"]["loglha_col"]].to_numpy(dtype=np.float32)

    n = len(df_train)
    n_cond = min(n_cond, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_cond, replace=False)

    U = np.stack([logm[idx], loglha[idx]], axis=1)
    Un = (U - U_mean) / U_std
    Un_jax = jnp.asarray(Un)

    key = jr.key(seed + 999)
    keys = jr.split(key, n_cond)

    def sample_one(k, u):
        return flow.sample(k, sample_shape=(), condition=u)

    Xn_samp = jax.vmap(sample_one, in_axes=(0, 0))(keys, Un_jax)
    Xn_samp = np.array(Xn_samp)

    ratios = Xn_samp * X_std + X_mean                 # log10(line) - log10(Ha)
    logLHa = loglha[idx].reshape(-1, 1)
    logL_lines = ratios + logLHa

    print("\n================ QUICK SAMPLE CHECK ================")
    print(f"Sampled {n_cond} conditions.")
    print("LOG_LHA sampled range:", float(np.min(loglha[idx])), float(np.max(loglha[idx])))

    line_names = ["Hbeta", "Hgamma", "NII6584", "SII6717", "SII6731", "OII3726", "OII3729", "OIII5007"]
    for j, nm in enumerate(line_names):
        arr = logL_lines[:, j]
        p16, p50, p84 = np.percentile(arr, [16, 50, 84])
        print(f"{nm:8s} logL  min/16/50/84/max = {arr.min():.2f}  {p16:.2f}  {p50:.2f}  {p84:.2f}  {arr.max():.2f}")
    print("====================================================\n")


def main():
    # -----------------------------
    # Load FITS safely (drop multidim columns before pandas)
    # -----------------------------
    t = Table.read(infile, hdu=1)
    names = [name for name in t.colnames if len(t[name].shape) <= 1]
    t = t[names]
    df = t.to_pandas()

    # -----------------------------
    # Compute log10(flux) for Hα and 8 lines (mask non-positive)
    # -----------------------------
    # Convert to cgs
    df["_HA_FLUX_CGS"] = df[ha_flux_col].astype(float) * FLUX_SCALE

    for c in line_flux_cols:
        df[f"_{c}_CGS"] = df[c].astype(float) * FLUX_SCALE

    # Mask: require positive fluxes for log
    # (Strict option: require all 9 lines > 0; this matches your current "drop NaNs" approach.)
    pos_mask = np.isfinite(df[z_col].astype(float)) & np.isfinite(df[logm_col].astype(float))
    pos_mask &= np.isfinite(df["_HA_FLUX_CGS"]) & (df["_HA_FLUX_CGS"] > 0)

    for c in line_flux_cols:
        pos_mask &= np.isfinite(df[f"_{c}_CGS"]) & (df[f"_{c}_CGS"] > 0)

    df = df.loc[pos_mask].copy()
    df.reset_index(drop=True, inplace=True)

    # Now safe logs
    df["LOG10_HA_FLUX"] = np.log10(df["_HA_FLUX_CGS"].to_numpy())

    for c in line_flux_cols:
        df[f"LOG10_{c}"] = np.log10(df[f"_{c}_CGS"].to_numpy())

    # -----------------------------
    # Compute log10 L(Hα) in erg/s (conditioner)
    # -----------------------------
    z = df[z_col].to_numpy(dtype=float)
    ha_flux_cgs = df["_HA_FLUX_CGS"].to_numpy(dtype=float)
    df["LOG_LHA"] = log10_luminosity_from_flux(z, ha_flux_cgs)

    print("Rows after requiring all 9 lines > 0:", len(df))
    print("LOG_LHA range:", float(df["LOG_LHA"].min()), float(df["LOG_LHA"].max()))
    print("LOGM_COLOR range:", float(df[logm_col].min()), float(df[logm_col].max()))

    # -----------------------------
    # Train NF
    # Train on ratio targets using LOG10(line flux) - LOG10(Hα flux)
    # Condition on (LOGM_COLOR, LOG_LHA)
    # -----------------------------
    # Provide aliases that point to the LOG10_ columns we just created
    line_aliases_log = {
        "hbeta":   ["LOG10_H_BETA_FLUX"],
        "hgamma":  ["LOG10_H_GAMMA_FLUX"],
        "nii6584": ["LOG10_NII_6584_FLUX"],
        "sii6716": ["LOG10_SII_6717_FLUX"],   # mapped into "sii6716" key
        "sii6731": ["LOG10_SII_6731_FLUX"],
        "oii3726": ["LOG10_OII_3726_FLUX"],
        "oii3729": ["LOG10_OII_3729_FLUX"],
        "oiii5007":["LOG10_OIII_5007_FLUX"],
    }

    res = train_line_ratio_flow(
        df,
        logmstar_col=logm_col,
        loglha_col="LOG_LHA",
        logha_col="LOG10_HA_FLUX",
        line_aliases=line_aliases_log,
        use_ratios_to_ha=True,
        seed=SEED,
        batch_size=BATCH,
        epochs=EPOCHS,
        lr=LR,
        clip=CLIP,
    )

    flow = res["flow"]
    meta = res["meta"]
    meta["extra"] = dict(
        infile=str(infile),
        z_col=z_col,
        logm_col=logm_col,
        flux_scale=FLUX_SCALE,
        ha_flux_col=ha_flux_col,
        line_flux_cols=line_flux_cols,
        derived_cols=dict(
            loglha="LOG_LHA",
            logha_flux="LOG10_HA_FLUX",
            log_line_flux_cols=[f"LOG10_{c}" for c in line_flux_cols],
        ),
        n_train=len(res["df_train"]),
        note="Training used log10 flux ratios; LOG_LHA computed from Hα flux and z.",
    )

    # -----------------------------
    # Quick sampling sanity check
    # -----------------------------
    quick_sample_sanity(flow, meta, res["df_train"], n_cond=3000, seed=SEED + 2024)

    # -----------------------------
    # Save flow + meta
    # -----------------------------
    eqx.tree_serialise_leaves(out_flow, flow)
    with open(out_meta, "wb") as f:
        pickle.dump(meta, f)

    print("Saved flow to:", out_flow)
    print("Saved meta to:", out_meta)


if __name__ == "__main__":
    main()
