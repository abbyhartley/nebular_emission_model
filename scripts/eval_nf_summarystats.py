# Evaluates a trained flow (eqx + meta.pkl) on one or more FITS catalogs.
# Computes RMSE + Spearman for the 8 line LUMINOSITIES (log10 erg/s),
# where predictions are produced by sampling log(line/Ha) ratios from the flow
# and then adding LOG_LHA.
#
# Notes:
# - This assumes the flow was trained on LOG10 flux ratios: log10(F_line) - log10(F_Ha)
# - It assumes you can compute LOG_LHA from (z, Hα flux) in each eval catalog.
# - It requires positive fluxes for all 9 lines (Hα + 8) to define logs.
# - For DESI it also (optionally) enforces *_FLUX_IVAR > 0.

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import sys
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr


# -----------------------------
# Line definitions / ordering
# -----------------------------
LINE_ORDER = [
    ("Hbeta",   ["HBETA_FLUX", "H_BETA_FLUX"]),
    ("Hgamma",  ["HGAMMA_FLUX", "H_GAMMA_FLUX"]),
    ("NII6584", ["NII_6584_FLUX"]),
    ("SII671x", ["SII_6716_FLUX", "SII_6717_FLUX"]),   # DESI 6716, SDSS 6717
    ("SII6731", ["SII_6731_FLUX"]),
    ("OII3726", ["OII_3726_FLUX"]),
    ("OII3729", ["OII_3729_FLUX"]),
    ("OIII5007",["OIII_5007_FLUX"]),
]

def resolve_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Could not resolve any of: {candidates}")

def log10_luminosity_from_flux(z, flux_cgs):
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    return np.log10(flux_cgs) + np.log10(4.0 * np.pi) + 2.0 * np.log10(dl_cm)

def prep_eval_dataframe(fits_path, *, survey):
    """
    Returns a pandas DataFrame with columns:
      LOG_LHA, LOG10_HA_FLUX, and LOG10_<line_flux_col> for the 8 lines,
      plus LOGM_COLOR if present (not needed for metrics but helpful).
    Also returns the resolved column names.
    """
    t = Table.read(fits_path, hdu=1)
    names = [name for name in t.colnames if len(t[name].shape) <= 1]
    t = t[names]
    df = t.to_pandas()

    if survey.lower() == "sdss":
        z_col = "Z_1"
        ha_col = "H_ALPHA_FLUX"
        flux_scale = 1e-17  # based on your earlier check
        require_ivar = False
        ivar_cols = None
    elif survey.lower() == "desi":
        z_col = "Z"
        ha_col = "HALPHA_FLUX"
        flux_scale = 1e-17
        require_ivar = True
        ivar_cols = {
            "ha": "HALPHA_FLUX_IVAR",
            "lines": [
                "HBETA_FLUX_IVAR",
                "HGAMMA_FLUX_IVAR",
                "NII_6584_FLUX_IVAR",
                "SII_6716_FLUX_IVAR",
                "SII_6731_FLUX_IVAR",
                "OII_3726_FLUX_IVAR",
                "OII_3729_FLUX_IVAR",
                "OIII_5007_FLUX_IVAR",
            ],
        }
    else:
        raise ValueError("survey must be 'sdss' or 'desi'")

    # Resolve 8 line flux columns in this df
    resolved = {"z": z_col, "ha": ha_col, "lines": []}
    for _, aliases in LINE_ORDER:
        resolved["lines"].append(resolve_col(df, aliases))

    # Optional IVAR mask for DESI
    mask = np.isfinite(df[z_col].astype(float))
    mask &= np.isfinite(df[ha_col].astype(float))

    if require_ivar:
        if ivar_cols["ha"] in df.columns:
            mask &= (df[ivar_cols["ha"]].astype(float) > 0)
        for iv in ivar_cols["lines"]:
            if iv in df.columns:
                mask &= (df[iv].astype(float) > 0)

    # Convert to cgs
    ha_cgs = df[ha_col].astype(float) * flux_scale
    mask &= np.isfinite(ha_cgs) & (ha_cgs > 0)

    line_cgs = {}
    for c in resolved["lines"]:
        x = df[c].astype(float) * flux_scale
        line_cgs[c] = x
        mask &= np.isfinite(x) & (x > 0)

    df = df.loc[mask].copy().reset_index(drop=True)

    # Build log fluxes
    df["LOG10_HA_FLUX"] = np.log10(df[ha_col].astype(float).to_numpy() * flux_scale)
    for c in resolved["lines"]:
        df[f"LOG10_{c}"] = np.log10(df[c].astype(float).to_numpy() * flux_scale)

    # Build log L(Ha)
    z = df[z_col].astype(float).to_numpy()
    ha_cgs = df[ha_col].astype(float).to_numpy() * flux_scale
    df["LOG_LHA"] = log10_luminosity_from_flux(z, ha_cgs)

    return df, resolved


def predict_logL_lines(flow, meta, df_eval, resolved, *, seed=0, n_mc=1):
    """
    Predict log line luminosities for the 8 lines:
      logL_line = sample[log(line/Ha)] + LOG_LHA

    n_mc = number of Monte Carlo samples per object. If >1, we return the mean logL.
    """
    # Conditions
    logm_col = meta["resolved"]["logmstar_col"]  # e.g. LOGM_COLOR
    loglha_col = meta["resolved"]["loglha_col"]  # LOG_LHA

    if logm_col not in df_eval.columns:
        raise KeyError(f"Eval df missing required conditioning column {logm_col}.")
    if loglha_col not in df_eval.columns:
        raise KeyError(f"Eval df missing required conditioning column {loglha_col}.")

    U = df_eval[[logm_col, loglha_col]].to_numpy(dtype=np.float32)
    Un = (U - meta["U_mean"]) / meta["U_std"]
    Un_jax = jnp.asarray(Un)

    # sample ratios in normalized X-space
    X_mean, X_std = meta["X_mean"], meta["X_std"]
    n = len(df_eval)

    # We'll draw n_mc samples per object and average in log-space (cheap baseline).
    # (If you want proper mean in linear space, transform later.)
    key = jr.key(seed + 999)
    keys = jr.split(key, n * n_mc).reshape(n_mc, n, 2)  # 2 uint32s per key

    def sample_one(k, u):
        return flow.sample(k, sample_shape=(), condition=u)  # (8,)

    # vmap over objects; loop over n_mc to avoid huge memory
    ratio_samples = []
    for i in range(n_mc):
        k_i = keys[i]
        Xn = jax.vmap(sample_one, in_axes=(0, 0))(k_i, Un_jax)  # (n, 8)
        Xn = np.array(Xn)
        ratios = Xn * X_std + X_mean  # (n, 8) in log10(line/Ha)
        ratio_samples.append(ratios)

    ratios_mean = np.mean(np.stack(ratio_samples, axis=0), axis=0)  # (n, 8)

    logLHa = df_eval[loglha_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    logL_pred = ratios_mean + logLHa
    return logL_pred  # shape (n, 8)


def true_logL_lines(df_eval, resolved):
    """Compute true log line luminosities from fluxes + z for the same 8 lines."""
    logLHa = df_eval["LOG_LHA"].to_numpy(dtype=float).reshape(-1, 1)
    # true log flux ratios from the log flux columns we built
    logFHa = df_eval["LOG10_HA_FLUX"].to_numpy(dtype=float).reshape(-1, 1)

    logL_true = []
    for c in resolved["lines"]:
        logF = df_eval[f"LOG10_{c}"].to_numpy(dtype=float).reshape(-1, 1)
        # logL(line) = logL(Ha) + [logF(line) - logF(Ha)]
        logL_true.append(logLHa + (logF - logFHa))
    return np.hstack(logL_true)  # (n, 8)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def eval_one(flow, meta, fits_path, survey, *, seed=0, n_mc=1):
    df_eval, resolved = prep_eval_dataframe(fits_path, survey=survey)
    print(f"\n=== Evaluating on {survey.upper()} ===")
    print("Catalog:", fits_path)
    print("N after masks:", len(df_eval))

    logL_pred = predict_logL_lines(flow, meta, df_eval, resolved, seed=seed, n_mc=n_mc)
    logL_true = true_logL_lines(df_eval, resolved)

    # Metrics per line
    results = []
    for j, (linename, _) in enumerate(LINE_ORDER):
        y = logL_true[:, j]
        yhat = logL_pred[:, j]
        r = rmse(yhat, y)
        rho, _ = spearmanr(y, yhat)
        results.append((linename, r, float(rho)))

    # Print table
    print(f"{'Line':10s}  {'RMSE(dex)':>10s}  {'Spearman rho':>12s}")
    for linename, r, rho in results:
        print(f"{linename:10s}  {r:10.4f}  {rho:12.4f}")

    # Aggregate metric (optional): concatenate all lines
    y_all = logL_true.reshape(-1)
    yhat_all = logL_pred.reshape(-1)
    r_all = rmse(yhat_all, y_all)
    rho_all, _ = spearmanr(y_all, yhat_all)
    print("\nALL LINES concatenated:")
    print("RMSE(dex):", r_all)
    print("Spearman rho:", float(rho_all))

    return pd.DataFrame(results, columns=["line", "rmse_dex", "spearman_rho"])


def main():
    # ---- Load trained SDSS flow ----
    flow_path = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main.eqx")
    meta_path = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main_meta.pkl")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Recreate flow structure by loading directly into an existing object is tricky.
    # We assume you have a helper or you kept the flow class definition stable.
    #
    # In practice, easiest is to load flow by reconstructing it the same way as training.
    # But eqx.tree_deserialise_leaves needs a target object instance.
    #
    # If you still have the original training script, you can import the flow builder
    # OR save the whole res dict. For now, we do the standard pattern:
    from normflow.train_NF import train_line_ratio_flow  # for access to same flow type
    # Build a dummy flow by training on 1 row? No. Instead: you should store a pickled "flow template"
    # or rebuild using the same FlowJAX call.
    #
    # ---- Practical fix ----
    # We can load by first reconstructing a flow of the correct shape using meta dimensions.
    # X_dim = len(meta['resolved']['out_cols']); U_dim = 2
    from flowjax.distributions import Normal
    from flowjax.flows import block_neural_autoregressive_flow
    import jax.random as jr
    import jax.numpy as jnp

    xdim = len(meta["resolved"]["out_cols"])
    udim = len(meta["resolved"].get("cond_cols", [])) if "cond_cols" in meta["resolved"] else 2

    # Make a template flow and deserialize into it
    key = jr.key(meta.get("seed", 0))
    flow_template = block_neural_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=udim
    )
    flow = eqx.tree_deserialise_leaves(flow_path, flow_template)

    # ---- Evaluate on SDSS (in-survey) ----
    sdss_fits = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")
    desi_fits = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

    res_sdss = eval_one(flow, meta, sdss_fits, "sdss", seed=0, n_mc=1)
    res_desi = eval_one(flow, meta, desi_fits, "desi", seed=1, n_mc=1)

    # Save results
    out_csv = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_eval_metrics.csv")
    pd.concat(
        [res_sdss.assign(dataset="sdss"), res_desi.assign(dataset="desi")],
        ignore_index=True
    ).to_csv(out_csv, index=False)
    print("\nWrote metrics to:", out_csv)


if __name__ == "__main__":
    # Ensure repo src/ is importable like your other scripts
    REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
    sys.path.insert(0, str(REPO / "src"))
    main()
