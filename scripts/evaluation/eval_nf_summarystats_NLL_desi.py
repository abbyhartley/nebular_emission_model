from __future__ import annotations

from pathlib import Path
import sys
import pickle
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow


LINE_ORDER = [
    ("Hbeta",    ["HBETA_FLUX", "H_BETA_FLUX"]),
    ("Hgamma",   ["HGAMMA_FLUX", "H_GAMMA_FLUX"]),
    ("NII6584",  ["NII_6584_FLUX"]),
    ("SII671x",  ["SII_6716_FLUX", "SII_6717_FLUX"]),
    ("SII6731",  ["SII_6731_FLUX"]),
    ("OII3726",  ["OII_3726_FLUX"]),
    ("OII3729",  ["OII_3729_FLUX"]),
    ("OIII5007", ["OIII_5007_FLUX"]),
]


def resolve_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Could not resolve any of: {candidates}")


def log10_luminosity_from_flux(z: np.ndarray, flux_cgs: np.ndarray) -> np.ndarray:
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    return np.log10(flux_cgs) + np.log10(4.0 * np.pi) + 2.0 * np.log10(dl_cm)


def prep_eval_dataframe(fits_path: Path, *, survey: str) -> tuple[pd.DataFrame, dict]:
    survey = survey.lower().strip()
    if survey not in {"sdss", "desi"}:
        raise ValueError("survey must be 'sdss' or 'desi'")

    t = Table.read(fits_path, hdu=1)
    names = [name for name in t.colnames if len(t[name].shape) <= 1]
    t = t[names]
    df = t.to_pandas()

    if survey == "sdss":
        z_col = "Z_1"
        ha_col = "H_ALPHA_FLUX"
        flux_scale = 1e-17
        require_ivar = False
        ivar_cols = {}
    else:
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

    resolved = {"z": z_col, "ha": ha_col, "lines": []}
    for _, aliases in LINE_ORDER:
        resolved["lines"].append(resolve_col(df, aliases))

    mask = np.isfinite(df[z_col].astype(float)) & np.isfinite(df[ha_col].astype(float))

    if require_ivar:
        if ivar_cols.get("ha") in df.columns:
            mask &= (df[ivar_cols["ha"]].astype(float) > 0)
        for iv in ivar_cols.get("lines", []):
            if iv in df.columns:
                mask &= (df[iv].astype(float) > 0)

    ha_cgs = df[ha_col].astype(float) * flux_scale
    mask &= np.isfinite(ha_cgs) & (ha_cgs > 0)

    for c in resolved["lines"]:
        x = df[c].astype(float) * flux_scale
        mask &= np.isfinite(x) & (x > 0)

    df = df.loc[mask].copy().reset_index(drop=True)

    df["LOG10_HA_FLUX"] = np.log10(df[ha_col].astype(float).to_numpy() * flux_scale)
    for c in resolved["lines"]:
        df[f"LOG10_{c}"] = np.log10(df[c].astype(float).to_numpy() * flux_scale)

    z = df[z_col].astype(float).to_numpy()
    ha_cgs = df[ha_col].astype(float).to_numpy() * flux_scale
    df["LOG_LHA"] = log10_luminosity_from_flux(z, ha_cgs)

    return df, resolved


def predict_logL_lines(
    flow,
    meta: dict,
    df_eval: pd.DataFrame,
    *,
    seed: int = 0,
    n_mc: int = 50,
    batch_size: int = 200_000,
) -> np.ndarray:
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]

    U_all = df_eval[[logm_col, loglha_col]].to_numpy(dtype=np.float32)
    logLHa_all = df_eval[loglha_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    X_mean, X_std = meta["X_mean"], meta["X_std"]
    U_mean, U_std = meta["U_mean"], meta["U_std"]

    n = len(df_eval)
    out = np.empty((n, len(meta["resolved"]["out_cols"])), dtype=np.float32)

    def sample_one(k, u):
        return flow.sample(k, sample_shape=(), condition=u)

    key = jr.key(seed + 999)

    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        U = U_all[lo:hi]
        logLHa = logLHa_all[lo:hi]
        Un = (U - U_mean) / U_std
        Un_jax = jnp.asarray(Un)
        nb = hi - lo

        ratio_samples = []
        for _ in range(int(n_mc)):
            key, subkey = jr.split(key)
            keys_i = jr.split(subkey, nb)
            Xn = jax.vmap(sample_one, in_axes=(0, 0))(keys_i, Un_jax)
            Xn = np.array(Xn)
            ratios = Xn * X_std + X_mean
            ratio_samples.append(ratios)

        ratios_mean = np.mean(np.stack(ratio_samples, axis=0), axis=0)
        out[lo:hi] = ratios_mean + logLHa

    return out


def true_logL_lines(df_eval: pd.DataFrame, resolved: dict) -> np.ndarray:
    logLHa = df_eval["LOG_LHA"].to_numpy(dtype=float).reshape(-1, 1)
    logFHa = df_eval["LOG10_HA_FLUX"].to_numpy(dtype=float).reshape(-1, 1)

    out = []
    for c in resolved["lines"]:
        logF = df_eval[f"LOG10_{c}"].to_numpy(dtype=float).reshape(-1, 1)
        out.append(logLHa + (logF - logFHa))
    return np.hstack(out)


def true_ratios(df_eval: pd.DataFrame, resolved: dict) -> np.ndarray:
    """Return true 8-D log10 flux ratios: logF(line) - logF(Ha)."""
    logFHa = df_eval["LOG10_HA_FLUX"].to_numpy(dtype=float).reshape(-1, 1)
    out = []
    for c in resolved["lines"]:
        logF = df_eval[f"LOG10_{c}"].to_numpy(dtype=float).reshape(-1, 1)
        out.append(logF - logFHa)
    return np.hstack(out)  # (N, 8)


def nll_bits_per_dim(flow, meta: dict, df_eval: pd.DataFrame, resolved: dict, *, batch_size: int = 200_000) -> float:
    """
    Compute mean negative log-likelihood in bits per dimension for the *true ratio targets*:
      X = logF(line)-logF(Ha)

    We evaluate:
      -log p(Xn | Un) / (D * ln 2)
    where Xn and Un are normalized using training means/stds.
    """
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]

    U = df_eval[[logm_col, loglha_col]].to_numpy(dtype=np.float32)
    Un = (U - meta["U_mean"]) / meta["U_std"]
    X = true_ratios(df_eval, resolved).astype(np.float32)
    Xn = (X - meta["X_mean"]) / meta["X_std"]

    Xn_j = jnp.asarray(Xn)
    Un_j = jnp.asarray(Un)

    D = Xn.shape[1]
    ln2 = np.log(2.0)

    n = Xn.shape[0]
    total = 0.0
    count = 0

    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        xb = Xn_j[lo:hi]
        ub = Un_j[lo:hi]
        # average negative log prob (nats)
        nll_nats = -jnp.mean(flow.log_prob(xb, condition=ub))
        total += float(nll_nats) * (hi - lo)
        count += (hi - lo)

    mean_nll_nats = total / count
    return float(mean_nll_nats / (D * ln2))


def rmse_dex(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def resid_summary(resid: np.ndarray) -> dict:
    p1, p16, p50, p84, p99 = np.percentile(resid, [1, 16, 50, 84, 99])
    frac_gt1 = float(np.mean(np.abs(resid) > 1.0))
    return dict(p1=float(p1), p16=float(p16), p50=float(p50), p84=float(p84), p99=float(p99), frac_abs_gt1=frac_gt1)


def eval_one(flow, meta: dict, fits_path: Path, survey: str, *, seed: int = 0, n_mc: int = 50) -> pd.DataFrame:
    df_eval, resolved = prep_eval_dataframe(fits_path, survey=survey)

    print(f"\n=== Evaluating on {survey.upper()} ===")
    print("Catalog:", fits_path)
    print("N after masks:", len(df_eval))
    print("Using n_mc =", n_mc)

    # NLL on true ratios (proper generative metric)
    bpd = nll_bits_per_dim(flow, meta, df_eval, resolved)
    print("NLL (bits/dim) on true ratios:", bpd)

    # Point-prediction metrics (from MC mean)
    logL_pred = predict_logL_lines(flow, meta, df_eval, seed=seed, n_mc=n_mc)
    logL_true = true_logL_lines(df_eval, resolved)

    header = (
        f"{'Line':10s}  {'RMSE':>7s}  {'Bias':>7s}  {'Scat':>7s}  {'rho':>7s}  "
        f"{'p01':>7s}  {'p16':>7s}  {'p50':>7s}  {'p84':>7s}  {'p99':>7s}  {'|r|>1':>7s}"
    )
    print(header)

    results = []
    for j, (linename, _) in enumerate(LINE_ORDER):
        y = logL_true[:, j]
        yhat = logL_pred[:, j]
        resid = yhat - y

        rmse = rmse_dex(yhat, y)
        bias = float(np.median(resid))
        p16, p84 = np.percentile(resid, [16, 84])
        scat = float(0.5 * (p84 - p16))
        rho, _ = spearmanr(y, yhat)

        rs = resid_summary(resid)
        results.append((
            linename, rmse, bias, scat, float(rho),
            rs["p1"], rs["p16"], rs["p50"], rs["p84"], rs["p99"], rs["frac_abs_gt1"]
        ))

        print(
            f"{linename:10s}  {rmse:7.3f}  {bias:7.3f}  {scat:7.3f}  {float(rho):7.3f}  "
            f"{rs['p1']:7.3f}  {rs['p16']:7.3f}  {rs['p50']:7.3f}  {rs['p84']:7.3f}  {rs['p99']:7.3f}  {rs['frac_abs_gt1']:7.3%}"
        )

    # Aggregate over all lines
    y_all = logL_true.reshape(-1)
    yhat_all = logL_pred.reshape(-1)
    resid_all = yhat_all - y_all
    rmse_all = rmse_dex(yhat_all, y_all)
    bias_all = float(np.median(resid_all))
    p16_all, p84_all = np.percentile(resid_all, [16, 84])
    scat_all = float(0.5 * (p84_all - p16_all))
    rho_all, _ = spearmanr(y_all, yhat_all)
    rs_all = resid_summary(resid_all)

    print("\nALL LINES concatenated:")
    print("RMSE(dex):   ", rmse_all)
    print("Bias(dex):   ", bias_all)
    print("Scatter(dex):", scat_all)
    print("Spearman rho:", float(rho_all))
    print("Residual percentiles p01/p16/p50/p84/p99:",
          rs_all["p1"], rs_all["p16"], rs_all["p50"], rs_all["p84"], rs_all["p99"])
    print("Frac |resid|>1 dex:", rs_all["frac_abs_gt1"])

    out = pd.DataFrame(
        results,
        columns=["line", "rmse_dex", "bias_med", "scatter_p84p16_half", "spearman_rho",
                 "res_p01", "res_p16", "res_p50", "res_p84", "res_p99", "frac_abs_resid_gt1dex"]
    )
    out["nll_bits_per_dim"] = bpd
    return out


def load_flow(flow_path: Path, meta: dict):
    xdim = len(meta["resolved"]["out_cols"])
    cond_dim = 2
    key = jr.key(int(meta.get("seed", 0)))
    template = block_neural_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=cond_dim,
    )
    return eqx.tree_deserialise_leaves(flow_path, template)


def main():
    REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
    sys.path.insert(0, str(REPO / "src"))

    # ---- choose which flow to evaluate by changing these two paths ----
    flow_path = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs.eqx")
    meta_path = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs_meta.pkl")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    print("Flow expects conditioning cols:",
          meta["resolved"]["logmstar_col"], meta["resolved"]["loglha_col"])
    flow = load_flow(flow_path, meta)

    sdss_fits = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")
    desi_fits = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

    N_MC = 50
    res_sdss = eval_one(flow, meta, sdss_fits, "sdss", seed=0, n_mc=N_MC)
    res_desi = eval_one(flow, meta, desi_fits, "desi", seed=1, n_mc=N_MC)

    out_csv = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_eval_metrics_with_nll.csv")
    pd.concat(
        [res_sdss.assign(dataset="sdss"), res_desi.assign(dataset="desi")],
        ignore_index=True,
    ).to_csv(out_csv, index=False)
    print("\nWrote metrics to:", out_csv)


if __name__ == "__main__":
    main()
