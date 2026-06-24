"""
Corrected NF evaluation metrics (review items 2.5a + 2.5b).

For all four train->test combinations (DESI/SDSS flow x DESI/SDSS test set):

  2.5a  Report Spearman rho AND RMSE on the RATIOS (log10 L_line/L_Ha) -- the
        actual model output -- in addition to absolute log-luminosities. The
        absolute-luminosity rho was inflated because both prediction and truth
        share the conditioning input LOG_LHA.

  2.5b  - Add NMAD (1.4826 * MAD) and median residual alongside RMSE (outlier-robust).
        - Report a standardization-INDEPENDENT NLL ("raw" bits/dim) so the two flows
          are comparable on a common scale:
              NLL_raw_nats = NLL_norm_nats + sum_j log(X_std_j)
          (change of variables for the per-feature standardization x = xn*std + mean).
          The original per-flow "norm" bits/dim is also reported for continuity.

Outputs (repo root):
  nf_eval_pointmetrics_corrected.csv   per-line, both spaces (abs|ratio)
  nf_eval_nll_corrected.csv            per-combo NLL (norm + raw)
"""
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

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model")
BASE = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs")

FLOWS = {
    "desi": (REPO / "nf_desi_bgs.eqx",  REPO / "nf_desi_bgs_meta.pkl"),
    "sdss": (REPO / "nf_sdss_main.eqx", REPO / "nf_sdss_main_meta.pkl"),
}
FITS = {
    "desi": BASE / "DESI_BGS_training_data.fits",
    "sdss": BASE / "SDSS_main_training_data.fits",
}

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

N_MC = 50


def resolve_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"Could not resolve any of: {candidates}")


def log10_lum_from_flux(z, flux_cgs):
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    return np.log10(flux_cgs) + np.log10(4.0 * np.pi) + 2.0 * np.log10(dl_cm)


def prep_eval_dataframe(fits_path, *, survey):
    survey = survey.lower().strip()
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    df = t[names].to_pandas()

    if survey == "sdss":
        z_col, ha_col, require_ivar = "Z_1", "H_ALPHA_FLUX", False
        ivar_cols = {}
    else:
        z_col, ha_col, require_ivar = "Z", "HALPHA_FLUX", True
        ivar_cols = {
            "ha": "HALPHA_FLUX_IVAR",
            "lines": ["HBETA_FLUX_IVAR", "HGAMMA_FLUX_IVAR", "NII_6584_FLUX_IVAR",
                      "SII_6716_FLUX_IVAR", "SII_6731_FLUX_IVAR", "OII_3726_FLUX_IVAR",
                      "OII_3729_FLUX_IVAR", "OIII_5007_FLUX_IVAR"],
        }

    resolved = {"z": z_col, "ha": ha_col, "lines": []}
    for _, aliases in LINE_ORDER:
        resolved["lines"].append(resolve_col(df, aliases))

    flux_scale = 1e-17
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
    df["LOG_LHA"] = log10_lum_from_flux(z, df[ha_col].astype(float).to_numpy() * flux_scale)
    return df, resolved


def true_ratios(df, resolved):
    logFHa = df["LOG10_HA_FLUX"].to_numpy(float).reshape(-1, 1)
    out = [df[f"LOG10_{c}"].to_numpy(float).reshape(-1, 1) - logFHa for c in resolved["lines"]]
    return np.hstack(out)  # (N,8) true log10 ratios


def predict_mean_ratios(flow, meta, df, *, seed, n_mc=N_MC, batch_size=200_000):
    """Return MC-mean predicted log10 ratios (N,8)."""
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]
    U_all = df[[logm_col, loglha_col]].to_numpy(np.float32)
    X_mean, X_std = meta["X_mean"], meta["X_std"]
    U_mean, U_std = meta["U_mean"], meta["U_std"]
    n = len(df)
    out = np.empty((n, len(meta["resolved"]["out_cols"])), dtype=np.float32)

    def sample_one(k, u):
        return flow.sample(k, sample_shape=(), condition=u)

    key = jr.key(seed + 999)
    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        Un = (U_all[lo:hi] - U_mean) / U_std
        Un_j = jnp.asarray(Un)
        nb = hi - lo
        acc = []
        for _ in range(int(n_mc)):
            key, sub = jr.split(key)
            keys_i = jr.split(sub, nb)
            Xn = np.array(jax.vmap(sample_one, in_axes=(0, 0))(keys_i, Un_j))
            acc.append(Xn * X_std + X_mean)
        out[lo:hi] = np.mean(np.stack(acc, 0), 0)
    return out


def nll_bits(flow, meta, df, resolved, *, batch_size=200_000):
    """Return (norm_bits_per_dim, raw_bits_per_dim)."""
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]
    U = df[[logm_col, loglha_col]].to_numpy(np.float32)
    Un = jnp.asarray((U - meta["U_mean"]) / meta["U_std"])
    X = true_ratios(df, resolved).astype(np.float32)
    Xn = jnp.asarray((X - meta["X_mean"]) / meta["X_std"])

    D = X.shape[1]
    ln2 = np.log(2.0)
    n = X.shape[0]
    total = 0.0
    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        nll_nats = -jnp.mean(flow.log_prob(Xn[lo:hi], condition=Un[lo:hi]))
        total += float(nll_nats) * (hi - lo)
    nll_norm_nats = total / n
    # change of variables: density on raw ratios is standardization-independent
    nll_raw_nats = nll_norm_nats + float(np.sum(np.log(np.asarray(meta["X_std"]))))
    return nll_norm_nats / (D * ln2), nll_raw_nats / (D * ln2)


def metrics(yhat, y):
    resid = yhat - y
    med = np.median(resid)
    p1, p16, p50, p84, p99 = np.percentile(resid, [1, 16, 50, 84, 99])
    rho, _ = spearmanr(y, yhat)
    return dict(
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        bias_med=float(med),
        scatter=float(0.5 * (p84 - p16)),
        nmad=float(1.4826 * np.median(np.abs(resid - med))),
        spearman_rho=float(rho),
        res_p01=float(p1), res_p16=float(p16), res_p50=float(p50),
        res_p84=float(p84), res_p99=float(p99),
        frac_abs_resid_gt1dex=float(np.mean(np.abs(resid) > 1.0)),
    )


def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    template = block_neural_autoregressive_flow(
        key=jr.key(int(meta.get("seed", 0))),
        base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=2,
    )
    return eqx.tree_deserialise_leaves(flow_path, template)


def main():
    sys.path.insert(0, str(REPO / "src"))
    metas = {k: pickle.load(open(v[1], "rb")) for k, v in FLOWS.items()}
    flows = {k: load_flow(FLOWS[k][0], metas[k]) for k in FLOWS}

    # prep each test set once (reused across flows)
    evalsets = {s: prep_eval_dataframe(FITS[s], survey=s) for s in FITS}

    point_rows, nll_rows = [], []
    seed = 0
    for train in ("sdss", "desi"):
        for test in ("sdss", "desi"):
            seed += 1
            df, resolved = evalsets[test]
            meta, flow = metas[train], flows[train]
            print(f"\n=== {train.upper()} -> {test.upper()}  (N={len(df):,}, n_mc={N_MC}) ===", flush=True)

            logLHa = df["LOG_LHA"].to_numpy(float).reshape(-1, 1)
            ratio_pred = predict_mean_ratios(flow, meta, df, seed=seed)   # (N,8)
            ratio_true = true_ratios(df, resolved)                        # (N,8)
            absL_pred = ratio_pred + logLHa
            absL_true = ratio_true + logLHa

            nll_norm, nll_raw = nll_bits(flow, meta, df, resolved)
            nll_rows.append(dict(train=train, test=test, n_obj=len(df),
                                 nll_bits_norm=nll_norm, nll_bits_raw=nll_raw))
            print(f"  NLL bits/dim  norm={nll_norm:.3f}  raw(common-scale)={nll_raw:.3f}")

            for j, (line, _) in enumerate(LINE_ORDER):
                for space, yh, yt in (("abs", absL_pred[:, j], absL_true[:, j]),
                                      ("ratio", ratio_pred[:, j], ratio_true[:, j])):
                    row = dict(train=train, test=test, line=line, space=space, n_obj=len(df))
                    row.update(metrics(yh, yt))
                    point_rows.append(row)

            # aggregate over all 8 lines, both spaces
            for space, yh, yt in (("abs", absL_pred.reshape(-1), absL_true.reshape(-1)),
                                  ("ratio", ratio_pred.reshape(-1), ratio_true.reshape(-1))):
                row = dict(train=train, test=test, line="ALL", space=space, n_obj=len(df))
                row.update(metrics(yh, yt))
                point_rows.append(row)
                if space == "ratio":
                    print(f"  ALL ratio: rho={row['spearman_rho']:.3f} "
                          f"RMSE={row['rmse']:.3f} NMAD={row['nmad']:.3f}")

    pm = pd.DataFrame(point_rows)
    nl = pd.DataFrame(nll_rows)
    pm.to_csv(REPO / "nf_eval_pointmetrics_corrected.csv", index=False)
    nl.to_csv(REPO / "nf_eval_nll_corrected.csv", index=False)
    print("\nWrote nf_eval_pointmetrics_corrected.csv and nf_eval_nll_corrected.csv")


if __name__ == "__main__":
    main()
