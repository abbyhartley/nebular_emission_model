# cross_survey_transfer_rows_cmasher_v4_fullrange.py
#
# Updates vs v3:
# - larger fonts throughout
# - 1:1 line is thicker, dotted, red
# - axis limits per column are computed from the FULL min/max across BOTH rows (obs+pred),
#   with a small pad, so no data are clipped (not percentile-based)
# - still enforces a faint-end floor (optional) for readability

from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import cmasher as cmr

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow


plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

FLOW_SDSS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main.eqx")
META_SDSS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main_meta.pkl")

FLOW_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs.eqx")
META_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs_meta.pkl")

SDSS_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")
DESI_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

FLUX_SCALE = 1e-17
N_MC = 50
MINCNT = 5
SEED = 0

CMAP = cmr.bubblegum

SDSS_COLS = dict(z="Z_1", ha="H_ALPHA_FLUX", logm="LOGM_COLOR")
DESI_COLS = dict(z="Z", ha="HALPHA_FLUX", ha_ivar="HALPHA_FLUX_IVAR", logm="LOGM_COLOR")

PLOT_LINES = [
    ("H$\\beta$",   "HBETA_FLUX",      "H_BETA_FLUX",      "H_BETA",    "HBETA"),
    ("[NII]6584",   "NII_6584_FLUX",   "NII_6584_FLUX",    "NII_6584",  "NII_6584"),
    ("[OII]3727",   "OII_TOTAL",       "OII_TOTAL",        "OII_TOTAL", "OII_TOTAL"),
    ("[OIII]5007",  "OIII_5007_FLUX",  "OIII_5007_FLUX",   "OIII_5007", "OIII_5007"),
]


def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    key = jr.key(int(meta.get("seed", 0)))
    template = block_neural_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=2,
    )
    return eqx.tree_deserialise_leaves(flow_path, template)


def log10_lum_from_flux(z, flux_1e17):
    z = np.asarray(z, float)
    f = np.asarray(flux_1e17, float) * FLUX_SCALE
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    return np.log10(f) + np.log10(4*np.pi) + 2*np.log10(dl_cm)


def per_line_stats(y_true, y_pred):
    resid = y_pred - y_true
    rmse = float(np.sqrt(np.mean(resid**2)))
    p16, p84 = np.percentile(resid, [16, 84])
    scat = float(0.5 * (p84 - p16))
    rho, _ = spearmanr(y_true, y_pred)
    return rmse, scat, float(rho)


def prep_df(fits_path, survey):
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    t = t[names]
    df = t.to_pandas()

    if survey == "sdss":
        z = df[SDSS_COLS["z"]].to_numpy(float)
        ha = df[SDSS_COLS["ha"]].to_numpy(float)
        m = df[SDSS_COLS["logm"]].to_numpy(float)
        mask = np.isfinite(z) & np.isfinite(ha) & (ha > 0) & np.isfinite(m)

        required = ["H_BETA_FLUX", "NII_6584_FLUX", "OIII_5007_FLUX", "OII_3726_FLUX", "OII_3729_FLUX"]
        for c in required:
            x = df[c].to_numpy(float)
            mask &= np.isfinite(x) & (x > 0)

        df = df.loc[mask].copy().reset_index(drop=True)
        df["LOG_LHA"] = log10_lum_from_flux(df[SDSS_COLS["z"]].to_numpy(float),
                                            df[SDSS_COLS["ha"]].to_numpy(float))
        return df

    if survey == "desi":
        z = df[DESI_COLS["z"]].to_numpy(float)
        ha = df[DESI_COLS["ha"]].to_numpy(float)
        m = df[DESI_COLS["logm"]].to_numpy(float)
        mask = np.isfinite(z) & np.isfinite(ha) & (ha > 0) & np.isfinite(m)

        if DESI_COLS["ha_ivar"] in df.columns:
            mask &= (df[DESI_COLS["ha_ivar"]].to_numpy(float) > 0)

        required = ["HBETA_FLUX", "NII_6584_FLUX", "OIII_5007_FLUX", "OII_3726_FLUX", "OII_3729_FLUX"]
        for c in required:
            x = df[c].to_numpy(float)
            mask &= np.isfinite(x) & (x > 0)

        df = df.loc[mask].copy().reset_index(drop=True)
        df["LOG_LHA"] = log10_lum_from_flux(df[DESI_COLS["z"]].to_numpy(float),
                                            df[DESI_COLS["ha"]].to_numpy(float))
        return df

    raise ValueError("survey must be 'sdss' or 'desi'")


def sample_ratios(flow, meta, df, *, seed=0, n_mc=50, batch_size=200_000):
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]

    U_all = df[[logm_col, loglha_col]].to_numpy(np.float32)
    Un_all = (U_all - meta["U_mean"]) / meta["U_std"]
    Un_all = jnp.asarray(Un_all)

    X_mean, X_std = meta["X_mean"], meta["X_std"]
    n = len(df)
    out = np.empty((n, len(meta["resolved"]["out_cols"])), dtype=np.float32)

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

    return out


def outcol_index(meta, substr):
    out_cols = meta["resolved"]["out_cols"]
    for i, name in enumerate(out_cols):
        if substr in name:
            return i
    raise KeyError(f"Could not find '{substr}' in out_cols.")


def pred_logL_single(df, meta, ratios8, substr):
    idx = outcol_index(meta, substr)
    return df["LOG_LHA"].to_numpy(float) + ratios8[:, idx].astype(float)


def pred_logL_oii_total(df, meta, ratios8):
    out_cols = meta["resolved"]["out_cols"]

    def find(substrs):
        for i, name in enumerate(out_cols):
            if any(s in name for s in substrs):
                return i
        raise KeyError(f"Could not find any of {substrs} in out_cols.")

    i26 = find(["OII_3726"])
    i29 = find(["OII_3729"])
    r26 = ratios8[:, i26].astype(float)
    r29 = ratios8[:, i29].astype(float)
    log_ratio = np.log10(np.power(10.0, r26) + np.power(10.0, r29))
    return df["LOG_LHA"].to_numpy(float) + log_ratio


def true_logL(df, survey, desi_col, sdss_col):
    if desi_col == "OII_TOTAL":
        f = df["OII_3726_FLUX"].to_numpy(float) + df["OII_3729_FLUX"].to_numpy(float)
        if survey == "desi":
            return log10_lum_from_flux(df[DESI_COLS["z"]].to_numpy(float), f)
        return log10_lum_from_flux(df[SDSS_COLS["z"]].to_numpy(float), f)

    if survey == "desi":
        return log10_lum_from_flux(df[DESI_COLS["z"]].to_numpy(float), df[desi_col].to_numpy(float))
    return log10_lum_from_flux(df[SDSS_COLS["z"]].to_numpy(float), df[sdss_col].to_numpy(float))


def hex_panel(ax, x, y, *, lo, hi, mincnt=5):
    hb = ax.hexbin(
        x, y,
        gridsize=70,
        extent=(lo, hi, lo, hi),
        bins="log",
        mincnt=mincnt,
        cmap=CMAP,
    )
    ax.plot([lo, hi], [lo, hi], color="black", lw=3.0, ls=":", alpha=0.95)  # thicker dotted red 1:1
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    return hb


def _finite_minmax(*arrays):
    v = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    return float(np.min(v)), float(np.max(v))


def main():
    with open(META_SDSS, "rb") as f:
        meta_sdss = pickle.load(f)
    with open(META_DESI, "rb") as f:
        meta_desi = pickle.load(f)

    flow_sdss = load_flow(FLOW_SDSS, meta_sdss)
    flow_desi = load_flow(FLOW_DESI, meta_desi)

    df_desi = prep_df(DESI_FITS, "desi")
    df_sdss = prep_df(SDSS_FITS, "sdss")

    ratios_sdss_on_desi = sample_ratios(flow_sdss, meta_sdss, df_desi, seed=SEED + 1, n_mc=N_MC)
    ratios_desi_on_sdss = sample_ratios(flow_desi, meta_desi, df_sdss, seed=SEED + 2, n_mc=N_MC)

    # ---- Precompute per-column axis limits using FULL min/max across BOTH rows ----
    col_limits = []
    PAD = 0.05   # in dex, small margin
    LO_FLOOR = 38.9  # optional common faint-end floor

    for (label, desi_col, sdss_col, substr_sdss, substr_desi) in PLOT_LINES:
        # Top (SDSS->DESI)
        yobs_top = true_logL(df_desi, "desi", desi_col, sdss_col)
        ypred_top = pred_logL_oii_total(df_desi, meta_sdss, ratios_sdss_on_desi) if desi_col == "OII_TOTAL" else \
                    pred_logL_single(df_desi, meta_sdss, ratios_sdss_on_desi, substr_sdss)

        # Bottom (DESI->SDSS)
        yobs_bot = true_logL(df_sdss, "sdss", desi_col, sdss_col)
        ypred_bot = pred_logL_oii_total(df_sdss, meta_desi, ratios_desi_on_sdss) if desi_col == "OII_TOTAL" else \
                    pred_logL_single(df_sdss, meta_desi, ratios_desi_on_sdss, substr_desi)

        lo, hi = _finite_minmax(yobs_top, ypred_top, yobs_bot, ypred_bot)
        lo = min(lo - PAD, LO_FLOOR)   # expand down and enforce floor
        hi = hi + PAD                  # expand up
        col_limits.append((float(lo), float(hi)))

    # ---- Plot ----
    ncols = len(PLOT_LINES)
    fig, axes = plt.subplots(
        2, ncols,
        figsize=(3.2 * ncols + 1.2, 6.9),
        sharex="col",
        sharey="col",
        constrained_layout=True,
    )

    hb_last = None

    for j, (label, desi_col, sdss_col, substr_sdss, substr_desi) in enumerate(PLOT_LINES):
        lo, hi = col_limits[j]

        # Top: SDSS -> DESI
        yobs = true_logL(df_desi, "desi", desi_col, sdss_col)
        ypred = pred_logL_oii_total(df_desi, meta_sdss, ratios_sdss_on_desi) if desi_col == "OII_TOTAL" else \
                pred_logL_single(df_desi, meta_sdss, ratios_sdss_on_desi, substr_sdss)

        rmse, scat, rho = per_line_stats(yobs, ypred)
        hb_last = hex_panel(axes[0, j], yobs, ypred, lo=lo, hi=hi, mincnt=MINCNT)
        axes[0, j].set_title(label)
        axes[0, j].text(
            0.03, 0.97,
            f"RMSE={rmse:.3f}\nscat={scat:.3f}\n$\\rho$={rho:.3f}",
            transform=axes[0, j].transAxes,
            va="top", ha="left", fontsize=14,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round,pad=0.2"),
        )

        # Bottom: DESI -> SDSS
        yobs2 = true_logL(df_sdss, "sdss", desi_col, sdss_col)
        ypred2 = pred_logL_oii_total(df_sdss, meta_desi, ratios_desi_on_sdss) if desi_col == "OII_TOTAL" else \
                 pred_logL_single(df_sdss, meta_desi, ratios_desi_on_sdss, substr_desi)

        rmse2, scat2, rho2 = per_line_stats(yobs2, ypred2)
        hb_last = hex_panel(axes[1, j], yobs2, ypred2, lo=lo, hi=hi, mincnt=MINCNT)
        axes[1, j].text(
            0.03, 0.97,
            f"RMSE={rmse2:.3f}\nscat={scat2:.3f}\n$\\rho$={rho2:.3f}",
            transform=axes[1, j].transAxes,
            va="top", ha="left", fontsize=14,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round,pad=0.2"),
        )

        # x-labels only on bottom row
        axes[1, j].set_xlabel(r"$\log L_{\rm obs}\;[\mathrm{erg\,s^{-1}}]$")

        # y-label only on leftmost panel of each row; hide y tick labels elsewhere
        if j == 0:
            axes[0, j].set_ylabel(r"$\log L_{\rm pred}$")
            axes[1, j].set_ylabel(r"$\log L_{\rm pred}$")
        else:
            axes[0, j].set_ylabel("")
            axes[1, j].set_ylabel("")
            axes[0, j].tick_params(labelleft=False)
            axes[1, j].tick_params(labelleft=False)

    fig.text(-0.02, 0.74, r"SDSS$\rightarrow$DESI", rotation=90, va="center", ha="left", fontsize=15)
    fig.text(-0.02, 0.28, r"DESI$\rightarrow$SDSS", rotation=90, va="center", ha="left", fontsize=15)

    cbar = fig.colorbar(hb_last, ax=axes.ravel().tolist(), location="right", shrink=0.98, pad=0.01)
    cbar.set_label(f"$\\log_{{10}}(N)$ per hexbin (mincnt={MINCNT})", fontsize=15)

    out = "cross_survey_transfer_rows_cmasher_bubblegum_fullrange.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()
