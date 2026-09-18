# cross_survey_transfer_panels_all8_fixed2.py
from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow


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

# (label, DESI flux col, SDSS flux col, substring in SDSS meta out_cols, substring in DESI meta out_cols)
PLOT_LINES = [
    ("H$\\beta$",   "HBETA_FLUX",      "H_BETA_FLUX",      "H_BETA",    "HBETA"),
    ("H$\\gamma$",  "HGAMMA_FLUX",     "H_GAMMA_FLUX",     "H_GAMMA",   "HGAMMA"),
    ("[NII]6584",   "NII_6584_FLUX",   "NII_6584_FLUX",    "NII_6584",  "NII_6584"),
    ("[SII]671x",   "SII_6716_FLUX",   "SII_6717_FLUX",    "SII_671",   "SII_671"),  # match 6716 or 6717
    ("[SII]6731",   "SII_6731_FLUX",   "SII_6731_FLUX",    "SII_6731",  "SII_6731"),
    ("[OII]3726",   "OII_3726_FLUX",   "OII_3726_FLUX",    "OII_3726",  "OII_3726"),
    ("[OII]3729",   "OII_3729_FLUX",   "OII_3729_FLUX",    "OII_3729",  "OII_3729"),
    ("[OIII]5007",  "OIII_5007_FLUX",  "OIII_5007_FLUX",   "OIII_5007", "OIII_5007"),
]

SDSS_COLS = dict(z="Z_1", ha="H_ALPHA_FLUX", logm="LOGM_COLOR")
DESI_COLS = dict(z="Z", ha="HALPHA_FLUX", ha_ivar="HALPHA_FLUX_IVAR", logm="LOGM_COLOR")


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

        for _, _, sdss_col, _, _ in PLOT_LINES:
            x = df[sdss_col].to_numpy(float)
            mask &= np.isfinite(x) & (x > 0)

        df = df.loc[mask].copy().reset_index(drop=True)
        df["LOG_LHA"] = log10_lum_from_flux(df[SDSS_COLS["z"]].to_numpy(float),
                                            df[SDSS_COLS["ha"]].to_numpy(float))
        return df

    elif survey == "desi":
        z = df[DESI_COLS["z"]].to_numpy(float)
        ha = df[DESI_COLS["ha"]].to_numpy(float)
        m = df[DESI_COLS["logm"]].to_numpy(float)
        mask = np.isfinite(z) & np.isfinite(ha) & (ha > 0) & np.isfinite(m)

        if DESI_COLS["ha_ivar"] in df.columns:
            mask &= (df[DESI_COLS["ha_ivar"]].to_numpy(float) > 0)

        for _, desi_col, _, _, _ in PLOT_LINES:
            x = df[desi_col].to_numpy(float)
            mask &= np.isfinite(x) & (x > 0)

        df = df.loc[mask].copy().reset_index(drop=True)
        df["LOG_LHA"] = log10_lum_from_flux(df[DESI_COLS["z"]].to_numpy(float),
                                            df[DESI_COLS["ha"]].to_numpy(float))
        return df
    else:
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


def pred_logL(df, meta, ratios8, substr):
    idx = outcol_index(meta, substr)
    return df["LOG_LHA"].to_numpy(float) + ratios8[:, idx].astype(float)


def true_logL(df, survey, desi_col, sdss_col):
    if survey == "desi":
        return log10_lum_from_flux(df[DESI_COLS["z"]].to_numpy(float), df[desi_col].to_numpy(float))
    else:
        return log10_lum_from_flux(df[SDSS_COLS["z"]].to_numpy(float), df[sdss_col].to_numpy(float))


def hex_panel(ax, x, y, title):
    lo = np.percentile(np.concatenate([x, y]), 0.5)
    hi = np.percentile(np.concatenate([x, y]), 99.5)
    lo -= 0.1
    hi += 0.1
    hb = ax.hexbin(x, y, gridsize=70, extent=(lo, hi, lo, hi),
                   bins="log", mincnt=MINCNT, cmap="viridis")
    ax.plot([lo, hi], [lo, hi], color="k", lw=1.0, alpha=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$\log L_{\rm obs}$")
    ax.set_ylabel(r"$\log L_{\rm pred}$")
    return hb


def main():
    with open(META_SDSS, "rb") as f:
        meta_sdss = pickle.load(f)
    with open(META_DESI, "rb") as f:
        meta_desi = pickle.load(f)

    print("SDSS meta out_cols:", meta_sdss["resolved"]["out_cols"])
    print("DESI meta out_cols:", meta_desi["resolved"]["out_cols"])

    flow_sdss = load_flow(FLOW_SDSS, meta_sdss)
    flow_desi = load_flow(FLOW_DESI, meta_desi)

    df_desi = prep_df(DESI_FITS, "desi")
    df_sdss = prep_df(SDSS_FITS, "sdss")
    print("N DESI used:", len(df_desi))
    print("N SDSS used:", len(df_sdss))

    ratios_sdss_on_desi = sample_ratios(flow_sdss, meta_sdss, df_desi, seed=SEED+1, n_mc=N_MC)
    ratios_desi_on_sdss = sample_ratios(flow_desi, meta_desi, df_sdss, seed=SEED+2, n_mc=N_MC)

    nrows = len(PLOT_LINES)
    fig, axes = plt.subplots(nrows, 2, figsize=(10.4, 3.0*nrows), constrained_layout=True)

    hb_last = None
    for i, (label, desi_col, sdss_col, substr_sdss, substr_desi) in enumerate(PLOT_LINES):
        # SDSS->DESI uses SDSS meta substrings
        yobs = true_logL(df_desi, "desi", desi_col, sdss_col)
        ypred = pred_logL(df_desi, meta_sdss, ratios_sdss_on_desi, substr_sdss)
        rmse, scat, rho = per_line_stats(yobs, ypred)
        hb_last = hex_panel(axes[i, 0], yobs, ypred,
                            f"SDSS→DESI  {label}\nRMSE={rmse:.3f}, scat={scat:.3f}, ρ={rho:.3f}")

        # DESI->SDSS uses DESI meta substrings
        yobs2 = true_logL(df_sdss, "sdss", desi_col, sdss_col)
        ypred2 = pred_logL(df_sdss, meta_desi, ratios_desi_on_sdss, substr_desi)
        rmse2, scat2, rho2 = per_line_stats(yobs2, ypred2)
        hb_last = hex_panel(axes[i, 1], yobs2, ypred2,
                            f"DESI→SDSS  {label}\nRMSE={rmse2:.3f}, scat={scat2:.3f}, ρ={rho2:.3f}")

    cbar = fig.colorbar(hb_last, ax=axes.ravel().tolist(), location="right", shrink=0.97, pad=0.01)
    cbar.set_label(f"log10(N per hex), mincnt={MINCNT}")

    fig.suptitle("Cross-survey transfer: predicted vs observed line luminosities", fontsize=14)
    out = "cross_survey_transfer_panels_all8.png"
    fig.savefig(out, dpi=250)
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()
