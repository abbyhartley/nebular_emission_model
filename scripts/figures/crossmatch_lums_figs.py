# crossmatch_lums_figs.py
#
# Recreate the in-survey and cross-survey line-luminosity figures, but restricted to
# the CROSS-MATCHED (same-galaxy) SDSS<->DESI sample, using the SAME pre-trained
# full-sample flows (nf_sdss_main, nf_desi_bgs) as the earlier figures. Only the set
# of galaxies changes: each matched galaxy has both an SDSS and a DESI measurement
# (docs/crossmatch_sdss_desi_fluxes.csv; logm_{sv} == LOGM_COLOR).
#
# Produces two 2x4 panels (Hb / [NII]6584 / [OII]3727 / [OIII]5007), log L_pred vs
# log L_obs, matching the style of cross_survey_transfer_rows_* and in_survey_lineratios:
#   cross_survey_transfer_crossmatched.png : top SDSS->DESI, bottom DESI->SDSS
#   in_survey_lineratios_crossmatched.png  : top DESI->DESI, bottom SDSS->SDSS

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import cmasher as cmr

from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
})

REPO = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model"
FLOW_SDSS = Path(f"{REPO}/nf_sdss_main.eqx"); META_SDSS = Path(f"{REPO}/nf_sdss_main_meta.pkl")
FLOW_DESI = Path(f"{REPO}/nf_desi_bgs.eqx");  META_DESI = Path(f"{REPO}/nf_desi_bgs_meta.pkl")
CSV = Path(f"{REPO}/docs/crossmatch_sdss_desi_fluxes.csv")

FLUX_SCALE = 1e-17
N_MC = 50
SEED = 0
GRIDSIZE = 50     # fewer galaxies than the full-sample figs -> finer/looser binning
MINCNT = 1
CMAP = cmr.bubblegum

# label, csv-line-key
PLOT_LINES = [
    ("H$\\beta$",   "Hbeta"),
    ("[NII]6584",   "NII6584"),
    ("[OII]3727",   "OII_TOTAL"),
    ("[OIII]5007",  "OIII5007"),
]
# out_cols substrings differ between the SDSS flow (LOG10_H_BETA_..) and the DESI flow
# (LOG10_HBETA_..); try survey-appropriate candidates so one map works for both metas.
OUTCOL = {
    "Hbeta": ["H_BETA", "HBETA"],
    "NII6584": ["NII_6584"],
    "OIII5007": ["OIII_5007"],
}
LINES_CSV = ["Hbeta", "Hgamma", "NII6584", "SII6717", "SII6731", "OII3726", "OII3729", "OIII5007"]


def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    key = jr.key(int(meta.get("seed", 0)))
    template = block_neural_autoregressive_flow(
        key=key, base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    return eqx.tree_deserialise_leaves(flow_path, template)


def log10_lum_from_flux(z, flux_1e17):
    z = np.asarray(z, float); f = np.asarray(flux_1e17, float) * FLUX_SCALE
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    return np.log10(f) + np.log10(4 * np.pi) + 2 * np.log10(dl_cm)


def per_line_stats(y_true, y_pred):
    r = y_pred - y_true
    rmse = float(np.sqrt(np.mean(r**2)))
    p16, p84 = np.percentile(r, [16, 84])
    return rmse, float(0.5 * (p84 - p16)), float(spearmanr(y_true, y_pred).correlation)


def build_survey_df(df, sv):
    """Return per-survey frame for matched galaxies with the columns the samplers need."""
    z = df[f"z_{sv}"].to_numpy(float)
    ha = df[f"{sv}_Halpha"].to_numpy(float)
    logm = df[f"logm_{sv}"].to_numpy(float)
    out = pd.DataFrame({
        "Z": z,
        "LOGM_COLOR": logm,
        "LOG_LHA": log10_lum_from_flux(z, ha),
        "F_Hbeta": df[f"{sv}_Hbeta"].to_numpy(float),
        "F_NII6584": df[f"{sv}_NII6584"].to_numpy(float),
        "F_OIII5007": df[f"{sv}_OIII5007"].to_numpy(float),
        "F_OII3726": df[f"{sv}_OII3726"].to_numpy(float),
        "F_OII3729": df[f"{sv}_OII3729"].to_numpy(float),
    })
    return out


def sample_ratios(flow, meta, df, *, seed=0, n_mc=N_MC):
    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]
    U = df[[logm_col, loglha_col]].to_numpy(np.float32)
    Un = jnp.asarray((U - meta["U_mean"]) / meta["U_std"])
    X_mean, X_std = meta["X_mean"], meta["X_std"]
    n = len(df)
    key = jr.key(seed + 999)
    acc = np.zeros((n, len(meta["resolved"]["out_cols"])), np.float64)
    for _ in range(int(n_mc)):
        key, sk = jr.split(key)
        keys = jr.split(sk, n)
        Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un))
        acc += Xn * X_std + X_mean
    return acc / n_mc


def outcol_index(meta, candidates):
    for i, name in enumerate(meta["resolved"]["out_cols"]):
        if any(c in name for c in candidates):
            return i
    raise KeyError(candidates)


def pred_logL(df, meta, ratios, key):
    ll = df["LOG_LHA"].to_numpy(float)
    if key == "OII_TOTAL":
        i26 = outcol_index(meta, ["OII_3726"]); i29 = outcol_index(meta, ["OII_3729"])
        log_ratio = np.log10(10.0**ratios[:, i26] + 10.0**ratios[:, i29])
        return ll + log_ratio
    return ll + ratios[:, outcol_index(meta, OUTCOL[key])].astype(float)


def true_logL(df, key):
    z = df["Z"].to_numpy(float)
    if key == "OII_TOTAL":
        f = df["F_OII3726"].to_numpy(float) + df["F_OII3729"].to_numpy(float)
        return log10_lum_from_flux(z, f)
    colmap = {"Hbeta": "F_Hbeta", "NII6584": "F_NII6584", "OIII5007": "F_OIII5007"}
    return log10_lum_from_flux(z, df[colmap[key]].to_numpy(float))


def hex_panel(ax, x, y, lo, hi):
    hb = ax.hexbin(x, y, gridsize=GRIDSIZE, extent=(lo, hi, lo, hi),
                   bins="log", mincnt=MINCNT, cmap=CMAP)
    ax.plot([lo, hi], [lo, hi], color="black", lw=3.0, ls=":", alpha=0.95)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
    return hb


def finite_minmax(*arrs):
    v = np.concatenate([np.asarray(a, float).ravel() for a in arrs]); v = v[np.isfinite(v)]
    return float(v.min()), float(v.max())


def make_figure(rows, outname, title):
    """rows = list of (row_label, df, meta, ratios) top-to-bottom."""
    ncols = len(PLOT_LINES)
    # per-column limits across both rows
    col_lim = []
    for (label, csv_key) in PLOT_LINES:
        arrs = []
        for (_lbl, df, meta, ratios) in rows:
            arrs.append(true_logL(df, csv_key)); arrs.append(pred_logL(df, meta, ratios, csv_key))
        lo, hi = finite_minmax(*arrs)
        col_lim.append((lo - 0.05, hi + 0.05))

    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols + 1.2, 6.9),
                             sharex="col", sharey="col", constrained_layout=True)
    hb_last = None
    for r, (row_label, df, meta, ratios) in enumerate(rows):
        for j, (label, csv_key) in enumerate(PLOT_LINES):
            lo, hi = col_lim[j]
            yobs = true_logL(df, csv_key)
            ypred = pred_logL(df, meta, ratios, csv_key)
            m = np.isfinite(yobs) & np.isfinite(ypred)
            rmse, scat, rho = per_line_stats(yobs[m], ypred[m])
            hb_last = hex_panel(axes[r, j], yobs[m], ypred[m], lo, hi)
            if r == 0:
                axes[r, j].set_title(label)
            axes[r, j].text(0.03, 0.97, f"RMSE={rmse:.3f}\nscat={scat:.3f}\n$\\rho$={rho:.3f}",
                            transform=axes[r, j].transAxes, va="top", ha="left", fontsize=13,
                            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round,pad=0.2"))
            if r == 1:
                axes[r, j].set_xlabel(r"$\log L_{\rm obs}\;[\mathrm{erg\,s^{-1}}]$")
            if j == 0:
                axes[r, j].set_ylabel(r"$\log L_{\rm pred}$")
            else:
                axes[r, j].tick_params(labelleft=False)

    fig.text(-0.02, 0.74, rows[0][0], rotation=90, va="center", ha="left", fontsize=15)
    fig.text(-0.02, 0.28, rows[1][0], rotation=90, va="center", ha="left", fontsize=15)
    cbar = fig.colorbar(hb_last, ax=axes.ravel().tolist(), location="right", shrink=0.98, pad=0.01)
    cbar.set_label(f"$\\log_{{10}}(N)$ per hexbin (mincnt={MINCNT})", fontsize=15)
    fig.suptitle(title, fontsize=15)
    fig.savefig(outname, dpi=250, bbox_inches="tight")
    print("Saved:", outname, flush=True)


def main():
    meta_sdss = pickle.load(open(META_SDSS, "rb"))
    meta_desi = pickle.load(open(META_DESI, "rb"))
    flow_sdss = load_flow(FLOW_SDSS, meta_sdss)
    flow_desi = load_flow(FLOW_DESI, meta_desi)

    df = pd.read_csv(CSV)
    # intersection validity (Ha>0, z>0, logm finite, all shown lines >0) so the SAME
    # galaxies appear in every panel
    def good(sv):
        m = (df[f"{sv}_Halpha"] > 0) & np.isfinite(df[f"z_{sv}"]) & (df[f"z_{sv}"] > 0) & np.isfinite(df[f"logm_{sv}"])
        for L in LINES_CSV:
            m &= (df[f"{sv}_{L}"] > 0) & np.isfinite(df[f"{sv}_{L}"])
        return m.to_numpy()
    keep = good("sdss") & good("desi")
    df = df.loc[keep].reset_index(drop=True)
    N = len(df)
    print(f"matched galaxies used (valid both surveys): {N}", flush=True)

    df_sdss = build_survey_df(df, "sdss")
    df_desi = build_survey_df(df, "desi")

    # sample the pre-trained flows on the matched galaxies' conditions (cache to disk
    # so figure-only re-runs don't repeat the ~minutes of BNAF sampling)
    cache = Path("crossmatch_ratios_cache.npz")
    if cache.exists():
        d = np.load(cache)
        if d["n"] == N:
            r_sdss_on_sdss, r_desi_on_desi = d["ss"], d["dd"]
            r_sdss_on_desi, r_desi_on_sdss = d["sd"], d["ds"]
            print("loaded cached ratios", flush=True)
        else:
            cache.unlink()
    if not cache.exists():
        r_sdss_on_sdss = sample_ratios(flow_sdss, meta_sdss, df_sdss, seed=SEED + 1)
        r_desi_on_desi = sample_ratios(flow_desi, meta_desi, df_desi, seed=SEED + 2)
        r_sdss_on_desi = sample_ratios(flow_sdss, meta_sdss, df_desi, seed=SEED + 3)
        r_desi_on_sdss = sample_ratios(flow_desi, meta_desi, df_sdss, seed=SEED + 4)
        np.savez(cache, n=N, ss=r_sdss_on_sdss, dd=r_desi_on_desi,
                 sd=r_sdss_on_desi, ds=r_desi_on_sdss)

    # CROSS-SURVEY: top SDSS->DESI, bottom DESI->SDSS
    make_figure(
        rows=[(r"SDSS$\rightarrow$DESI", df_desi, meta_sdss, r_sdss_on_desi),
              (r"DESI$\rightarrow$SDSS", df_sdss, meta_desi, r_desi_on_sdss)],
        outname="cross_survey_transfer_crossmatched.png",
        title=f"Cross-survey transfer — cross-matched sample (N={N})",
    )

    # IN-SURVEY: top DESI->DESI, bottom SDSS->SDSS (matches in_survey_lineratios row order)
    make_figure(
        rows=[(r"DESI$\rightarrow$DESI", df_desi, meta_desi, r_desi_on_desi),
              (r"SDSS$\rightarrow$SDSS", df_sdss, meta_sdss, r_sdss_on_sdss)],
        outname="in_survey_lineratios_crossmatched.png",
        title=f"In-survey — cross-matched sample (N={N})",
    )


if __name__ == "__main__":
    main()
