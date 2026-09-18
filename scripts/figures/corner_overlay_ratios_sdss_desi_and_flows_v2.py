# corner_overlay_ratios_sdss_desi_and_flows_scienceplots.py
from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

import corner
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow


# -----------------------
# Style
# -----------------------
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 250,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
})

# Okabe–Ito palette (colorblind friendly)
C_SDSS = "#0072B2"   # blue
C_DESI = "#E69F00"   # orange
C_NF_SDSS = "#CC79A7"  # purple
C_NF_DESI = "#009E73"  # green

# Line widths
LW_DATA = 2.0
LW_NF = 2.0

# -----------------------
# Inputs
# -----------------------
SDSS_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")
DESI_FITS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

FLOW_SDSS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main.eqx")
META_SDSS = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_sdss_main_meta.pkl")

FLOW_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs.eqx")
META_DESI = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/nf_desi_bgs_meta.pkl")

FLUX_SCALE = 1e-17
N_PLOT = 30000
SEED = 0

LEVELS = (0.68, 0.95)
SMOOTH = 1.0
RANGE_PLO = 0.5
RANGE_PHI = 99.5
RANGE_PAD = 0.15


# -----------------------
# Data prep
# -----------------------
def load_scalar_df(fits_path):
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()

def add_loglha_sdss(df):
    df = df.copy()
    z = df["Z_1"].to_numpy(dtype=float)
    ha = df["H_ALPHA_FLUX"].to_numpy(dtype=float) * FLUX_SCALE
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    df["LOG_LHA"] = np.log10(ha) + np.log10(4.0*np.pi) + 2.0*np.log10(dl_cm)
    return df

def add_loglha_desi(df):
    df = df.copy()
    z = df["Z"].to_numpy(dtype=float)
    ha = df["HALPHA_FLUX"].to_numpy(dtype=float) * FLUX_SCALE
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    df["LOG_LHA"] = np.log10(ha) + np.log10(4.0*np.pi) + 2.0*np.log10(dl_cm)
    return df

def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    key = jr.key(int(meta.get("seed", 0)))
    template = block_neural_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=2,
    )
    return eqx.tree_deserialise_leaves(flow_path, template)

def thin(x, n, seed=0):
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx]

def compute_ratios_sdss(df):
    ha = df["H_ALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    hb = df["H_BETA_FLUX"].to_numpy(float) * FLUX_SCALE
    o3 = df["OIII_5007_FLUX"].to_numpy(float) * FLUX_SCALE
    n2 = df["NII_6584_FLUX"].to_numpy(float) * FLUX_SCALE
    o2a = df["OII_3726_FLUX"].to_numpy(float) * FLUX_SCALE
    o2b = df["OII_3729_FLUX"].to_numpy(float) * FLUX_SCALE

    m = np.isfinite(ha) & np.isfinite(hb) & np.isfinite(o3) & np.isfinite(n2) & np.isfinite(o2a) & np.isfinite(o2b)
    m &= (ha > 0) & (hb > 0) & (o3 > 0) & (n2 > 0) & (o2a > 0) & (o2b > 0)

    ha, hb, o3, n2, o2a, o2b = ha[m], hb[m], o3[m], n2[m], o2a[m], o2b[m]
    o2 = o2a + o2b

    x1 = np.log10(n2 / ha)
    x2 = np.log10(o3 / hb)
    x3 = np.log10(o2 / ha)
    return np.vstack([x1, x2, x3]).T

def compute_ratios_desi(df):
    ha = df["HALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    hb = df["HBETA_FLUX"].to_numpy(float) * FLUX_SCALE
    o3 = df["OIII_5007_FLUX"].to_numpy(float) * FLUX_SCALE
    n2 = df["NII_6584_FLUX"].to_numpy(float) * FLUX_SCALE
    o2a = df["OII_3726_FLUX"].to_numpy(float) * FLUX_SCALE
    o2b = df["OII_3729_FLUX"].to_numpy(float) * FLUX_SCALE

    m = np.ones(len(df), dtype=bool)
    if "HALPHA_FLUX_IVAR" in df.columns:
        m &= (df["HALPHA_FLUX_IVAR"].to_numpy(float) > 0)

    m &= np.isfinite(ha) & np.isfinite(hb) & np.isfinite(o3) & np.isfinite(n2) & np.isfinite(o2a) & np.isfinite(o2b)
    m &= (ha > 0) & (hb > 0) & (o3 > 0) & (n2 > 0) & (o2a > 0) & (o2b > 0)

    ha, hb, o3, n2, o2a, o2b = ha[m], hb[m], o3[m], n2[m], o2a[m], o2b[m]
    o2 = o2a + o2b

    x1 = np.log10(n2 / ha)
    x2 = np.log10(o3 / hb)
    x3 = np.log10(o2 / ha)
    return np.vstack([x1, x2, x3]).T

def sample_ratios_from_flow(flow, meta, df_cond, *, seed=0, n_plot=30000):
    rng = np.random.default_rng(seed)
    n = min(n_plot, len(df_cond))
    idx = rng.choice(len(df_cond), size=n, replace=False)

    logm_col = meta["resolved"]["logmstar_col"]
    loglha_col = meta["resolved"]["loglha_col"]

    U = df_cond[[logm_col, loglha_col]].to_numpy(dtype=np.float32)[idx]
    Un = (U - meta["U_mean"]) / meta["U_std"]
    Un_j = jnp.asarray(Un)

    key = jr.key(seed + 1234)
    keys = jr.split(key, n)

    def sample_one(k, u):
        return flow.sample(k, sample_shape=(), condition=u)

    Xn = jax.vmap(sample_one, in_axes=(0, 0))(keys, Un_j)
    Xn = np.array(Xn)

    ratios8 = Xn * meta["X_std"] + meta["X_mean"]
    out_cols = meta["resolved"]["out_cols"]

    def find_dim(substrs):
        for i, name in enumerate(out_cols):
            for s in substrs:
                if s in name:
                    return i
        raise KeyError(f"Could not find any of {substrs} in out_cols.")

    i_nii = find_dim(["NII_6584"])
    i_oiii = find_dim(["OIII_5007"])
    i_hb = find_dim(["HBETA", "H_BETA"])
    i_oii3726 = find_dim(["OII_3726"])
    i_oii3729 = find_dim(["OII_3729"])

    logNII_Ha = ratios8[:, i_nii]
    logOIII_Hb = ratios8[:, i_oiii] - ratios8[:, i_hb]
    r26 = ratios8[:, i_oii3726]
    r29 = ratios8[:, i_oii3729]
    logOII_Ha = np.log10(np.power(10.0, r26) + np.power(10.0, r29))
    return np.vstack([logNII_Ha, logOIII_Hb, logOII_Ha]).T

def ranges_from_percentiles(X, p_lo=0.5, p_hi=99.5, pad=0.0):
    out = []
    for j in range(X.shape[1]):
        lo, hi = np.percentile(X[:, j], [p_lo, p_hi])
        out.append((float(lo - pad), float(hi + pad)))
    return out


# -----------------------
# Plot helpers
# -----------------------
def make_corner_contours_only(X_sdss, X_desi, X_nf_sdss, X_nf_desi, labels, crange, out_png):
    fig = corner.corner(
        X_sdss, labels=labels, color=C_SDSS, range=crange,
        plot_datapoints=False, fill_contours=False, plot_density=False,
        levels=LEVELS, smooth=SMOOTH,
        contour_kwargs={"linewidths": LW_DATA},
    )
    corner.corner(
        X_desi, fig=fig, color=C_DESI, range=crange,
        plot_datapoints=False, fill_contours=False, plot_density=False,
        levels=LEVELS, smooth=SMOOTH,
        contour_kwargs={"linewidths": LW_DATA},
    )
    corner.corner(
        X_nf_sdss, fig=fig, color=C_NF_SDSS, range=crange,
        plot_datapoints=False, fill_contours=False, plot_density=False,
        levels=LEVELS, smooth=SMOOTH,
        contour_kwargs={"linewidths": LW_NF, "linestyles": "--"},
    )
    corner.corner(
        X_nf_desi, fig=fig, color=C_NF_DESI, range=crange,
        plot_datapoints=False, fill_contours=False, plot_density=False,
        levels=LEVELS, smooth=SMOOTH,
        contour_kwargs={"linewidths": LW_NF, "linestyles": "--"},
    )

    handles = [
        plt.Line2D([0], [0], color=C_SDSS, lw=LW_DATA, label="SDSS data"),
        plt.Line2D([0], [0], color=C_DESI, lw=LW_DATA, label="DESI data"),
        plt.Line2D([0], [0], color=C_NF_SDSS, lw=LW_NF, ls="--", label="NF trained on SDSS (samples)"),
        plt.Line2D([0], [0], color=C_NF_DESI, lw=LW_NF, ls="--", label="NF trained on DESI (samples)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)

    fig.savefig(out_png, bbox_inches="tight")
    print("Saved:", out_png)
    plt.close(fig)

def make_corner_filled_data_contours_nf(X_sdss, X_desi, X_nf_sdss, X_nf_desi, labels, crange, out_png):
    # Fill SDSS contours lightly
    fig = corner.corner(
        X_sdss, labels=labels, color=C_SDSS, range=crange,
        plot_datapoints=False, fill_contours=True, plot_density=False,
        levels=LEVELS, smooth=SMOOTH,
        contour_kwargs={"linewidths": LW_DATA},
        contourf_kwargs={"alpha": 0.22},
    )
    # Fill DESI contours lightly (overlaid)
    corner.corner(
        X_desi, fig=fig, color=C_DESI, range=crange,
        plot_datapoints=False, fill_contours=True, plot_density=False,
        levels=LEVELS, smooth=SMOOTH,
        contour_kwargs={"linewidths": LW_DATA},
        contourf_kwargs={"alpha": 0.18},
    )
    # NF contours as dashed lines only
    corner.corner(
        X_nf_sdss, fig=fig, color=C_NF_SDSS, range=crange,
        plot_datapoints=False, fill_contours=False, plot_density=False,
        levels=LEVELS, smooth=SMOOTH,
        contour_kwargs={"linewidths": LW_NF, "linestyles": "--"},
    )
    corner.corner(
        X_nf_desi, fig=fig, color=C_NF_DESI, range=crange,
        plot_datapoints=False, fill_contours=False, plot_density=False,
        levels=LEVELS, smooth=SMOOTH,
        contour_kwargs={"linewidths": LW_NF, "linestyles": "--"},
    )

    handles = [
        plt.Line2D([0], [0], color=C_SDSS, lw=LW_DATA, label="SDSS data (filled)"),
        plt.Line2D([0], [0], color=C_DESI, lw=LW_DATA, label="DESI data (filled)"),
        plt.Line2D([0], [0], color=C_NF_SDSS, lw=LW_NF, ls="--", label="NF trained on SDSS"),
        plt.Line2D([0], [0], color=C_NF_DESI, lw=LW_NF, ls="--", label="NF trained on DESI"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)

    fig.savefig(out_png, bbox_inches="tight")
    print("Saved:", out_png)
    plt.close(fig)


def main():
    df_sdss = add_loglha_sdss(load_scalar_df(SDSS_FITS))
    df_desi = add_loglha_desi(load_scalar_df(DESI_FITS))

    X_sdss = thin(compute_ratios_sdss(df_sdss), N_PLOT, seed=0)
    X_desi = thin(compute_ratios_desi(df_desi), N_PLOT, seed=1)

    with open(META_SDSS, "rb") as f:
        meta_sdss = pickle.load(f)
    with open(META_DESI, "rb") as f:
        meta_desi = pickle.load(f)

    flow_sdss = load_flow(FLOW_SDSS, meta_sdss)
    flow_desi = load_flow(FLOW_DESI, meta_desi)

    X_nf_sdss = sample_ratios_from_flow(flow_sdss, meta_sdss, df_sdss, seed=SEED + 10, n_plot=N_PLOT)
    X_nf_desi = sample_ratios_from_flow(flow_desi, meta_desi, df_desi, seed=SEED + 20, n_plot=N_PLOT)

    labels = [
        r"$\log([\mathrm{NII}]/\mathrm{H}\alpha)$",
        r"$\log([\mathrm{OIII}]/\mathrm{H}\beta)$",
        r"$\log([\mathrm{OII}]/\mathrm{H}\alpha)$",
    ]

    # ranges from SDSS percentiles (as you prefer)
    crange = ranges_from_percentiles(X_sdss, p_lo=RANGE_PLO, p_hi=RANGE_PHI, pad=RANGE_PAD)
    print(f"Using corner ranges (SDSS {RANGE_PLO}–{RANGE_PHI}% + pad={RANGE_PAD}):", crange)

    # Version 1: contours only
    make_corner_contours_only(
        X_sdss, X_desi, X_nf_sdss, X_nf_desi,
        labels, crange,
        out_png="corner_contours_only_scienceplots.png",
    )

    # Version 2: fill SDSS+DESI, contours for NFs
    make_corner_filled_data_contours_nf(
        X_sdss, X_desi, X_nf_sdss, X_nf_desi,
        labels, crange,
        out_png="corner_filled_data_plus_nf_contours_scienceplots.png",
    )


if __name__ == "__main__":
    main()
