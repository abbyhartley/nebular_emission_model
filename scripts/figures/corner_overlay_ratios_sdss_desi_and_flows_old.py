# corner_overlay_ratios_sdss_desi_and_flows.py
#
# Makes an overlaid 3D "corner" contour plot in key ratio space:
#   x1 = log([NII]/Hα)
#   x2 = log([OIII]/Hβ)
#   x3 = log([OII]/Hα) with [OII]=3726+3729
#
# Overlays:
#   - SDSS data
#   - DESI data
#   - samples from SDSS-trained flow (conditioned on SDSS galaxies)
#   - samples from DESI-trained flow (conditioned on DESI galaxies)
#
# IMPORTANT: Flow conditioning requires LOG_LHA, which we compute on the fly here.

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

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow


# -----------------------
# Paths
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


# -----------------------
# Utilities
# -----------------------
def load_scalar_df(fits_path):
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()


def add_loglha_sdss(df):
    """Add LOG_LHA computed from (Z_1, H_ALPHA_FLUX)."""
    df = df.copy()
    z = df["Z_1"].to_numpy(dtype=float)
    ha = df["H_ALPHA_FLUX"].to_numpy(dtype=float) * FLUX_SCALE
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    df["LOG_LHA"] = np.log10(ha) + np.log10(4.0 * np.pi) + 2.0 * np.log10(dl_cm)
    return df


def add_loglha_desi(df):
    """Add LOG_LHA computed from (Z, HALPHA_FLUX)."""
    df = df.copy()
    z = df["Z"].to_numpy(dtype=float)
    ha = df["HALPHA_FLUX"].to_numpy(dtype=float) * FLUX_SCALE
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    df["LOG_LHA"] = np.log10(ha) + np.log10(4.0 * np.pi) + 2.0 * np.log10(dl_cm)
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


# -----------------------
# Ratio feature construction (data)
# -----------------------
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

    x1 = np.log10(n2 / ha)      # NII/Ha
    x2 = np.log10(o3 / hb)      # OIII/Hb
    x3 = np.log10(o2 / ha)      # OII/Ha (sum)
    return np.vstack([x1, x2, x3]).T


def compute_ratios_desi(df):
    ha = df["HALPHA_FLUX"].to_numpy(float) * FLUX_SCALE
    hb = df["HBETA_FLUX"].to_numpy(float) * FLUX_SCALE
    o3 = df["OIII_5007_FLUX"].to_numpy(float) * FLUX_SCALE
    n2 = df["NII_6584_FLUX"].to_numpy(float) * FLUX_SCALE
    o2a = df["OII_3726_FLUX"].to_numpy(float) * FLUX_SCALE
    o2b = df["OII_3729_FLUX"].to_numpy(float) * FLUX_SCALE

    # Require IVAR>0 if present
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


# -----------------------
# Ratio feature construction (NF samples)
# -----------------------
def sample_ratios_from_flow(flow, meta, df_cond, *, seed=0, n_plot=30000):
    """
    Samples the 8-D ratio vector from the flow and then returns a 3-D projection:
      [log(NII/Ha), log(OIII/Hb), log(OII/Ha)]
    """
    rng = np.random.default_rng(seed)
    n = min(n_plot, len(df_cond))
    idx = rng.choice(len(df_cond), size=n, replace=False)

    logm_col = meta["resolved"]["logmstar_col"]   # expect LOGM_COLOR
    loglha_col = meta["resolved"]["loglha_col"]   # expect LOG_LHA

    # Conditioning matrix
    U = df_cond[[logm_col, loglha_col]].to_numpy(dtype=np.float32)[idx]
    Un = (U - meta["U_mean"]) / meta["U_std"]
    Un_j = jnp.asarray(Un)

    key = jr.key(seed + 1234)
    keys = jr.split(key, n)

    def sample_one(k, u):
        return flow.sample(k, sample_shape=(), condition=u)  # (8,)

    Xn = jax.vmap(sample_one, in_axes=(0, 0))(keys, Un_j)
    Xn = np.array(Xn)

    ratios8 = Xn * meta["X_std"] + meta["X_mean"]  # (n,8), log(line/Ha)

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


# -----------------------
# Main
# -----------------------
def main():
    # Load data
    df_sdss = add_loglha_sdss(load_scalar_df(SDSS_FITS))
    df_desi = add_loglha_desi(load_scalar_df(DESI_FITS))

    # Compute observed ratio features
    X_sdss = thin(compute_ratios_sdss(df_sdss), N_PLOT, seed=0)
    X_desi = thin(compute_ratios_desi(df_desi), N_PLOT, seed=1)

    # Load flows + metas
    with open(META_SDSS, "rb") as f:
        meta_sdss = pickle.load(f)
    with open(META_DESI, "rb") as f:
        meta_desi = pickle.load(f)

    flow_sdss = load_flow(FLOW_SDSS, meta_sdss)
    flow_desi = load_flow(FLOW_DESI, meta_desi)

    # Sample ratios from each flow using its native survey conditions
    X_nf_sdss = sample_ratios_from_flow(flow_sdss, meta_sdss, df_sdss, seed=SEED + 10, n_plot=N_PLOT)
    X_nf_desi = sample_ratios_from_flow(flow_desi, meta_desi, df_desi, seed=SEED + 20, n_plot=N_PLOT)

    labels = [
        r"$\log([\mathrm{NII}]/\mathrm{H}\alpha)$",
        r"$\log([\mathrm{OIII}]/\mathrm{H}\beta)$",
        r"$\log([\mathrm{OII}]/\mathrm{H}\alpha)$",
    ]

    # Corner contours
    levels = (0.393, 0.865, 0.989)  # ~1,2,3σ for 2D if Gaussian-ish

    fig = corner.corner(
        X_sdss,
        labels=labels,
        color="k",
        plot_datapoints=False,
        fill_contours=False,
        plot_density=False,
        levels=levels,
        contour_kwargs={"linewidths": 1.6},
    )

    corner.corner(
        X_desi,
        fig=fig,
        color="tab:blue",
        plot_datapoints=False,
        fill_contours=False,
        plot_density=False,
        levels=levels,
        contour_kwargs={"linewidths": 1.6},
    )

    corner.corner(
        X_nf_sdss,
        fig=fig,
        color="tab:orange",
        plot_datapoints=False,
        fill_contours=False,
        plot_density=False,
        levels=levels,
        contour_kwargs={"linewidths": 1.6, "linestyles": "--"},
    )

    corner.corner(
        X_nf_desi,
        fig=fig,
        color="tab:green",
        plot_datapoints=False,
        fill_contours=False,
        plot_density=False,
        levels=levels,
        contour_kwargs={"linewidths": 1.6, "linestyles": "--"},
    )

    # Legend in top-left panel
    axes = np.array(fig.axes).reshape((3, 3))
    axes[0, 0].legend(
        handles=[
            plt.Line2D([0], [0], color="k", lw=2, label="SDSS data"),
            plt.Line2D([0], [0], color="tab:blue", lw=2, label="DESI data"),
            plt.Line2D([0], [0], color="tab:orange", lw=2, ls="--", label="NF trained on SDSS (samples)"),
            plt.Line2D([0], [0], color="tab:green", lw=2, ls="--", label="NF trained on DESI (samples)"),
        ],
        loc="upper right",
        fontsize=9,
        frameon=True,
    )

    out = "corner_overlay_ratios_sdss_desi_and_flows.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()
