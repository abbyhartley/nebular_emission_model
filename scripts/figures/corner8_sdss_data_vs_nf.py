# corner8_sdss_data_vs_nf.py
# 8x8 corner plot of the 8 TARGET log line ratios  log10(F_line / F_Halpha):
#   SDSS data          -> solid  (blue)
#   SDSS-trained NF     -> dashed (reddish-purple)
# Contours only (68% & 95%), overlaid to show how well the flow matches the data.
from pathlib import Path
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from astropy.table import Table
import corner
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from functools import partial as _pf
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces
_ROBUST_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 240,
                     "xtick.labelsize": 13, "ytick.labelsize": 13})

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
SDSS_FITS = Path(BASE + "SDSS_main_training_data_ALTB.fits")
FLOW_SDSS = Path(REPO + "models/nf_sdss_ALTB.eqx")
META_SDSS = Path(REPO + "models/nf_sdss_ALTB_meta.pkl")
OUT = REPO + "figs_ALTB/corner8_sdss_data_vs_nf.png"

N_PLOT = 30000
SEED = 0
LEVELS = (0.68, 0.95)
SMOOTH = 1.0
C_DATA = "#0072B2"   # blue  (solid)  = SDSS data
C_NF = "#CC79A7"     # reddish-purple (dashed) = SDSS-trained NF
LW = 2.3


def load_scalar_df(fits_path):
    t = Table.read(fits_path, hdu=1)
    return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()

def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    template = block_neural_autoregressive_flow(
        key=jr.key(int(meta.get("seed", 0))), base_dist=Normal(jnp.zeros(xdim)),
        cond_dim=2, inverter=_ROBUST_INV)
    return eqx.tree_deserialise_leaves(flow_path, template)

def add_loglha_sdss(df):
    from astropy.cosmology import Planck15 as cosmo
    df = df.copy()
    z = df["Z_1"].to_numpy(float); ha = df["H_ALPHA_FLUX"].to_numpy(float) * 1e-17
    dl = cosmo.luminosity_distance(z).to("cm").value
    df["LOG_LHA"] = np.log10(ha) + np.log10(4 * np.pi) + 2 * np.log10(dl)
    return df

def line_label(name):
    for key, lab in [("HBETA", r"H\beta"), ("H_BETA", r"H\beta"), ("HGAMMA", r"H\gamma"),
                     ("H_GAMMA", r"H\gamma"), ("NII_6584", "[N II]"), ("SII_6716", r"[S II]_{16}"),
                     ("SII_6717", r"[S II]_{16}"), ("SII_6731", r"[S II]_{31}"),
                     ("OII_3726", r"[O II]_{26}"), ("OII_3729", r"[O II]_{29}"), ("OIII_5007", "[O III]")]:
        if key in name:
            return rf"$\log\frac{{{lab}}}{{\mathrm{{H}}\alpha}}$"
    return name

def data_log_ratios(df, meta):
    raw = [c[6:] if c.startswith("LOG10_") else c for c in meta["resolved"]["target_cols"]]
    ha_col = next(c for c in ["H_ALPHA_FLUX", "HALPHA_FLUX", "HA_FLUX"] if c in df.columns)
    F = np.column_stack([df[c].to_numpy(float) for c in raw]); fha = df[ha_col].to_numpy(float)
    g = np.all(F > 0, axis=1) & (fha > 0) & np.isfinite(fha) & np.all(np.isfinite(F), axis=1)
    return np.log10(F[g]) - np.log10(fha[g])[:, None], raw

def sample_nf(flow, meta, df, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(N_PLOT, len(df)), replace=False)
    lm, ll = meta["resolved"]["logmstar_col"], meta["resolved"]["loglha_col"]
    U = (df[[lm, ll]].to_numpy(np.float32)[idx] - meta["U_mean"]) / meta["U_std"]
    keys = jr.split(jr.key(seed + 1234), len(idx))
    Xn = jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, jnp.asarray(U))
    return np.array(Xn) * meta["X_std"] + meta["X_mean"]

def thin(X, n, seed):
    if len(X) <= n:
        return X
    return X[np.random.default_rng(seed).choice(len(X), size=n, replace=False)]

def ranges(X, plo=0.5, phi=99.5, pad=0.15):
    return [(float(np.percentile(X[:, j], plo) - pad), float(np.percentile(X[:, j], phi) + pad))
            for j in range(X.shape[1])]


def main():
    meta = pickle.load(open(META_SDSS, "rb"))
    flow = load_flow(FLOW_SDSS, meta)
    df = add_loglha_sdss(load_scalar_df(SDSS_FITS))

    Xd_all, raw = data_log_ratios(df, meta)
    Xd = thin(Xd_all, N_PLOT, seed=1)
    Xn = sample_nf(flow, meta, df, seed=SEED + 10)
    labels = [line_label(c) for c in meta["resolved"]["out_cols"]]
    print("order:", raw)

    crange = ranges(Xd)
    fig = corner.corner(Xd, labels=labels, color=C_DATA, range=crange,
                        plot_datapoints=False, plot_density=False, fill_contours=False,
                        levels=LEVELS, smooth=SMOOTH, contour_kwargs={"linewidths": LW},
                        label_kwargs={"fontsize": 17}, hist_kwargs={"lw": LW})
    corner.corner(Xn, fig=fig, color=C_NF, range=crange,
                  plot_datapoints=False, plot_density=False, fill_contours=False,
                  levels=LEVELS, smooth=SMOOTH,
                  contour_kwargs={"linewidths": LW, "linestyles": "--"},
                  hist_kwargs={"lw": LW, "linestyle": "--"})

    handles = [plt.Line2D([0], [0], color=C_DATA, lw=2.8, label="SDSS data"),
               plt.Line2D([0], [0], color=C_NF, lw=2.8, ls="--", label="SDSS-trained NF (samples)")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98),
               frameon=True, fontsize=18)
    fig.savefig(OUT, bbox_inches="tight")
    print("Saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
