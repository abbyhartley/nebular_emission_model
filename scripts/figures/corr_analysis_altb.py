# corr_analysis_altb.py
# Correlation-matrix analysis for the ALT-B flows, addressing:
#  - add conditioning params (logM*, log F_Halpha) to the 8 line ratios -> 10x10
#  - cross-matched sample (same galaxies in SDSS & DESI)  -> isolates MEASUREMENT diff
#  - bootstrap/population-matched sample (DESI resampled to SDSS z,M*,L_Ha) -> isolates POPULATION diff
#  - data, NF, and NF/data RATIO matrices
#  - print model-error vs survey-difference magnitudes (esp. the [NII] block)
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from functools import partial as _pf
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import scienceplots  # noqa

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
CSV = REPO + "docs/crossmatch_sdss_desi_fluxes.csv"
FLOW_SDSS, META_SDSS = REPO + "models/nf_sdss_ALTB.eqx", REPO + "models/nf_sdss_ALTB_meta.pkl"
FLOW_DESI, META_DESI = REPO + "models/nf_desi_ALTB.eqx", REPO + "models/nf_desi_ALTB_meta.pkl"
SDSS_FITS, DESI_FITS = BASE + "SDSS_main_training_data_ALTB.fits", BASE + "DESI_BGS_training_data_ALTB.fits"
FLUX_SCALE = 1e-17; SEED = 0
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"axes.titlesize": 12, "xtick.labelsize": 8, "ytick.labelsize": 8})

LAB8 = [r"H$\beta$", r"H$\gamma$", "[NII]", r"[SII]a", r"[SII]b", r"[OII]a", r"[OII]b", "[OIII]"]
LAB10 = LAB8 + [r"$\log M_*$", r"$\log L_{H\alpha}$"]
# crossmatch CSV line keys (order = flow out_cols order)
CM_LINES = ["Hbeta", "Hgamma", "NII6584", "SII6717", "SII6731", "OII3726", "OII3729", "OIII5007"]
NII = 2  # index of [NII] among the 8


def log10_lum(z, f1e17):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f1e17, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


_ROBUST_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))


def load_flow(fp, meta):
    xdim = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xdim)), cond_dim=2,
                                            inverter=_ROBUST_INV)
    return eqx.tree_deserialise_leaves(fp, tmpl)


def sample_ratios(flow, meta, logm, loglha, seed=0):
    U = (np.column_stack([logm, loglha]).astype(np.float32) - meta["U_mean"]) / meta["U_std"]
    keys = jr.split(jr.key(seed + 1234), len(U))
    Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, jnp.asarray(U)))
    return Xn * meta["X_std"] + meta["X_mean"]          # (N,8) log ratios


def corr10(ratios8, logm, logfha, weights=None):
    X = np.column_stack([ratios8, logm, logfha])
    good = np.all(np.isfinite(X), axis=1)
    X = X[good]
    if weights is None:
        return np.corrcoef(X, rowvar=False)
    w = weights[good]; w = w / w.sum()
    mu = np.average(X, axis=0, weights=w)
    Xc = X - mu
    cov = (Xc * w[:, None]).T @ Xc
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


# ---------- data / model log-ratios for a given sample ----------
def cm_ratios(df, sv):
    ha = df[f"{sv}_Halpha"].to_numpy(float)
    F = np.column_stack([df[f"{sv}_{L}"].to_numpy(float) for L in CM_LINES])
    good = (ha > 0) & np.all(F > 0, axis=1)
    r = np.full((len(df), 8), np.nan); r[good] = np.log10(F[good]) - np.log10(ha[good])[:, None]
    logm = df[f"logm_{sv}"].to_numpy(float)
    z = df[f"z_{sv}"].to_numpy(float)
    logfha = np.where(ha > 0, np.log10(ha), np.nan)
    return r, logm, logfha, log10_lum(z, ha)


def fits_arrays(path, sv):
    t = Table.read(path, hdu=1); df = t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()
    if sv == "sdss":
        z = df["Z_1"].to_numpy(float); ha = df["H_ALPHA_FLUX"].to_numpy(float)
        cols = ["H_BETA_FLUX", "H_GAMMA_FLUX", "NII_6584_FLUX", "SII_6717_FLUX", "SII_6731_FLUX",
                "OII_3726_FLUX", "OII_3729_FLUX", "OIII_5007_FLUX"]
    else:
        z = df["Z"].to_numpy(float); ha = df["HALPHA_FLUX"].to_numpy(float)
        cols = ["HBETA_FLUX", "HGAMMA_FLUX", "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX",
                "OII_3726_FLUX", "OII_3729_FLUX", "OIII_5007_FLUX"]
    F = np.column_stack([df[c].to_numpy(float) for c in cols])
    logm = df["LOGM_COLOR"].to_numpy(float)
    good = (ha > 0) & np.all(F > 0, axis=1) & np.isfinite(z) & (z > 0) & np.isfinite(logm)
    r = np.full((len(df), 8), np.nan); r[good] = np.log10(F[good]) - np.log10(ha[good])[:, None]
    logfha = np.where(ha > 0, np.log10(ha), np.nan)
    return r, logm, logfha, log10_lum(z, ha), z


def match_weights(desi_zml, sdss_zml, nb=12):
    edges = [np.linspace(min(desi_zml[:, k].min(), sdss_zml[:, k].min()),
                         max(desi_zml[:, k].max(), sdss_zml[:, k].max()), nb + 1) for k in range(3)]
    Hs, _ = np.histogramdd(sdss_zml, bins=edges)
    Hd, _ = np.histogramdd(desi_zml, bins=edges)
    Hs = Hs / Hs.sum(); Hd = Hd / Hd.sum()
    ratio = np.where(Hd > 0, Hs / (Hd + 1e-12), 0.0)
    idx = tuple(np.clip(np.digitize(desi_zml[:, k], edges[k]) - 1, 0, nb - 1) for k in range(3))
    return ratio[idx]


# ---------- plotting ----------
def add_matrix(ax, M, title, kind="corr"):
    n = M.shape[0]
    if kind == "corr":
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    elif kind == "ratio":
        im = ax.imshow(M, cmap="PiYG", norm=TwoSlopeNorm(vcenter=1.0, vmin=0.7, vmax=1.3))
    else:  # diff (NF - data)
        im = ax.imshow(M, cmap="PuOr_r", vmin=-0.3, vmax=0.3)
    ax.set_title(title); ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(LAB10, rotation=90); ax.set_yticklabels(LAB10)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.0,
                        color="white" if (kind == "corr" and abs(v) > 0.6) else "black")
    return im


def ratio_matrix(nf, data, floor=0.1):
    R = nf / data
    R[np.abs(data) < floor] = np.nan          # unstable where |data corr| tiny
    np.fill_diagonal(R, 1.0)
    return R


def diff_matrix(nf, data):
    D = nf - data
    np.fill_diagonal(D, 0.0)
    return D


def make_fig(mats, outname, suptitle):
    # mats: dict with sdss_data, sdss_nf, desi_data, desi_nf (10x10)
    fig, ax = plt.subplots(2, 4, figsize=(25, 12.5), constrained_layout=True)
    rows = [("SDSS", "sdss"), ("DESI", "desi")]
    for r, (lab, k) in enumerate(rows):
        im_c = add_matrix(ax[r, 0], mats[f"{k}_data"], f"{lab} data", "corr")
        add_matrix(ax[r, 1], mats[f"{k}_nf"], f"{lab} NF", "corr")
        im_r = add_matrix(ax[r, 2], ratio_matrix(mats[f"{k}_nf"], mats[f"{k}_data"]), f"{lab} NF/data ratio", "ratio")
        im_d = add_matrix(ax[r, 3], diff_matrix(mats[f"{k}_nf"], mats[f"{k}_data"]), f"{lab} NF $-$ data", "diff")
    fig.colorbar(im_c, ax=list(ax[:, 0]) + list(ax[:, 1]), location="bottom", shrink=0.6, pad=0.04).set_label("correlation")
    fig.colorbar(im_r, ax=list(ax[:, 2]), location="bottom", shrink=0.7, pad=0.04).set_label("NF/data ratio")
    fig.colorbar(im_d, ax=list(ax[:, 3]), location="bottom", shrink=0.7, pad=0.04).set_label("NF $-$ data")
    fig.suptitle(suptitle, fontsize=15)
    fig.savefig(outname, dpi=160, bbox_inches="tight"); print("Saved:", outname, flush=True)


def summarize(tag, mats):
    # [NII] row (index 2), off-diagonal, first 8 lines: model error vs survey difference
    ii = NII
    sd = mats["sdss_data"][ii, :8]; dd = mats["desi_data"][ii, :8]
    sn = mats["sdss_nf"][ii, :8]; dn = mats["desi_nf"][ii, :8]
    off = [j for j in range(8) if j != ii]
    surv = np.mean(np.abs(dd[off] - sd[off]))
    err_s = np.mean(np.abs(sn[off] - sd[off])); err_d = np.mean(np.abs(dn[off] - dd[off]))
    print(f"\n[{tag}] [NII] off-diagonal correlations (line ratios):", flush=True)
    print(f"   SDSS data : {np.round(sd[off],2)}", flush=True)
    print(f"   DESI data : {np.round(dd[off],2)}", flush=True)
    print(f"   mean |DESI_data - SDSS_data| (SURVEY difference) = {surv:.3f}", flush=True)
    print(f"   mean |NF - data| SDSS = {err_s:.3f}   DESI = {err_d:.3f}  (MODEL error)", flush=True)


def main():
    ms = pickle.load(open(META_SDSS, "rb")); md = pickle.load(open(META_DESI, "rb"))
    fs, fd = load_flow(FLOW_SDSS, ms), load_flow(FLOW_DESI, md)

    # ===== CROSS-MATCHED =====
    df = pd.read_csv(CSV)
    rs, lms, fhas, lhas = cm_ratios(df, "sdss")
    rd, lmd, fhad, lhad = cm_ratios(df, "desi")
    cm = {
        "sdss_data": corr10(rs, lms, lhas),
        "desi_data": corr10(rd, lmd, lhad),
        "sdss_nf": corr10(sample_ratios(fs, ms, lms, lhas, seed=SEED + 1), lms, lhas),
        "desi_nf": corr10(sample_ratios(fd, md, lmd, lhad, seed=SEED + 2), lmd, lhad),
    }
    make_fig(cm, "corr_crossmatched.png", "Correlation matrices — CROSS-MATCHED sample (same galaxies)")
    summarize("cross-matched", cm)
    np.savez(REPO + "figs_ALTB/corr_matrices_cm.npz",
             sdss_data=cm["sdss_data"], sdss_nf=cm["sdss_nf"],
             desi_data=cm["desi_data"], desi_nf=cm["desi_nf"], labels=np.array(LAB10))
    print("Saved: corr_matrices_cm.npz", flush=True)

    # ===== BOOTSTRAP / POPULATION-MATCHED (DESI resampled to SDSS z,M*,L_Ha) =====
    rs2, lms2, fhas2, lhas2, zs = fits_arrays(SDSS_FITS, "sdss")
    rd2, lmd2, fhad2, lhad2, zd = fits_arrays(DESI_FITS, "desi")
    gd = np.all(np.isfinite(np.column_stack([rd2, lmd2, lhad2, zd])), axis=1)
    gs = np.all(np.isfinite(np.column_stack([rs2, lms2, lhas2, zs])), axis=1)
    desi_zml = np.column_stack([zd, lmd2, lhad2])[gd]
    sdss_zml = np.column_stack([zs, lms2, lhas2])[gs]
    w = match_weights(desi_zml, sdss_zml)                 # weight per (finite) DESI galaxy
    rng = np.random.default_rng(SEED)
    idx_all = np.where(gd)[0]
    p = w / w.sum()
    pick = rng.choice(len(idx_all), size=min(120000, (w > 0).sum() * 3), replace=True, p=p)
    sel = idx_all[pick]                                    # matched DESI subset (indices into full arrays)
    bm = {
        "sdss_data": corr10(rs2, lms2, lhas2),             # SDSS = full ALT-B (target)
        "sdss_nf": corr10(sample_ratios(fs, ms, lms2[gs], lhas2[gs], seed=SEED + 3), lms2[gs], lhas2[gs]),
        "desi_data": corr10(rd2[sel], lmd2[sel], lhad2[sel]),
        "desi_nf": corr10(sample_ratios(fd, md, lmd2[sel], lhad2[sel], seed=SEED + 4), lmd2[sel], lhad2[sel]),
    }
    make_fig(bm, "corr_bootstrap_matched.png",
             "Correlation matrices — DESI bootstrap-matched to SDSS (z, M*, L_Ha)")
    summarize("bootstrap-matched", bm)
    print(f"\n[bootstrap] DESI matched sample size: {len(sel):,} (from {int((w>0).sum()):,} in-SDSS-support DESI galaxies)", flush=True)


if __name__ == "__main__":
    main()
