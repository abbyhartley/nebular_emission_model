#!/usr/bin/env python3
"""
Population-level (fully predictive) high-z test:
  lightcone (M*, SFR)  ->  logL_Ha = Kennicutt(SFR)+Delta  ->  NF ratios  ->  emission-line LUMINOSITIES
  (NO observed lines used as input.)
Compare the PREDICTED line-luminosity distributions to the REAL DESI-COSMOS line luminosities in
matched redshift bins, applying the SAME selection: each mock galaxy gets per-line noise drawn from
the DESI data in that z-bin, and we require S/N>3 exactly like the data (mimics the flux limit incl.
Eddington bias).

Lines:  Ha (z<0.49 only; tests SFR->L_Ha bridge, no flow), [OIII]5007 (z<0.96), [OII]3727 total (z<1.6)
"""
import pickle, numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import ks_2samp
from functools import partial as _pf
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import scienceplots  # noqa
    plt.style.use(["science", "no-latex"])
except Exception:
    pass
plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 12, "axes.titlesize": 14, "figure.dpi": 130})


def sp(m): print(m, flush=True)


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
LC = "/oak/stanford/orgs/kipac/users/risahw/aemulus_lightcones/results/box017_v1"
FLUX_SCALE = 1e-17; KE = 41.27; MASS_ZP = 0.13
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
IHB, IOII_A, IOII_B, IOIII = 0, 5, 6, 7
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
ZBINS = [(0.30, 0.50), (0.50, 0.70), (0.70, 0.90)]
_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))


def log10_lum(z, f1e17):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f1e17, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def flux_from_logL(z, logL):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return 10.0 ** np.asarray(logL, float) / (4 * np.pi * dl ** 2) / FLUX_SCALE   # -> 1e-17 units


def load_flow():
    meta = pickle.load(open(REPO + "models/nf_desi_ALTB_meta.pkl", "rb"))
    xdim = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xdim)), cond_dim=2, inverter=_INV)
    return eqx.tree_deserialise_leaves(REPO + "models/nf_desi_ALTB.eqx", tmpl), meta


def sample_one(flow, meta, logm, loglha, seed, batch=40000):
    U = np.column_stack([logm, loglha]).astype(np.float32)
    Un_all = jnp.asarray((U - meta["U_mean"]) / meta["U_std"])
    Xm, Xs = np.asarray(meta["X_mean"]), np.asarray(meta["X_std"]); n = len(U)
    out = np.zeros((n, 8)); key = jr.key(seed + 7)
    for lo in range(0, n, batch):
        hi = min(n, lo + batch); Un = Un_all[lo:hi]
        key, sk = jr.split(key); keys = jr.split(sk, hi - lo)
        Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un))
        out[lo:hi] = Xn * Xs + Xm
    return out


def decode(a):
    a = np.asarray(a)
    if a.dtype.kind == "S":
        a = np.char.decode(a)
    return np.char.strip(a.astype(str))


def main():
    rng = np.random.default_rng(11)
    h = fits.open(COSMOS); d1, d6 = h[1].data, h[6].data
    z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
    stype = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float); lpsfr = np.asarray(d6["lp_SFR_med"], float)
    base = (zwarn == 0) & (stype == "GALAXY") & (z > 0.05)

    def cal_flux_sig(name):
        raw = np.asarray(d1[name + "_FLUX"], float); iv = np.asarray(d1[name + "_FLUX_IVAR"], float)
        f = raw * 10.0 ** CAL[name]; sig = np.where(iv > 0, 1.0 / np.sqrt(np.clip(iv, 1e-30, None)), np.nan) * 10.0 ** CAL[name]
        sn = raw * np.sqrt(np.clip(iv, 0, None))
        return f, sig, sn
    Fha, Sha, SNha = cal_flux_sig("HALPHA"); Fo3, So3, SNo3 = cal_flux_sig("OIII_5007")
    Foa, Soa, SNoa = cal_flux_sig("OII_3726"); Fob, Sob, SNob = cal_flux_sig("OII_3729")
    # [OII] total (calibrated)
    Foii = Foa + Fob; Soii = np.sqrt(Soa ** 2 + Sob ** 2); SNoii = Foii / Soii

    # ------- dust/aperture bridge Delta from z<0.49 (obs L_Ha vs Kennicutt(lp_SFR)) -------
    lha_obs = log10_lum(z, Fha)
    lpsfr_log = lpsfr if np.nanmedian(np.abs(lpsfr[np.isfinite(lpsfr)])) < 5 else np.log10(np.clip(lpsfr, 1e-4, None))
    cal = base & (z < 0.49) & (SNha > 3) & (Fha > 0) & np.isfinite(lpsfr_log) & (lpsfr_log > -3) & (lpsfr_log < 3) & (lpm > 6)
    dv = (lha_obs - (lpsfr_log + KE))[cal]; Delta = float(np.median(dv[np.isfinite(dv)]))
    sp("Delta = %.3f dex (N=%d)" % (Delta, np.isfinite(dv).sum()))

    # ------- lightcone mock -------
    tab = pq.read_table(LC + "/desi_cosmos_z1.60_um_sm.parquet",
                        columns=["z_obs", "obs_sm", "obs_sfr", "ssfr"]).to_pandas()
    zl = tab["z_obs"].to_numpy(float); ml = np.log10(np.clip(tab["obs_sm"].to_numpy(float), 1, None))
    sfrl = tab["obs_sfr"].to_numpy(float); lsl = np.where(sfrl > 0, np.log10(sfrl), np.nan)
    lha_lc = lsl + KE + Delta
    sf = (sfrl > 0) & (tab["ssfr"].to_numpy(float) > 1e-11) & (zl >= 0.28) & (zl < 0.92) & np.isfinite(ml)
    ix = np.where(sf)[0]
    if len(ix) > 160000:
        ix = rng.choice(ix, 160000, replace=False)
    sp("lightcone SF mock in 0.28<z<0.92: %d (using %d)" % (int(sf.sum()), len(ix)))
    flow, meta = load_flow()
    r8 = sample_one(flow, meta, ml[ix], lha_lc[ix], seed=77)
    zlc = zl[ix]; lha_m = lha_lc[ix]; logm_m = ml[ix]
    inbox_m = (logm_m >= BOX_M[0]) & (logm_m <= BOX_M[1]) & (lha_m >= BOX_L[0]) & (lha_m <= BOX_L[1])
    # predicted mock luminosities (log10 erg/s)
    L_ha_m = lha_m
    L_o3_m = lha_m + r8[:, IOIII]
    L_oii_m = lha_m + np.log10(10.0 ** r8[:, IOII_A] + 10.0 ** r8[:, IOII_B])

    # ------- comparison config -------
    LINES = {
        "Halpha":   dict(lab=r"$\log_{10} L_{\mathrm{H}\alpha}$",     Ld=lha_obs, sig=Sha, sn=SNha, Lm=L_ha_m, zmax=0.49),
        "OIII5007": dict(lab=r"$\log_{10} L_{\mathrm{[OIII]}5007}$",  Ld=log10_lum(z, Fo3), sig=So3, sn=SNo3, Lm=L_o3_m, zmax=0.96),
        "OII3727":  dict(lab=r"$\log_{10} L_{\mathrm{[OII]}3727}$",   Ld=log10_lum(z, Foii), sig=Soii, sn=SNoii, Lm=L_oii_m, zmax=1.60),
    }

    def detect_mock(Lm_bin, zbin_mock, sig_pool):
        Ft = flux_from_logL(zbin_mock, Lm_bin)
        sig = rng.choice(sig_pool, size=len(Ft))
        Fo = Ft + rng.normal(0, 1, size=len(Ft)) * sig
        sn = Fo / sig; keep = (sn > 3) & (Fo > 0)
        return log10_lum(zbin_mock[keep], Fo[keep]), keep

    nrow, ncol = len(LINES), len(ZBINS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.7 * nrow))
    sp("\nline        zbin        N_data  N_mock  med_data med_mock  d(med)   KS   inbox%")
    for ri, (lname, cfg) in enumerate(LINES.items()):
        for ci, (za, zb) in enumerate(ZBINS):
            ax = axes[ri, ci]
            if za >= cfg["zmax"]:
                ax.text(0.5, 0.5, "out of DESI window", ha="center", va="center", transform=ax.transAxes,
                        fontsize=12, color="0.4"); ax.set_xticks([]); ax.set_yticks([])
                if ci == 0: ax.set_ylabel(cfg["lab"])
                if ri == 0: ax.set_title(r"$%.1f<z<%.1f$" % (za, zb))
                continue
            zc = min(zb, cfg["zmax"])
            dsel = base & (z >= za) & (z < zc) & (cfg["sn"] > 3) & np.isfinite(cfg["Ld"])
            Ld = cfg["Ld"][dsel]
            sig_pool = cfg["sig"][base & (z >= za) & (z < zc) & np.isfinite(cfg["sig"]) & (cfg["sig"] > 0)]
            mmask = (zlc >= za) & (zlc < zc)
            Lm, keepm = detect_mock(cfg["Lm"][mmask], zlc[mmask], sig_pool)
            ibm = inbox_m[mmask][keepm]
            if len(Ld) < 20 or len(Lm) < 20:
                ax.text(0.5, 0.5, "too few", ha="center", va="center", transform=ax.transAxes)
            else:
                lo, hi = np.percentile(np.concatenate([Ld, Lm]), [1, 99]); bins = np.linspace(lo, hi, 34)
                ax.hist(Ld, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color="#0072B2", label="DESI-COSMOS")
                ax.hist(Lm, bins=bins, density=True, histtype="step", lw=2.3, color="#CC79A7", label="lightcone$\\times$NF")
                dmed = np.median(Lm) - np.median(Ld); ks = ks_2samp(Ld, Lm).statistic
                ax.text(0.03, 0.97, r"$\Delta_{\rm med}%+.2f$" % dmed + "\nKS %.2f" % ks, transform=ax.transAxes,
                        va="top", fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
                sp("%-9s [%.1f,%.1f]  %6d  %6d   %6.2f  %6.2f   %+.2f  %.2f  %4.0f"
                   % (lname, za, zb, len(Ld), len(Lm), np.median(Ld), np.median(Lm), dmed, ks, 100 * ibm.mean()))
            ax.set_yticks([])
            ax.set_xlabel(cfg["lab"])
            if ci == 0:
                ax.set_ylabel("density")
            if ri == 0:
                ax.set_title(r"$%.1f<z<%.1f$" % (za, zb))
            if ri == 0 and ci == 0:
                ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.suptitle("Predicted (lightcone $\\times$ NF) vs observed DESI-COSMOS emission-line luminosities (matched selection)",
                 fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = REPO + "figs_ALTB/hiz_lc_luminosity.png"; fig.savefig(out, bbox_inches="tight", dpi=150)
    sp("Saved: " + out); sp("=== DONE ===")


if __name__ == "__main__":
    main()
