#!/usr/bin/env python3
"""
Follow-ups on the lightcone luminosity test:
 (a) shift each mock distribution to align medians -> recompute KS (is the residual just a shift?)
 (b) replace constant dust/aperture bridge Delta with a z-DEPENDENT aperture bridge:
        logL_Ha_obs = logSFR + 41.27 + log10 f_ap(z) + C
     f_ap(z) = median fiber/total flux ratio (FIBERFLUX_R/FLUX_R) measured from DESI-COSMOS vs z
               (fixed 1.5" fiber captures a larger physical fraction at higher z -> less aperture loss)
     C       = constant dust+SFR-calibration residual, calibrated at z<0.49 so it matches the data there.
Compares constant-Delta vs z-bridge vs shift-aligned, per line x z-bin.
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


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
LC = "/oak/stanford/orgs/kipac/users/risahw/aemulus_lightcones/results/box017_v1"
FLUX_SCALE = 1e-17; KE = 41.27; MASS_ZP = 0.13
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
IHB, IOII_A, IOII_B, IOIII = 0, 5, 6, 7
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
ZBINS = [(0.30, 0.50), (0.50, 0.70), (0.70, 0.90)]
_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def flux_from_logL(z, logL):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return 10.0 ** np.asarray(logL, float) / (4 * np.pi * dl ** 2) / FLUX_SCALE


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
    h = fits.open(COSMOS); d1, d6, d2 = h[1].data, h[6].data, h[2].data
    z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
    stype = decode(d1["SPECTYPE"]); lpsfr = np.asarray(d6["lp_SFR_med"], float)
    lpm = np.asarray(d6["lp_mass_med"], float)
    valid = np.isfinite(lpm) & (lpm > 6) & (lpm < 13)     # require real COSMOS2020 fit (SFR+mass come together)
    base = (zwarn == 0) & (stype == "GALAXY") & (z > 0.05)

    def cal_flux_sig(name):
        raw = np.asarray(d1[name + "_FLUX"], float); iv = np.asarray(d1[name + "_FLUX_IVAR"], float)
        f = raw * 10.0 ** CAL[name]
        sig = np.where(iv > 0, 1.0 / np.sqrt(np.clip(iv, 1e-30, None)), np.nan) * 10.0 ** CAL[name]
        sn = raw * np.sqrt(np.clip(iv, 0, None))
        return f, sig, sn
    Fha, Sha, SNha = cal_flux_sig("HALPHA"); Fo3, So3, SNo3 = cal_flux_sig("OIII_5007")
    Foa, Soa, SNoa = cal_flux_sig("OII_3726"); Fob, Sob, SNob = cal_flux_sig("OII_3729")
    Foii = Foa + Fob; Soii = np.sqrt(Soa ** 2 + Sob ** 2); SNoii = Foii / Soii

    # ---- aperture fraction f_ap(z) from fiber/total r-band ----
    ff = np.asarray(d2["FIBERFLUX_R"], float); tf = np.asarray(d2["FLUX_R"], float)
    fap = np.where((tf > 0) & (ff > 0), ff / tf, np.nan)
    fap = np.clip(fap, 1e-3, 1.2)
    zg = np.linspace(0.05, 1.6, 32); zc_ap = 0.5 * (zg[:-1] + zg[1:])
    fap_med = np.array([np.nanmedian(fap[base & (z >= a) & (z < b)]) if (base & (z >= a) & (z < b)).sum() > 30 else np.nan
                        for a, b in zip(zg[:-1], zg[1:])])
    ok = np.isfinite(fap_med)
    fap_of_z = lambda zz: np.interp(zz, zc_ap[ok], fap_med[ok])
    sp("f_ap(z) median: " + "  ".join("z%.2f:%.2f" % (c, f) for c, f in zip(zc_ap[ok][::3], fap_med[ok][::3])))

    # ---- calibrate constant C (dust+SFRcal) using z<0.49, population f_ap(z) ----
    lha_obs = log10_lum(z, Fha)
    _v = valid & np.isfinite(lpsfr)
    lpsfr_log = lpsfr if np.nanmedian(np.abs(lpsfr[_v])) < 5 else np.log10(np.clip(lpsfr, 1e-4, None))
    calm = valid & base & (z < 0.49) & (SNha > 3) & (Fha > 0) & np.isfinite(lpsfr_log) & (lpsfr_log > -3) & (lpsfr_log < 3)
    resid = (lha_obs - (lpsfr_log + KE) - np.log10(fap_of_z(z)))[calm]
    C = float(np.median(resid[np.isfinite(resid)]))
    Delta_const = float(np.median((lha_obs - (lpsfr_log + KE))[calm]))   # old constant bridge
    sp("z-bridge C (dust+SFRcal) = %.3f ; implied f_ap(z_cal_med)=%.2f ; old constant Delta=%.3f"
       % (C, np.median(fap_of_z(z[calm])), Delta_const))

    # ---- lightcone mock ----
    tab = pq.read_table(LC + "/desi_cosmos_z1.60_um_sm.parquet",
                        columns=["z_obs", "obs_sm", "obs_sfr", "ssfr"]).to_pandas()
    zl = tab["z_obs"].to_numpy(float); ml = np.log10(np.clip(tab["obs_sm"].to_numpy(float), 1, None))
    sfrl = tab["obs_sfr"].to_numpy(float); lsl = np.where(sfrl > 0, np.log10(sfrl), np.nan)
    sf = (sfrl > 0) & (tab["ssfr"].to_numpy(float) > 1e-11) & (zl >= 0.28) & (zl < 0.92) & np.isfinite(ml)
    ix = np.where(sf)[0]
    if len(ix) > 130000:
        ix = rng.choice(ix, 130000, replace=False)
    zlc = zl[ix]; logm_m = ml[ix]; lsfr_m = lsl[ix]
    lha_const = lsfr_m + KE + Delta_const
    lha_zdep = lsfr_m + KE + np.log10(fap_of_z(zlc)) + C
    sp("mock N=%d ; median lha_const=%.2f lha_zdep=%.2f" % (len(ix), np.median(lha_const), np.median(lha_zdep)))

    flow, meta = load_flow()
    r_const = sample_one(flow, meta, logm_m, lha_const, seed=77)
    r_zdep = sample_one(flow, meta, logm_m, lha_zdep, seed=91)

    def lums(lha, r):
        return dict(Halpha=lha, OIII5007=lha + r[:, IOIII],
                    OII3727=lha + np.log10(10.0 ** r[:, IOII_A] + 10.0 ** r[:, IOII_B]))
    Lm_const = lums(lha_const, r_const); Lm_zdep = lums(lha_zdep, r_zdep)

    LINES = {"Halpha": (r"$\log_{10} L_{\mathrm{H}\alpha}$", lha_obs, Sha, SNha, 0.49),
             "OIII5007": (r"$\log_{10} L_{\mathrm{[OIII]}5007}$", log10_lum(z, Fo3), So3, SNo3, 0.96),
             "OII3727": (r"$\log_{10} L_{\mathrm{[OII]}3727}$", log10_lum(z, Foii), Soii, SNoii, 1.60)}

    def detect(Lm_bin, zb_mock, sig_pool):
        Ft = flux_from_logL(zb_mock, Lm_bin); sig = rng.choice(sig_pool, size=len(Ft))
        Fo = Ft + rng.normal(0, 1, size=len(Ft)) * sig; sn = Fo / sig; keep = (sn > 3) & (Fo > 0)
        return log10_lum(zb_mock[keep], Fo[keep])

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.2))
    sp("\nline       zbin      Ndata  | const: dmed  KS  KSshift | zbridge: dmed  KS")
    for ri, (lname, (lab, Ld_all, sig_all, sn_all, zmax)) in enumerate(LINES.items()):
        for ci, (za, zb) in enumerate(ZBINS):
            ax = axes[ri, ci]
            if za >= zmax:
                ax.text(0.5, 0.5, "out of DESI window", ha="center", va="center", transform=ax.transAxes, color="0.4")
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0: ax.set_title(r"$%.1f<z<%.1f$" % (za, zb))
                if ci == 0: ax.set_ylabel("density")
                continue
            zc = min(zb, zmax)
            dsel = base & (z >= za) & (z < zc) & (sn_all > 3) & np.isfinite(Ld_all)
            Ld = Ld_all[dsel]
            sig_pool = sig_all[base & (z >= za) & (z < zc) & np.isfinite(sig_all) & (sig_all > 0)]
            mm = (zlc >= za) & (zlc < zc)
            Lc = detect(Lm_const[lname][mm], zlc[mm], sig_pool)
            Lz = detect(Lm_zdep[lname][mm], zlc[mm], sig_pool)
            if len(Ld) < 20 or len(Lz) < 20:
                ax.text(0.5, 0.5, "too few", ha="center", va="center", transform=ax.transAxes); continue
            dmed_c = np.median(Lc) - np.median(Ld); ks_c = ks_2samp(Ld, Lc).statistic
            ks_c_shift = ks_2samp(Ld, Lc - dmed_c).statistic     # (a) align medians
            dmed_z = np.median(Lz) - np.median(Ld); ks_z = ks_2samp(Ld, Lz).statistic
            sp("%-9s [%.1f,%.1f] %6d |  %+.2f %.2f %.2f  |  %+.2f %.2f"
               % (lname, za, zb, len(Ld), dmed_c, ks_c, ks_c_shift, dmed_z, ks_z))
            lo, hi = np.percentile(np.concatenate([Ld, Lz]), [1, 99]); bins = np.linspace(lo, hi, 34)
            ax.hist(Ld, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color="#0072B2", label="DESI-COSMOS")
            ax.hist(Lz, bins=bins, density=True, histtype="step", lw=2.3, color="#009E73", label="z-bridge mock")
            ax.hist(Lc, bins=bins, density=True, histtype="step", lw=1.6, ls=":", color="#CC79A7", label="const-$\\Delta$ mock")
            ax.text(0.03, 0.97, r"z-bridge: $\Delta_{\rm med}%+.2f$, KS %.2f" % (dmed_z, ks_z) +
                    "\n" + r"const: $\Delta_{\rm med}%+.2f$, KS %.2f" % (dmed_c, ks_c) +
                    "\n" + r"shift-aligned KS %.2f" % ks_c_shift,
                    transform=ax.transAxes, va="top", fontsize=9.5,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
            ax.set_yticks([]); ax.set_xlabel(lab)
            if ci == 0: ax.set_ylabel("density")
            if ri == 0: ax.set_title(r"$%.1f<z<%.1f$" % (za, zb))
            if ri == 0 and ci == 0: ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.suptitle("Lightcone$\\times$NF line luminosities: z-dependent aperture bridge vs constant $\\Delta$ vs data",
                 fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = REPO + "figs_ALTB/hiz_lc_lum_zbridge.png"; fig.savefig(out, bbox_inches="tight", dpi=150); sp("Saved: " + out)

    # small f_ap(z) figure
    f2, a2 = plt.subplots(figsize=(5.2, 4))
    a2.plot(zc_ap[ok], fap_med[ok], "-o", color="#009E73")
    a2.set_xlabel("redshift"); a2.set_ylabel(r"median $f_{\rm ap}=$ FIBERFLUX$_r$/FLUX$_r$")
    a2.set_title("DESI 1.5\" fiber aperture fraction vs z")
    f2.tight_layout(); out2 = REPO + "figs_ALTB/hiz_aperture_fap.png"; f2.savefig(out2, dpi=150); sp("Saved: " + out2)
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
