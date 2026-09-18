#!/usr/bin/env python3
"""
In-survey low-z (z~0.07) emission-line LFs: SDSS-trained NF predictions vs Comparat+2016.
Uses the OBSERVED L_Ha (no SFR->L_Ha bridge) and a standard 1/Vmax estimator.

Parent = mpa_rcsed2_combo.fits (r<17.77 SDSS Main, NOT line-cut) + Ha S/N>3 (SF, gives L_Ha).
For each galaxy: NF predicts [OII]/[OIII]/Hb from (LOGM_COLOR_approx, logL_Ha); L_line = L_Ha + ratio.
=> the flow fills in [OII]/[OIII] even where individually weak, so the LF is NOT [OII]-selection-biased.

CAVEATS (print + figure): (i) LOGM_COLOR approximated from corrmag g,r (kcorr ignored; ratios ~insensitive
to M*, so OK for conditioning); (ii) Vmax from r<17.77 (DM-based, kcorr ignored) + Ha-flux limit;
(iii) absolute normalization uncertain (parent ~incomplete vs full Main + assumed area 8032 deg^2) ->
trust SHAPE and L* near/above the knee, not the vertical zero-point.
"""
import pickle, numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
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
                     "legend.fontsize": 10, "axes.titlesize": 14, "figure.dpi": 130})


def sp(m): print(m, flush=True)


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO = BASE + "nebular_emission_model/"
PARENT = BASE + "mpa_rcsed2_combo.fits"
FLUX_SCALE = 1e-17; RLIM = 17.77; AREA_DEG2 = 8032.0
ZLO, ZHI = 0.04, 0.099
Msun_r = 4.64; ZP = 0.271     # LOGM_COLOR recipe constants
IHB, IOII_A, IOII_B, IOIII = 0, 5, 6, 7
_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))

# Comparat+16 Table 7 Schechter (observed), evaluated at z-bin centre
COMP = {"OII": dict(logLs0=41.10, bL=2.33, logPs0=-2.40, bP=-0.73, alpha=-1.46),
        "Hb": dict(logLs0=40.88, bL=2.19, logPs0=-3.34, bP=2.70, alpha=-1.51),
        "OIII": dict(logLs0=41.42, bL=3.91, logPs0=-3.41, bP=-0.76, alpha=-1.83)}
LAB = {"OII": r"[OII]3727", "Hb": r"H$\beta$", "OIII": r"[OIII]5007"}


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def schechter(logL, logLs, logPs, a):
    x = 10 ** (logL - logLs); return np.log(10) * 10 ** logPs * x ** (a + 1) * np.exp(-x)


def load_flow():
    meta = pickle.load(open(REPO + "models/nf_sdss_ALTB_meta.pkl", "rb"))
    xd = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xd)), cond_dim=2, inverter=_INV)
    return eqx.tree_deserialise_leaves(REPO + "models/nf_sdss_ALTB.eqx", tmpl), meta


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


def main():
    t = Table.read(PARENT, hdu=1)
    z = np.asarray(t["Z_1"], float); rmag = np.asarray(t["corrmag_r"], float); gmag = np.asarray(t["corrmag_g"], float)
    Fha = np.asarray(t["H_ALPHA_FLUX"], float); eha = np.asarray(t["H_ALPHA_FLUX_ERR"], float)
    def obsflux(L):
        return np.asarray(t[L], float), np.asarray(t[L + "_ERR"], float)
    Foa, eoa = obsflux("OII_3726_FLUX"); Fob, eob = obsflux("OII_3729_FLUX")
    Fo3, eo3 = obsflux("OIII_5007_FLUX"); Fhb, ehb = obsflux("H_BETA_FLUX")
    snha = Fha / np.where(eha > 0, eha, np.nan)
    sel = ((z > ZLO) & (z < ZHI) & np.isfinite(rmag) & (rmag < RLIM) & (rmag > 10)
           & np.isfinite(gmag) & (snha > 3) & (Fha > 0))
    sp("parent r<%.2f, %.2f<z<%.2f, Ha S/N>3: N=%d" % (RLIM, ZLO, ZHI, sel.sum()))
    z, rmag, gmag, Fha, eha = z[sel], rmag[sel], gmag[sel], Fha[sel], eha[sel]
    Foa, eoa, Fob, eob = Foa[sel], eoa[sel], Fob[sel], eob[sel]
    Fo3, eo3, Fhb, ehb = Fo3[sel], eo3[sel], Fhb[sel], ehb[sel]

    # approx LOGM_COLOR from apparent g,r (kcorr ignored; band-shift 0.1 ~ z_med)
    DM = cosmo.distmod(z).value
    Mr = rmag - DM
    logm = (1.062 * (gmag - rmag) - 0.555) + (-0.4 * (Mr - Msun_r)) + ZP
    loglha = log10_lum(z, Fha)
    sp("logM* approx p16/50/84: %s ; logL_Ha p16/50/84: %s"
       % (np.round(np.percentile(logm, [16, 50, 84]), 2), np.round(np.percentile(loglha, [16, 50, 84]), 2)))

    # ---- Vmax ----
    zg = np.linspace(0.001, 0.5, 4000); dmg = cosmo.distmod(zg).value
    def z_of_dm(dm): return np.interp(dm, dmg, zg)
    # r-limit
    dm_max_r = RLIM - rmag + DM
    zmax_r = z_of_dm(dm_max_r)
    # Ha-flux limit: F(z') > 3 e  => dL(z')^2 < dL(z)^2 * (Fha/(3 e))
    dl = cosmo.luminosity_distance(z).to("cm").value
    dl_max_ha = dl * np.sqrt(np.clip(Fha / (3 * eha), 1e-6, None))
    dlg = cosmo.luminosity_distance(zg).to("cm").value
    zmax_ha = np.interp(dl_max_ha, dlg, zg)
    zmax = np.minimum.reduce([zmax_r, zmax_ha, np.full_like(z, ZHI)])
    zmin = np.full_like(z, ZLO)
    frac = AREA_DEG2 / 41252.96
    Vz = lambda zz: cosmo.comoving_volume(zz).to_value(u.Mpc ** 3)
    Vmax = np.clip(Vz(zmax) - Vz(zmin), 1e-30, None) * frac
    w = 1.0 / Vmax
    sp("median zmax=%.3f (r-lim %.3f, Ha-lim %.3f)" % (np.median(zmax), np.median(zmax_r), np.median(zmax_ha)))

    # ---- flow predictions ----
    flow, meta = load_flow()
    r8 = sample_one(flow, meta, logm, loglha, seed=13)
    Lpred = {"OII": loglha + np.log10(10 ** r8[:, IOII_A] + 10 ** r8[:, IOII_B]),
             "Hb": loglha + r8[:, IHB], "OIII": loglha + r8[:, IOIII]}
    # observed line luminosities (only where detected) for cross-check
    def obsL(F, e):
        good = (F > 0) & (e > 0) & (F / e > 3)
        L = np.full_like(z, np.nan); L[good] = log10_lum(z[good], F[good]); return L
    Lobs = {"OII": obsL(Foa + Fob, np.sqrt(eoa ** 2 + eob ** 2)), "Hb": obsL(Fhb, ehb), "OIII": obsL(Fo3, eo3)}

    # ---- LFs ----
    Lbins = np.arange(38.6, 43.2, 0.2); Lc = 0.5 * (Lbins[:-1] + Lbins[1:]); dlog = 0.2
    zc = 0.5 * (ZLO + ZHI)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    sp("\nline   logL*(Comp,z=%.2f)  mock_peakL  N" % zc)
    for ax, line in zip(axes, ["OII", "Hb", "OIII"]):
        c = COMP[line]; Ls = c["logLs0"] + c["bL"] * np.log10(1 + zc); Ps = c["logPs0"] + c["bP"] * np.log10(1 + zc)
        # model LF (1/Vmax)
        phi_pred, _ = np.histogram(Lpred[line], bins=Lbins, weights=w)
        Npred, _ = np.histogram(Lpred[line], bins=Lbins)
        phi_pred = phi_pred / dlog
        # observed LF (1/Vmax, detected only)
        m = np.isfinite(Lobs[line])
        phi_obs, _ = np.histogram(Lobs[line][m], bins=Lbins, weights=w[m]); phi_obs /= dlog
        gp = Npred >= 20
        ipk = np.argmax(np.where(Npred >= 20, phi_pred, 0)); Lcomp = Lc[ipk]
        comp = gp & (Lc > Lcomp)
        ax.plot(Lc[comp], phi_pred[comp], "o", ms=5, color="#CC79A7", label="NF prediction", zorder=3)
        ax.plot(Lc[gp & ~comp], phi_pred[gp & ~comp], "o", ms=4, mfc="none", mec="#CC79A7", alpha=0.5)
        mo = (Npred >= 20) & (phi_obs > 0) & (Lc > Lcomp)
        ax.plot(Lc[mo], phi_obs[mo], "s", ms=4, color="#0072B2", label="observed (this sample)", alpha=0.8)
        xx = np.linspace(Lcomp, 42.8, 60)
        ax.plot(xx, schechter(xx, Ls, Ps, c["alpha"]), "-", color="k", lw=1.8, label="Comparat+16")
        ax.set_yscale("log"); ax.set_xlim(39.2, 42.8); ax.set_ylim(1e-6, 3e-2)
        ax.set_xlabel(r"$\log_{10} L_{\mathrm{%s}}$ [erg s$^{-1}$]" % LAB[line].replace("$", "").replace(r"\beta", "b"))
        if line == "OII": ax.set_ylabel(r"$\Phi$ [Mpc$^{-3}$ dex$^{-1}$]")
        ax.set_title(LAB[line]); ax.legend(frameon=False, fontsize=9, loc="lower left")
        sp("%-5s  %.2f              %.2f       %d" % (line, Ls, Lcomp, sel.sum()))
    fig.suptitle(r"In-survey SDSS ($z\approx%.2f$): NF-predicted vs observed vs Comparat+16 LFs (shape/L$^*$; norm. caveated)" % zc,
                 fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = REPO + "figs_ALTB/hiz_insurvey_lf_sdss.png"; fig.savefig(out, bbox_inches="tight", dpi=150); sp("Saved: " + out)
    # implied number density sanity
    sp("\nsanity: total n = sum(1/Vmax) = %.3e Mpc^-3 (SF, r<17.77, z~%.2f); area assumed %.0f deg^2" % (w.sum(), zc, AREA_DEG2))
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
