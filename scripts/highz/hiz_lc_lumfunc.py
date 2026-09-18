#!/usr/bin/env python3
"""
Emission-line LUMINOSITY FUNCTIONS from the lightcone x NF mock, vs Comparat et al. 2016
(observed-frame [OII]3727, Hb, [OIII]5007 Schechter LFs). The lightcone's known area
(15.2 deg^2 effective) gives an absolute survey volume, so we can build absolute Phi(L).

Mock chain (fully predictive, no observed lines):
  lightcone (M*, SFR) -> fiber L_Ha = logSFR+41.27+Delta  (condition NF, in training domain)
                      -> NF observed ratios /Ha
  total observed L_Ha = fiber L_Ha - log10 f_ap(z_cal)   (aperture-correct fiber->total; Comparat is total)
  L_line(total,obs)   = total L_Ha + ratio
Restrict to lightcone mass-complete + flow-valid: logM* > 9.0.
Compare to Comparat+2016 Table 7 Schechter (observed) evaluated at each z-shell centre.
"""
import pickle, numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import pyarrow.parquet as pq
from astropy.io import fits
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
                     "legend.fontsize": 11, "axes.titlesize": 14, "figure.dpi": 130})


def sp(m): print(m, flush=True)


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
LC = "/oak/stanford/orgs/kipac/users/risahw/aemulus_lightcones/results/box017_v1"
FLUX_SCALE = 1e-17; KE = 41.27; MASS_ZP = 0.13; AREA_DEG2 = 15.2
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
IHB, IOII_A, IOII_B, IOIII = 0, 5, 6, 7
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20); MASS_COMPLETE = 9.0
_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))

# Comparat+2016 Table 7 Schechter (observed): L*(z)=L*0 (1+z)^bL ; Phi*(z)=Phi*0 (1+z)^bP ; alpha const
COMP = {
    "OII":  dict(logLs0=41.10, bL=2.33, logPs0=-2.40, bP=-0.73, alpha=-1.46, zr=(0.5, 1.3)),
    "Hb":   dict(logLs0=40.88, bL=2.19, logPs0=-3.34, bP=2.70, alpha=-1.51, zr=(0.3, 0.83)),
    "OIII": dict(logLs0=41.42, bL=3.91, logPs0=-3.41, bP=-0.76, alpha=-1.83, zr=(0.3, 0.83)),
}
SHELLS = {"OII": [(0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.3)],
          "Hb": [(0.3, 0.5), (0.5, 0.7), (0.7, 0.83)],
          "OIII": [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]}
LAB = {"OII": r"$L_{\mathrm{[OII]}3727}$", "Hb": r"$L_{\mathrm{H}\beta}$", "OIII": r"$L_{\mathrm{[OIII]}5007}$"}


def log10_lum(z, f):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def schechter_perdex(logL, logLs, logPs, alpha):
    x = 10 ** (logL - logLs)
    return np.log(10) * 10 ** logPs * x ** (alpha + 1) * np.exp(-x)


def shell_volume(z1, z2):
    v = (cosmo.comoving_volume(z2) - cosmo.comoving_volume(z1)).to(u.Mpc ** 3).value
    return v * (AREA_DEG2 / 41252.96)


def load_flow():
    meta = pickle.load(open(REPO + "models/nf_desi_ALTB_meta.pkl", "rb"))
    xd = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xd)), cond_dim=2, inverter=_INV)
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
    rng = np.random.default_rng(5)
    # ---- Delta + aperture from DESI-COSMOS ----
    h = fits.open(COSMOS); d1, d6, d2 = h[1].data, h[6].data, h[2].data
    z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
    stype = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float); lpsfr = np.asarray(d6["lp_SFR_med"], float)
    Fha = np.asarray(d1["HALPHA_FLUX"], float) * 10 ** CAL["HALPHA"]
    SNha = np.asarray(d1["HALPHA_FLUX"], float) * np.sqrt(np.clip(np.asarray(d1["HALPHA_FLUX_IVAR"], float), 0, None))
    ff = np.asarray(d2["FIBERFLUX_R"], float); tf = np.asarray(d2["FLUX_R"], float)
    fap = np.clip(np.where((tf > 0) & (ff > 0), ff / tf, np.nan), 1e-3, 1.2)
    base = (zwarn == 0) & (stype == "GALAXY") & (z > 0.05) & np.isfinite(lpm) & (lpm > 6) & (lpm < 13)
    lpsfr_log = lpsfr if np.nanmedian(np.abs(lpsfr[np.isfinite(lpsfr) & base])) < 5 else np.log10(np.clip(lpsfr, 1e-4, None))
    calm = base & (z < 0.49) & (SNha > 3) & (Fha > 0) & np.isfinite(lpsfr_log) & (lpsfr_log > -3) & (lpsfr_log < 3)
    Delta = float(np.median((log10_lum(z, Fha) - (lpsfr_log + KE))[calm]))
    fap_cal = float(np.nanmedian(fap[calm]))
    ap_corr = -np.log10(fap_cal)      # fiber -> total (add this, dex)
    sp("Delta=%.3f  f_ap(cal)=%.3f  aperture fiber->total corr=+%.3f dex" % (Delta, fap_cal, ap_corr))

    # ---- lightcone mock ----
    tab = pq.read_table(LC + "/desi_cosmos_z1.60_um_sm.parquet",
                        columns=["z_obs", "obs_sm", "obs_sfr", "ssfr"]).to_pandas()
    zl = tab["z_obs"].to_numpy(float); ml = np.log10(np.clip(tab["obs_sm"].to_numpy(float), 1, None))
    sfrl = tab["obs_sfr"].to_numpy(float); lsl = np.where(sfrl > 0, np.log10(sfrl), np.nan)
    sf = (sfrl > 0) & (tab["ssfr"].to_numpy(float) > 1e-11) & (zl >= 0.30) & (zl < 1.30) & (ml > MASS_COMPLETE) & np.isfinite(ml)
    ix = np.where(sf)[0]
    sp("lightcone SF, logM*>%.1f, 0.3<z<1.3: %d" % (MASS_COMPLETE, len(ix)))
    zlc = zl[ix]; mlc = ml[ix]; lsfr = lsl[ix]
    lha_fiber = lsfr + KE + Delta                     # for NF conditioning (training/fiber scale)
    lha_total = lha_fiber + ap_corr                   # total observed
    flow, meta = load_flow()
    r8 = sample_one(flow, meta, mlc, lha_fiber, seed=31)
    Ltot = {"OII": lha_total + np.log10(10 ** r8[:, IOII_A] + 10 ** r8[:, IOII_B]),
            "Hb": lha_total + r8[:, IHB], "OIII": lha_total + r8[:, IOIII]}
    frac_extrap = np.mean((lha_fiber < BOX_L[0]) | (lha_fiber > BOX_L[1]))
    sp("fraction with fiber L_Ha outside training box (extrapolated): %.1f%%" % (100 * frac_extrap))

    # ---- LFs + figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    cmap = plt.get_cmap("viridis")
    Lbins = np.arange(39.6, 43.4, 0.2); Lc = 0.5 * (Lbins[:-1] + Lbins[1:]); dlog = 0.2
    for ax, line in zip(axes, ["OII", "Hb", "OIII"]):
        shells = SHELLS[line]; c = COMP[line]
        sp("\n=== %s LF ===" % line)
        for k, (z1, z2) in enumerate(shells):
            col = cmap(k / max(1, len(shells) - 1))
            s = (zlc >= z1) & (zlc < z2)
            V = shell_volume(z1, z2)
            L = Ltot[line][s]
            N, _ = np.histogram(L, bins=Lbins)
            phi = N / (V * dlog); ephi = np.sqrt(N) / (V * dlog)
            zc = 0.5 * (z1 + z2)
            Ls = c["logLs0"] + c["bL"] * np.log10(1 + zc); Ps = c["logPs0"] + c["bP"] * np.log10(1 + zc)
            good = N >= 10
            ax.errorbar(Lc[good], phi[good], yerr=ephi[good], fmt="o", ms=4, color=col, capsize=2,
                        label=r"mock $%.1f<z<%.1f$" % (z1, z2))
            xx = np.linspace(40, 43.2, 100)
            ax.plot(xx, schechter_perdex(xx, Ls, Ps, c["alpha"]), "-", color=col, lw=1.8, alpha=0.9)
            # peak L of mock vs L*
            sp("  z[%.1f,%.1f] V=%.2e Mpc^3  N=%d  logL*(Comparat)=%.2f  mock median logL=%.2f"
               % (z1, z2, V, s.sum(), Ls, np.median(L)))
        ax.set_yscale("log"); ax.set_xlim(40.2, 43.2); ax.set_ylim(1e-6, 5e-2)
        ax.set_xlabel(r"$\log_{10}$ %s [erg s$^{-1}$]" % LAB[line])
        if line == "OII":
            ax.set_ylabel(r"$\Phi$ [Mpc$^{-3}$ dex$^{-1}$]")
        ax.set_title(r"%s   (points: mock $\times$NF; lines: Comparat+16)" % LAB[line].replace("$L_{", "").replace("}$", "").replace(r"\mathrm", ""))
        ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(r"Lightcone $\times$ NF emission-line luminosity functions vs Comparat et al. 2016 (observed)", fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = REPO + "figs_ALTB/hiz_lc_lumfunc.png"; fig.savefig(out, bbox_inches="tight", dpi=150); sp("Saved: " + out)
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
