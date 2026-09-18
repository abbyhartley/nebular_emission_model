#!/usr/bin/env python3
"""
COMPLEMENTARY high-z check: lightcone as a mock emission-line survey.

Push the Aemulus/UniverseMachine lightcone (M*, SFR) through the DESI-trained NF to build a
mock high-z emission-line catalog, then compare its aggregate line-ratio statistics to the
REAL DESI-COSMOS observations at matched redshift (0.49<z<1.0).

Dust/aperture bridge (SFR is intrinsic; the flow wants OBSERVED L_Ha):
  logL_Ha_obs = logSFR + 41.27 + Delta,  where Delta is calibrated EMPIRICALLY from DESI-COSMOS
  z<0.49 galaxies that have BOTH observed L_Ha AND a COSMOS2020 SED SFR (lp_SFR_med):
      Delta = median[ logL_Ha_obs - (log10(lp_SFR) + 41.27) ]
  (folds in attenuation + aperture + SFR-calibration; a constant is used to avoid cross-scale
   M* issues between UM and LePhare masses.)

Caveat reported on the figure: the lightcone is ~volume-limited while DESI-COSMOS is flux-limited,
so residual distribution differences reflect selection + evolution, not only the flow.
"""
import pickle, numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
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
import cmasher as cmr
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "legend.fontsize": 13, "axes.titlesize": 15, "figure.dpi": 130})


def sp(m): print(m, flush=True)


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
LC = "/oak/stanford/orgs/kipac/users/risahw/aemulus_lightcones/results/box017_v1"
FLUX_SCALE = 1e-17; KE = 41.27; MASS_ZP = 0.13
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OIII_5007": 0.0502}
IHB, INII, IOIII = 0, 2, 7
BOX_M = (8.80, 11.05); BOX_L = (39.90, 42.20)
_ROBUST_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))


def log10_lum(z, f1e17):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f1e17, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def load_flow():
    meta = pickle.load(open(REPO + "models/nf_desi_ALTB_meta.pkl", "rb"))
    xdim = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xdim)), cond_dim=2, inverter=_ROBUST_INV)
    return eqx.tree_deserialise_leaves(REPO + "models/nf_desi_ALTB.eqx", tmpl), meta


def sample_one(flow, meta, logm, loglha, seed, batch=40_000):
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
    rng = np.random.default_rng(7)
    flow, meta = load_flow()

    # ---- DESI-COSMOS: calibrate Delta and get observed [OIII]/Hb at 0.49<z<1 ----
    h = fits.open(COSMOS); d1, d6 = h[1].data, h[6].data
    z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
    stype = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float)
    lpsfr = np.asarray(d6["lp_SFR_med"], float)
    def cflux(n):
        f = np.asarray(d1[n + "_FLUX"], float) * 10.0 ** CAL[n]
        sn = np.asarray(d1[n + "_FLUX"], float) * np.sqrt(np.clip(np.asarray(d1[n + "_FLUX_IVAR"], float), 0, None))
        return f, sn
    Fha, SNha = cflux("HALPHA"); Fhb, SNhb = cflux("HBETA"); Fo3, SNo3 = cflux("OIII_5007")
    base = (zwarn == 0) & (stype == "GALAXY") & np.isfinite(lpm) & (lpm > 6) & (lpm < 13) & (z > 0.05)
    logm_co = lpm + MASS_ZP

    # Delta calibration on z<0.49 with observed Ha + valid SED SFR
    cal = base & (z < 0.49) & (SNha > 3) & (Fha > 0) & np.isfinite(lpsfr) & (lpsfr > -3) & (lpsfr < 3)
    lha_obs = log10_lum(z, Fha)
    lha_ke = lpsfr + KE               # lp_SFR_med is log10(SFR) already? check units below
    # detect whether lp_SFR is log or linear
    if np.nanmedian(np.abs(lpsfr[cal])) > 5:       # linear SFR
        lha_ke = np.log10(np.clip(lpsfr, 1e-4, None)) + KE
    dvec = (lha_obs - lha_ke)[cal]
    Delta = float(np.median(dvec[np.isfinite(dvec)]))
    sp("Delta (logL_Ha_obs - Kennicutt(SFR)) = %.3f dex  (N=%d, NMAD=%.3f)"
       % (Delta, np.isfinite(dvec).sum(), 1.4826 * np.median(np.abs(dvec[np.isfinite(dvec)] - Delta))))

    # observed [OIII]/Hb, 0.49<z<1, in-box (L_Ha from Hb x 3.9 for box test, matching primary)
    obsm = base & (z >= 0.49) & (z < 1.0) & (SNhb > 3) & (SNo3 > 3) & (Fhb > 0) & (Fo3 > 0)
    loglha_co = log10_lum(z, Fhb) + np.log10(3.90)
    inbox_co = (logm_co >= BOX_M[0]) & (logm_co <= BOX_M[1]) & (loglha_co >= BOX_L[0]) & (loglha_co <= BOX_L[1])
    o3hb_co = (np.log10(Fo3) - np.log10(Fhb))[obsm & inbox_co]
    zco = z[obsm & inbox_co]
    sp("DESI-COSMOS obs [OIII]/Hb (0.49<z<1, in-box): N=%d" % len(o3hb_co))

    # ---- lightcone mock ----
    tab = pq.read_table(LC + "/desi_cosmos_z1.60_um_sm.parquet",
                        columns=["z_obs", "obs_sm", "obs_sfr", "ssfr"]).to_pandas()
    zl = tab["z_obs"].to_numpy(float); ml = np.log10(np.clip(tab["obs_sm"].to_numpy(float), 1, None))
    sfrl = tab["obs_sfr"].to_numpy(float); lsl = np.where(sfrl > 0, np.log10(sfrl), np.nan)
    lha_lc = lsl + KE + Delta          # observed-scale L_Ha
    sf = (sfrl > 0) & (tab["ssfr"].to_numpy(float) > 1e-11)
    sel = sf & (zl >= 0.49) & (zl < 1.0) & (ml >= BOX_M[0]) & (ml <= BOX_M[1]) & (lha_lc >= BOX_L[0]) & (lha_lc <= BOX_L[1])
    ix = np.where(sel)[0]
    if len(ix) > 40000:
        ix = rng.choice(ix, 40000, replace=False)
    sp("lightcone mock SF in-box (0.49<z<1): %d (using %d)" % (int(sel.sum()), len(ix)))
    r8 = sample_one(flow, meta, ml[ix], lha_lc[ix], seed=55)
    o3hb_lc = r8[:, IOIII] - r8[:, IHB]
    nii_lc = r8[:, INII]
    zl_use = zl[ix]

    # ================= FIGURE =================
    fig = plt.figure(figsize=(15, 4.4))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    # A: Delta calibration
    ax = fig.add_subplot(gs[0, 0])
    xx = lha_ke[cal]; yy = lha_obs[cal]
    ax.hexbin(xx, yy, gridsize=45, cmap=cmr.bubblegum_r, mincnt=1, bins="log")
    lim = [np.percentile(xx, 1), np.percentile(xx, 99)]
    ax.plot(lim, lim, "k-", lw=1, label="1:1")
    ax.plot(lim, np.array(lim) + Delta, "--", color="#d072d6", lw=2, label=r"$+\Delta=%.2f$" % Delta)
    ax.set_xlabel(r"$\log L_{\mathrm{H}\alpha}$ from SFR (Kennicutt)")
    ax.set_ylabel(r"observed $\log L_{\mathrm{H}\alpha}$")
    ax.set_title(r"dust/aperture bridge ($z<0.49$)"); ax.legend(frameon=False, loc="upper left")
    # B: [OIII]/Hb distribution mock vs data
    ax = fig.add_subplot(gs[0, 1])
    lo, hi = np.percentile(np.concatenate([o3hb_co, o3hb_lc]), [1, 99]); bins = np.linspace(lo, hi, 36)
    ax.hist(o3hb_co, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color="#3b6fb0",
            label="DESI-COSMOS (obs)")
    ax.hist(o3hb_lc, bins=bins, density=True, histtype="step", lw=2.4, color="#d072d6",
            label="lightcone $\\times$ NF (mock)")
    ax.set_xlabel(r"$\log_{10}(\mathrm{[OIII]}5007/\mathrm{H}\beta)$"); ax.set_yticks([])
    ax.set_title(r"$0.49<z<1.0$ (in-box)"); ax.legend(frameon=False, fontsize=11)
    # C: mock BPT with demarcations
    ax = fig.add_subplot(gs[0, 2])
    ax.hexbin(nii_lc, o3hb_lc, gridsize=50, cmap=cmr.bubblegum_r, mincnt=1, bins="log")
    xk = np.linspace(-2.0, 0.0, 100)
    ax.plot(xk, 0.61 / (xk - 0.05) + 1.30, "k-", lw=1.5, label="Kauffmann03")
    xw = np.linspace(-2.0, 0.35, 100)
    ax.plot(xw, 0.61 / (xw - 0.47) + 1.19, "k--", lw=1.5, label="Kewley01")
    ax.set_xlim(-1.8, 0.4); ax.set_ylim(-1.2, 1.3)
    ax.set_xlabel(r"$\log_{10}(\mathrm{[NII]}6584/\mathrm{H}\alpha)$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{[OIII]}/\mathrm{H}\beta)$")
    ax.set_title("mock BPT (flow-predicted)"); ax.legend(frameon=False, fontsize=11, loc="lower left")
    fig.suptitle("Lightcone mock-survey: DESI-trained NF applied to a realistic high-z population", fontsize=15, y=1.02)
    out = REPO + "figs_ALTB/hiz_lightcone_mock.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    sp("Saved: " + out)
    # summary stats
    sp("obs  [OIII]/Hb p16/50/84 = %.3f/%.3f/%.3f" % tuple(np.percentile(o3hb_co, [16, 50, 84])))
    sp("mock [OIII]/Hb p16/50/84 = %.3f/%.3f/%.3f" % tuple(np.percentile(o3hb_lc, [16, 50, 84])))
    sp("\n=== DONE ===")


if __name__ == "__main__":
    main()
