#!/usr/bin/env python3
"""
Test the claim: an SFR residual can fix the OVERALL LF normalization, but the line-specific
([OII]) residual survives.

Calibrate a per-z shift a(z) applied to logL_Ha (i.e. a residual on the UM SFR->L_Ha conversion)
that BEST matches the mock LFs to Comparat+16, jointly across [OII]/Hb/[OIII]. Because the shift
acts on L_Ha (shared by all lines), it shifts every line's LF by the same a -> it can only remove
the COMMON offset; the line-relative pattern is invariant. Show what survives.
"""
import pickle, numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.cosmology import FlatwCDM
import astropy.units as u
from functools import partial as _pf
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import scienceplots; plt.style.use(["science", "no-latex"])
except Exception: pass
plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 10, "axes.titlesize": 14})


def sp(m): print(m, flush=True)


BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO = BASE + "nebular_emission_model/"
COSMOS = "/oak/stanford/orgs/kipac/data/cosmos/DESI-COSMOS-v2.0.fits"
LC = "/oak/stanford/orgs/kipac/users/risahw/aemulus_lightcones/results/box017_v1"
FLUX_SCALE = 1e-17; KE = 41.27; MASS_ZP = 0.13; BEAM_SR = 0.004629629629629629
BOX = FlatwCDM(H0=63.437, Om0=0.3194, w0=-0.7069)
CAL = {"HALPHA": 0.0752, "HBETA": 0.0628, "OII_3726": 0.0650, "OII_3729": 0.0075, "OIII_5007": 0.0502}
IHB, IOII_A, IOII_B, IOIII = 0, 5, 6, 7
BOX_L = (39.90, 42.20); MASS_COMPLETE = 9.0; F_CV = 0.15
SHELLS = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
COMP = {"OII": dict(logLs0=41.10, bL=2.33, logPs0=-2.40, bP=-0.73, alpha=-1.46),
        "Hb": dict(logLs0=40.88, bL=2.19, logPs0=-3.34, bP=2.70, alpha=-1.51),
        "OIII": dict(logLs0=41.42, bL=3.91, logPs0=-3.41, bP=-0.76, alpha=-1.83)}
LAB = {"OII": "[OII]3727", "Hb": "Hb", "OIII": "[OIII]5007"}
COL = {(0.3, 0.5): "#440154", (0.5, 0.7): "#21908C", (0.7, 0.9): "#FDE725"}
_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))


def log10_lum(z, f):
    dl = BOX.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def schechter(logL, logLs, logPs, a):
    x = 10 ** (logL - logLs); return np.log(10) * 10 ** logPs * x ** (a + 1) * np.exp(-x)


def shell_vol(z1, z2):
    return (BOX.comoving_volume(z2) - BOX.comoving_volume(z1)).to_value(u.Mpc ** 3) * (BEAM_SR / (4 * np.pi))


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
    if a.dtype.kind == "S": a = np.char.decode(a)
    return np.char.strip(a.astype(str))


def main():
    rng = np.random.default_rng(5)
    # bridge Delta + aperture from DESI-COSMOS
    h = fits.open(COSMOS); d1, d6, d2 = h[1].data, h[6].data, h[2].data
    z = np.asarray(d1["Z"], float); zwarn = np.asarray(d1["ZWARN"], float)
    st = decode(d1["SPECTYPE"]); lpm = np.asarray(d6["lp_mass_med"], float); lpsfr = np.asarray(d6["lp_SFR_med"], float)
    Fha = np.asarray(d1["HALPHA_FLUX"], float) * 10 ** CAL["HALPHA"]
    SNha = np.asarray(d1["HALPHA_FLUX"], float) * np.sqrt(np.clip(np.asarray(d1["HALPHA_FLUX_IVAR"], float), 0, None))
    ff = np.asarray(d2["FIBERFLUX_R"], float); tf = np.asarray(d2["FLUX_R"], float)
    fap = np.clip(np.where((tf > 0) & (ff > 0), ff / tf, np.nan), 1e-3, 1.2)
    base = (zwarn == 0) & (st == "GALAXY") & (z > 0.05) & np.isfinite(lpm) & (lpm > 6) & (lpm < 13)
    lpl = lpsfr if np.nanmedian(np.abs(lpsfr[np.isfinite(lpsfr) & base])) < 5 else np.log10(np.clip(lpsfr, 1e-4, None))
    calm = base & (z < 0.49) & (SNha > 3) & (Fha > 0) & np.isfinite(lpl) & (lpl > -3) & (lpl < 3)
    Delta = float(np.median((log10_lum(z, Fha) - (lpl + KE))[calm])); ap = -np.log10(np.nanmedian(fap[calm]))

    # mock
    tab = pq.read_table(LC + "/desi_cosmos_z1.60_um_sm.parquet", columns=["z_obs", "obs_sm", "obs_sfr", "ssfr"]).to_pandas()
    zl = tab["z_obs"].to_numpy(float); ml = np.log10(np.clip(tab["obs_sm"].to_numpy(float), 1, None))
    sfrl = tab["obs_sfr"].to_numpy(float); lsl = np.where(sfrl > 0, np.log10(sfrl), np.nan)
    sf = (sfrl > 0) & (tab["ssfr"].to_numpy(float) > 1e-11) & (zl >= 0.30) & (zl < 0.90) & (ml > MASS_COMPLETE) & np.isfinite(ml)
    ix = np.where(sf)[0]; zlc = zl[ix]; mlc = ml[ix]
    lha_fiber = lsl[ix] + KE + Delta          # NF conditioning: fiber/training scale (as in v2)
    lha_total = lha_fiber + ap                # total observed, for the luminosity (Comparat is total)
    flow, meta = load_flow(); r8 = sample_one(flow, meta, mlc, lha_fiber, seed=31)
    L = {"OII": lha_total + np.log10(10 ** r8[:, IOII_A] + 10 ** r8[:, IOII_B]), "Hb": lha_total + r8[:, IHB], "OIII": lha_total + r8[:, IOIII]}

    Lbins = np.arange(39.6, 43.4, 0.1); Lc = 0.5 * (Lbins[:-1] + Lbins[1:]); dlog = 0.1
    # per-shell: original LF, completeness, best common shift a(z)
    sp("\nz-shell   a(z)_bestfit   per-line residual [dex, logPhi mock-Comp]  before -> after(+a)")
    results = {}
    for (z1, z2) in SHELLS:
        s = (zlc >= z1) & (zlc < z2); V = shell_vol(z1, z2); zc = 0.5 * (z1 + z2)
        phi = {}; comp = {}; Lcomp = {}; errs = {}
        for ln in ["OII", "Hb", "OIII"]:
            N, _ = np.histogram(L[ln][s], bins=Lbins); ph = N / (V * dlog); phi[ln] = ph
            errs[ln] = ph * np.sqrt(1.0 / np.maximum(N, 1) + F_CV ** 2)   # Poisson + cosmic variance
            ipk = np.argmax(np.where(N >= 20, ph, 0)); Lcomp[ln] = Lc[ipk]
            c = COMP[ln]; Ls = c["logLs0"] + c["bL"] * np.log10(1 + zc); Ps = c["logPs0"] + c["bP"] * np.log10(1 + zc)
            comp[ln] = (Ls, Ps, c["alpha"])
        # calibrate single a to minimize total (logPhi_mock(L-a) - logPhi_Comp)^2 over complete bins, all lines
        agrid = np.arange(-0.6, 0.31, 0.02); best = (1e9, 0.0)
        for a in agrid:
            tot = 0.0
            for ln in ["OII", "Hb", "OIII"]:
                Ls, Ps, al = comp[ln]
                m = (Lc > Lcomp[ln] + 0.05) & (Lc < 42.6)
                mock_sh = np.interp(Lc[m] - a, Lc, phi[ln], left=0, right=0)
                good = mock_sh > 0
                cmp = schechter(Lc[m][good], Ls, Ps, al)
                tot += np.sum((np.log10(mock_sh[good]) - np.log10(cmp)) ** 2)
            if tot < best[0]: best = (tot, a)
        a = best[1]; results[(z1, z2)] = (phi, comp, Lcomp, a, errs)
        # residuals before (a=0) and after (a) per line, median over complete bins
        line = []
        for ln in ["OII", "Hb", "OIII"]:
            Ls, Ps, al = comp[ln]; m = (Lc > Lcomp[ln] + 0.05) & (Lc < 42.4)
            def resid(shift):
                mock_sh = np.interp(Lc[m] - shift, Lc, phi[ln], left=0, right=0); g = mock_sh > 0
                return np.median(np.log10(mock_sh[g]) - np.log10(schechter(Lc[m][g], Ls, Ps, al)))
            line.append((ln, resid(0.0), resid(a)))
        sp("[%.1f,%.1f]  a=%+.2f    " % (z1, z2, a) + "  ".join("%s:%+.2f->%+.2f" % (ln, b, af) for ln, b, af in line))

    # figure: 2 rows (before / after SFR residual) x 3 lines; mock = points, Comparat = solid line, colour = z-shell
    from matplotlib.lines import Line2D
    LINES = ["OII", "Hb", "OIII"]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4), sharex=True, sharey=True)
    for col, ln in enumerate(LINES):
        for row, mode in enumerate(["before", "after"]):
            ax = axes[row, col]
            for (z1, z2) in SHELLS:
                phi, comp, Lcomp, a, errs = results[(z1, z2)]; colr = COL[(z1, z2)]
                shift = 0.0 if mode == "before" else a
                m = Lc > Lcomp[ln] + 0.05
                ax.errorbar(Lc[m] + shift, phi[ln][m], yerr=errs[ln][m], fmt="o", ms=4.5,
                            color=colr, capsize=2, elinewidth=1)                            # mock
                Ls, Ps, al = comp[ln]; xx = np.linspace(Lcomp[ln], 43.0, 60)
                ax.plot(xx, schechter(xx, Ls, Ps, al), "-", color=colr, lw=2.0)             # Comparat data
            ax.set_yscale("log"); ax.set_xlim(40.2, 43.0); ax.set_ylim(3e-6, 3e-2)
            if row == 1:
                ax.set_xlabel(r"$\log_{10} L$ [erg s$^{-1}$]")
            if col == 0:
                ax.set_ylabel(r"$\Phi$ [Mpc$^{-3}$ dex$^{-1}$]")
            if row == 0:
                ax.set_title(LAB[ln], fontsize=16)
    zh = [Line2D([], [], color=COL[s], lw=2.4, marker="o", ms=7, label=r"$%.1f<z<%.1f$" % s) for s in SHELLS]
    kh = [Line2D([], [], color="0.3", lw=0, marker="o", ms=8, label=r"mock (lightcone$\times$NF)"),
          Line2D([], [], color="0.3", lw=2.6, label="data (Comparat+16)")]
    axes[0, 0].legend(handles=zh, frameon=False, fontsize=14, loc="upper right", title="redshift", title_fontsize=14)
    axes[1, 0].legend(handles=kh, frameon=False, fontsize=14, loc="upper right")
    axes[1, 0].text(0.05, 0.12, "with SFR residual", transform=axes[1, 0].transAxes,
                    fontsize=14, fontweight="bold", va="bottom")
    fig.suptitle(r"Lightcone$\times$NF emission-line LFs vs Comparat+16: before vs after an SFR residual", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = REPO + "figs_ALTB/hiz_lc_lf_sfrresidual.png"; fig.savefig(out, bbox_inches="tight", dpi=150); sp("Saved: " + out)
    sp("=== DONE ===")


if __name__ == "__main__":
    main()
