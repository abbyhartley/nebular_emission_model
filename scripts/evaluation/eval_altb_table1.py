# eval_altb_table1.py
# Recompute Table 1 (+ 4.1 summary stats) for the ALT-B flows, matching the paper's defs:
#   - NLL (bits/dim): true 8-D log flux-ratio vector under the flow, evaluated on the
#       COMMON (unstandardized) scale so flows are comparable  [log_prob needs no inverter].
#   - RMSE / bias(median) / scatter(0.5[p84-p16]) / NMAD(1.4826 MAD): pooled over 8 lines
#       x galaxies, on log(L_line/L_Ha) residuals, MC-mean prediction (n_mc=50).
#   - Spearman rho: ratio-space, mean of per-line rho (avoids the between-line triviality).
# Plus Balmer decrement p16/p50/p84 & frac<2.86, and [SII]/[OII] out-of-band doublet fractions.
import pickle
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
from functools import partial as _pf
from flowjax.root_finding import root_finder_to_inverter as _r2i, bisect_check_expand_search as _bces
_ROBUST_INV = _r2i(_pf(_bces, midpoint=jnp.zeros(8), width=5, max_steps=1000, throw=False, max_width=200))

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
FLUX_SCALE = 1e-17
N_EVAL = 100_000      # subsample per test set (use all if fewer)
N_MC = 50
LN2 = np.log(2.0)
FLOOR = 2.86
SII_LO, SII_HI = 0.44, 1.45     # 6716/6731 band
OII_LO, OII_HI = 0.35, 1.47     # 3729/3726 band

# physical line order of the 8 flow targets (matches out_cols); aliases resolve per survey
LINE_ALIASES = [["HBETA_FLUX", "H_BETA_FLUX"], ["HGAMMA_FLUX", "H_GAMMA_FLUX"], ["NII_6584_FLUX"],
                ["SII_6716_FLUX", "SII_6717_FLUX"], ["SII_6731_FLUX"], ["OII_3726_FLUX"],
                ["OII_3729_FLUX"], ["OIII_5007_FLUX"]]
LINE_NAME = ["Hbeta", "Hgamma", "NII6584", "SIIa", "SIIb", "OIIa", "OIIb", "OIII5007"]

FLOWS = {
    "SDSS": (REPO + "models/nf_sdss_ALTB.eqx", REPO + "models/nf_sdss_ALTB_meta.pkl"),
    "DESI": (REPO + "models/nf_desi_ALTB.eqx", REPO + "models/nf_desi_ALTB_meta.pkl"),
}
FITS = {"SDSS": BASE + "SDSS_main_training_data_ALTB.fits", "DESI": BASE + "DESI_BGS_training_data_ALTB.fits"}


def load_flow(fp, meta):
    xdim = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))),
                                            base_dist=Normal(jnp.zeros(xdim)), cond_dim=2, inverter=_ROBUST_INV)
    return eqx.tree_deserialise_leaves(fp, tmpl)


def log10_lum(z, f1e17):
    dl = cosmo.luminosity_distance(np.asarray(z, float)).to("cm").value
    return np.log10(np.asarray(f1e17, float) * FLUX_SCALE) + np.log10(4 * np.pi) + 2 * np.log10(dl)


def resolve(df, aliases):
    for c in aliases:
        if c in df.columns:
            return c
    raise KeyError(aliases)


def load_survey(tag, seed):
    t = Table.read(FITS[tag], hdu=1)
    df = t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()
    z_col = "Z_1" if tag == "SDSS" else "Z"
    ha_col = "H_ALPHA_FLUX" if tag == "SDSS" else "HALPHA_FLUX"
    z = df[z_col].to_numpy(float); ha = df[ha_col].to_numpy(float)
    logm = df["LOGM_COLOR"].to_numpy(float); loglha = log10_lum(z, ha)
    cols = [resolve(df, a) for a in LINE_ALIASES]
    F = np.column_stack([df[c].to_numpy(float) for c in cols])
    m = (np.isfinite(z) & (z > 0) & np.isfinite(logm) & np.isfinite(loglha) & (ha > 0)
         & np.all(F > 0, axis=1) & np.all(np.isfinite(F), axis=1))
    F, ha, logm, loglha = F[m], ha[m], logm[m], loglha[m]
    true_ratios = np.log10(F) - np.log10(ha)[:, None]        # (n,8) log10(F_line/F_Ha)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ha)) if len(ha) <= N_EVAL else rng.choice(len(ha), N_EVAL, replace=False)
    d = dict(logm=logm[idx], loglha=loglha[idx], ratios=true_ratios[idx], N=len(ha))
    # doublet indices within the 8: SIIa=3, SIIb=4, OIIa=5(3726), OIIb=6(3729)
    d["sii_obs"] = 10.0 ** (true_ratios[idx, 3] - true_ratios[idx, 4])   # 6716/6731
    d["oii_obs"] = 10.0 ** (true_ratios[idx, 6] - true_ratios[idx, 5])   # 3729/3726
    d["R_obs"] = 10.0 ** (-true_ratios[idx, 0])                          # Ha/Hb = 1/(Hb/Ha)
    return d


def mc_mean_ratios(flow, meta, logm, loglha, seed, n_mc=N_MC, batch=50_000):
    U = np.column_stack([logm, loglha]).astype(np.float32)
    Un_all = jnp.asarray((U - meta["U_mean"]) / meta["U_std"])
    Xm, Xs = meta["X_mean"], meta["X_std"]; n = len(U)
    out = np.zeros((n, len(meta["resolved"]["out_cols"])))
    key = jr.key(seed + 7)
    for lo in range(0, n, batch):
        hi = min(n, lo + batch); Un = Un_all[lo:hi]; acc = np.zeros((hi - lo, out.shape[1]))
        for _ in range(n_mc):
            key, sk = jr.split(key); keys = jr.split(sk, hi - lo)
            Xn = np.array(jax.vmap(lambda k, u: flow.sample(k, sample_shape=(), condition=u))(keys, Un))
            acc += Xn * Xs + Xm
        out[lo:hi] = acc / n_mc
    return out


def nll_bits_dim(flow, meta, logm, loglha, true_ratios):
    Xs = np.asarray(meta["X_std"], float)
    x_std = jnp.asarray((true_ratios - meta["X_mean"]) / Xs)
    U = np.column_stack([logm, loglha]).astype(np.float32)
    Un = jnp.asarray((U - meta["U_mean"]) / meta["U_std"])
    lp_std = np.array(flow.log_prob(x_std, condition=Un))          # natural log, standardized space
    lp_raw = lp_std - np.sum(np.log(Xs))                            # common (unstandardized) scale
    return float(-np.mean(lp_raw) / (LN2 * len(Xs)))               # bits per dim


def pctile_report(name, R):
    R = R[np.isfinite(R) & (R > 0)]
    p16, p50, p84 = np.percentile(R, [16, 50, 84])
    print(f"  {name:26s} p16/p50/p84 = {p16:6.3f}/{p50:6.3f}/{p84:6.3f}   frac<{FLOOR} = {np.mean(R < FLOOR):6.2%}", flush=True)


def doublet_report(name, R, lo, hi):
    R = R[np.isfinite(R) & (R > 0)]
    print(f"  {name:26s} p50={np.median(R):.3f}  frac<{lo}={np.mean(R < lo):6.2%}  "
          f"frac>{hi}={np.mean(R > hi):6.2%}  OUT={np.mean((R < lo) | (R > hi)):6.2%}", flush=True)


def main():
    metas = {t: pickle.load(open(FLOWS[t][1], "rb")) for t in FLOWS}
    flows = {t: load_flow(FLOWS[t][0], metas[t]) for t in FLOWS}
    data = {"SDSS": load_survey("SDSS", 1), "DESI": load_survey("DESI", 2)}
    print(f"N (full ALT-B): SDSS={data['SDSS']['N']:,}  DESI={data['DESI']['N']:,}   (eval on <= {N_EVAL:,})\n", flush=True)

    combos = [("SDSS", "SDSS"), ("SDSS", "DESI"), ("DESI", "DESI"), ("DESI", "SDSS")]
    print("========== TABLE 1 ==========", flush=True)
    print(f"{'train->test':14s} {'NLL':>8s} {'RMSE':>7s} {'bias':>8s} {'scatter':>8s} {'NMAD':>7s} {'rho_ratio':>9s}", flush=True)
    nf_ratio_cache = {}      # MC-mean (n_mc=50): for Table 1 point metrics
    nf_single_cache = {}     # single draw (n_mc=1): for distributional diagnostics (Balmer, doublets)
    for ci, (tr, te) in enumerate(combos):
        fl, mt, D = flows[tr], metas[tr], data[te]
        nll = nll_bits_dim(fl, mt, D["logm"], D["loglha"], D["ratios"])
        pred = mc_mean_ratios(fl, mt, D["logm"], D["loglha"], seed=100 + ci)
        nf_single_cache[(tr, te)] = mc_mean_ratios(fl, mt, D["logm"], D["loglha"], seed=500 + ci, n_mc=1)
        res = (pred - D["ratios"]).ravel()
        rmse = float(np.sqrt(np.mean(res**2))); bias = float(np.median(res))
        p16, p84 = np.percentile(res, [16, 84]); scat = float(0.5 * (p84 - p16))
        nmad = float(1.4826 * np.median(np.abs(res - np.median(res))))
        rhos = [spearmanr(D["ratios"][:, j], pred[:, j]).correlation for j in range(8)]
        rho_mean = float(np.mean(rhos))
        rho_pool = float(spearmanr(D["ratios"].ravel(), pred.ravel()).correlation)
        print(f"{tr+'->'+te:14s} NLL={nll:7.3f} RMSE={rmse:.3f} bias={bias:+.3f} scat={scat:.3f} "
              f"NMAD={nmad:.3f} rho_pooled={rho_pool:.3f} rho_perlineMean={rho_mean:.3f}", flush=True)
        print(f"    per-line rho: " + "  ".join(f"{LINE_NAME[j]}={rhos[j]:.2f}" for j in range(8)), flush=True)
        nf_ratio_cache[(tr, te)] = pred

    print("\n========== BALMER DECREMENT R = Ha/Hb ==========", flush=True)
    for te in ["SDSS", "DESI"]:
        pctile_report(f"Observed {te}", data[te]["R_obs"])
    for tr, te, lab in [("SDSS", "SDSS", "SDSS->SDSS (in)"), ("DESI", "DESI", "DESI->DESI (in)"),
                        ("SDSS", "DESI", "SDSS->DESI (cross)"), ("DESI", "SDSS", "DESI->SDSS (cross)")]:
        R = 10.0 ** (-nf_single_cache[(tr, te)][:, 0])       # single draw preserves the scatter
        pctile_report(f"NF {lab}", R)

    print("\n========== DOUBLET OUT-OF-BAND FRACTIONS ==========", flush=True)
    print(" [S II] 6716/6731  band [0.44,1.45]:", flush=True)
    for te in ["SDSS", "DESI"]:
        doublet_report(f"Observed {te}", data[te]["sii_obs"], SII_LO, SII_HI)
    for tr, te, lab in [("SDSS", "SDSS", "SDSS->SDSS"), ("DESI", "DESI", "DESI->DESI"),
                        ("SDSS", "DESI", "SDSS->DESI"), ("DESI", "SDSS", "DESI->SDSS")]:
        r = nf_single_cache[(tr, te)]
        doublet_report(f"NF {lab}", 10.0 ** (r[:, 3] - r[:, 4]), SII_LO, SII_HI)
    print(" [O II] 3729/3726  band [0.35,1.47]:", flush=True)
    for te in ["SDSS", "DESI"]:
        doublet_report(f"Observed {te}", data[te]["oii_obs"], OII_LO, OII_HI)
    for tr, te, lab in [("SDSS", "SDSS", "SDSS->SDSS"), ("DESI", "DESI", "DESI->DESI"),
                        ("SDSS", "DESI", "SDSS->DESI"), ("DESI", "SDSS", "DESI->SDSS")]:
        r = nf_single_cache[(tr, te)]
        doublet_report(f"NF {lab}", 10.0 ** (r[:, 6] - r[:, 5]), OII_LO, OII_HI)


if __name__ == "__main__":
    main()
