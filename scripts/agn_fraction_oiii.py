"""
A4: (a) quantify the composite/AGN fraction in each selected sample using the
Kauffmann03 and Kewley01 NII-BPT demarcations; (b) test whether the residual
cross-survey [O III] scatter is driven by AGN/composite objects by recomputing
the [O III] ratio-space transfer metrics for the full test sample vs star-forming
only (below Kauffmann03). Uses the existing (recalibrated) flows; n_mc=50.
"""
from pathlib import Path
import pickle
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import spearmanr
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
FLUX_SCALE = 1e-17; SEED = 0; N_MC = 50
S = dict(name="SDSS", fits=BASE+"SDSS_main_training_data.fits", eqx=REPO+"nf_sdss_main.eqx", meta=REPO+"nf_sdss_main_meta.pkl", key="sdss")
D = dict(name="DESI", fits=BASE+"DESI_BGS_training_data.fits", eqx=REPO+"nf_desi_bgs.eqx", meta=REPO+"nf_desi_bgs_meta.pkl", key="desi")


def load_scalar_df(p):
    t = Table.read(p, hdu=1); return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()

def add_loglha(df, survey):
    if survey == "sdss":
        z = df["Z_1"].to_numpy(float); ha = df["H_ALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    else:
        z = df["Z"].to_numpy(float); ha = df["HALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    m = np.isfinite(z)&(z>0)&np.isfinite(ha)&(ha>0)
    out = np.full(len(df), np.nan)
    out[m] = np.log10(ha[m])+np.log10(4*np.pi)+2*np.log10(cosmo.luminosity_distance(z[m]).to("cm").value)
    df = df.copy(); df["LOG_LHA"] = out; return df

def load_flow(flow_path, meta):
    xdim = len(meta["resolved"]["out_cols"])
    tmpl = block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed", 0))), base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    return eqx.tree_deserialise_leaves(flow_path, tmpl)

def get_XU(df, meta):
    tcols = meta["resolved"]["target_cols"]; raw=[c[6:] if c.startswith("LOG10_") else c for c in tcols]
    ha = next(c for c in ["H_ALPHA_FLUX","HALPHA_FLUX"] if c in df.columns)
    logm = meta["resolved"]["logmstar_col"]
    F = np.column_stack([df[c].to_numpy(float) for c in raw]); fha=df[ha].to_numpy(float)
    u1=df[logm].to_numpy(float); u2=df["LOG_LHA"].to_numpy(float)
    good=np.all(F>0,axis=1)&(fha>0)&np.isfinite(fha)&np.all(np.isfinite(F),axis=1)&np.isfinite(u1)&np.isfinite(u2)
    return np.log10(F[good])-np.log10(fha[good])[:,None], np.column_stack([u1[good],u2[good]]), [c.upper() for c in raw]

def idx_of(cols, sub): return next(i for i,c in enumerate(cols) if sub in c)

def bpt_class(X, cols):
    """Return SF / composite / AGN masks from NII-BPT using the true log ratios."""
    nii = X[:, idx_of(cols,"NII_6584")]                      # log([NII]/Ha)
    oiii = X[:, idx_of(cols,"OIII_5007")] - X[:, idx_of(cols,"HBETA" if any("HBETA" in c for c in cols) else "H_BETA")]  # log([OIII]/Hb)
    ka = np.where(nii < 0.05, 0.61/(nii-0.05)+1.30, -np.inf)  # Kauffmann03
    ke = np.where(nii < 0.47, 0.61/(nii-0.47)+1.19, -np.inf)  # Kewley01
    sf = oiii < ka
    agn = (oiii > ke) | (nii >= 0.47)
    comp = (~sf) & (~agn)
    return sf, comp, agn, oiii, nii

def sample_ratios(flow, meta, U, n_mc=N_MC, seed=0):
    Un = ((U-meta["U_mean"])/meta["U_std"]).astype(np.float32); acc=np.zeros((len(U), len(meta["X_std"])))
    for j in range(n_mc):
        kk=jr.split(jr.key(seed+j), len(U))
        s=jax.vmap(lambda a,u: flow.sample(a, sample_shape=(), condition=u))(kk, jnp.asarray(Un))
        acc += np.array(s)
    return (acc/n_mc)*meta["X_std"]+meta["X_mean"]

def metrics(pred, true):
    r=pred-true; p16,p84=np.percentile(r,[16,84])
    return float(0.5*(p84-p16)), float(spearmanr(pred,true).correlation), float(np.median(r))

def main():
    dfs={}; metas={}; flows={}; XUs={}
    for c in (S,D):
        metas[c["name"]]=pickle.load(open(c["meta"],"rb")); flows[c["name"]]=load_flow(c["eqx"],metas[c["name"]])
        dfs[c["name"]]=add_loglha(load_scalar_df(c["fits"]), c["key"])
        XUs[c["name"]]=get_XU(dfs[c["name"]], metas[c["name"]])

    print("=== BPT classification fractions (selected samples) ===")
    for nm in ("SDSS","DESI"):
        X,U,cols=XUs[nm]; sf,comp,agn,_,_=bpt_class(X,cols); n=len(X)
        print(f"  {nm}: SF={sf.mean():.1%}  composite={comp.mean():.1%}  AGN={agn.mean():.1%}  (N={n})")

    print("\n=== [O III] ratio-space transfer: full vs SF-only (below Kauffmann03) ===")
    # SDSS->DESI: apply SDSS flow to DESI conditioning; classify DESI (target)
    for tr, teName in [(S,"DESI"), (D,"SDSS")]:
        Xt,Ut,cols=XUs[teName]
        oiii_i=idx_of(cols,"OIII_5007")
        pred=sample_ratios(flows[tr["name"]], metas[tr["name"]], Ut, seed=SEED+11)
        p=pred[:,oiii_i]; t=Xt[:,oiii_i]
        sf,comp,agn,_,_=bpt_class(Xt,cols)
        sc_all,rho_all,b_all=metrics(p,t)
        sc_sf,rho_sf,b_sf=metrics(p[sf],t[sf])
        print(f"  {tr['name']}->{teName}  [OIII]:  FULL scat={sc_all:.3f} rho={rho_all:.3f} bias={b_all:+.3f} (N={len(t)})")
        print(f"                        SF-only  scat={sc_sf:.3f} rho={rho_sf:.3f} bias={b_sf:+.3f} (N={int(sf.sum())}, {sf.mean():.0%})")

if __name__ == "__main__":
    main()
