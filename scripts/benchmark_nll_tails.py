"""
A2: does the flow beat simpler models? On an 80/20 split per survey we compare
three conditional models of x = log10(L_line/L_Ha) given u=(logM*, logL_Ha):
  NF           : the trained conditional flow (this work)
  CondGauss    : OLS mean mu(u) + full residual covariance, Gaussian
  Indep+noise  : OLS mean mu(u) + independent (diagonal) Gaussian noise
Metrics on the held-out 20%:
  (1) NLL in the raw log-ratio space (bits/dim) -- proper density comparison
  (2) physical doublet out-of-bounds fractions ([S II],[O II]) vs the data --
      a tail/non-Gaussian diagnostic the covariance alone cannot capture.
"""
from pathlib import Path
import pickle, csv
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
FLUX_SCALE = 1e-17; SEED = 0; NDRAW = 5
SII_LO, SII_HI = 0.44, 1.45      # 6716/6731
OII_LO, OII_HI = 0.35, 1.47      # 3729/3726
SURVEYS = [("SDSS", BASE+"SDSS_main_training_data.fits", REPO+"nf_sdss_main.eqx", REPO+"nf_sdss_main_meta.pkl", "sdss"),
           ("DESI", BASE+"DESI_BGS_training_data.fits", REPO+"nf_desi_bgs.eqx", REPO+"nf_desi_bgs_meta.pkl", "desi")]


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
    tcols = meta["resolved"]["target_cols"]; raw = [c[6:] if c.startswith("LOG10_") else c for c in tcols]
    ha = next(c for c in ["H_ALPHA_FLUX","HALPHA_FLUX"] if c in df.columns)
    logm = meta["resolved"]["logmstar_col"]
    F = np.column_stack([df[c].to_numpy(float) for c in raw]); fha = df[ha].to_numpy(float)
    u1 = df[logm].to_numpy(float); u2 = df["LOG_LHA"].to_numpy(float)
    good = np.all(F>0,axis=1)&(fha>0)&np.isfinite(fha)&np.all(np.isfinite(F),axis=1)&np.isfinite(u1)&np.isfinite(u2)
    return np.log10(F[good])-np.log10(fha[good])[:,None], np.column_stack([u1[good],u2[good]]), [c.upper() for c in raw]

def idx_of(cols, *subs):
    return next(i for i,c in enumerate(cols) if any(s in c for s in subs))

def doublet_fracs(X, cols):
    i16=idx_of(cols,"SII_6716","SII_6717"); i31=idx_of(cols,"SII_6731"); i26=idx_of(cols,"OII_3726"); i29=idx_of(cols,"OII_3729")
    sii = 10.0**(X[:,i16]-X[:,i31]); oii = 10.0**(X[:,i29]-X[:,i26])
    return (float(np.mean((sii<SII_LO)|(sii>SII_HI))), float(np.mean((oii<OII_LO)|(oii>OII_HI))))

def gauss_nll_bits(Xte, mu, Sig):
    k = Xte.shape[1]; L = np.linalg.cholesky(Sig); logdet = 2*np.sum(np.log(np.diag(L)))
    d = Xte - mu; sol = np.linalg.solve(Sig, d.T).T
    quad = np.sum(d*sol, axis=1)
    logp = -0.5*(k*np.log(2*np.pi) + logdet + quad)
    return float(-np.mean(logp)/(k*np.log(2)))

def main():
    rows=[]; tail_rows=[]
    for name, fits, flowp, metap, key in SURVEYS:
        meta = pickle.load(open(metap,"rb")); flow = load_flow(flowp, meta)
        df = add_loglha(load_scalar_df(fits), key)
        X, U, cols = get_XU(df, meta); N=len(X); k=X.shape[1]
        rng = np.random.default_rng(SEED); perm=rng.permutation(N); nte=int(0.2*N)
        te, tr = perm[:nte], perm[nte:]
        A = lambda u: np.column_stack([np.ones(len(u)), u])
        coef,*_ = np.linalg.lstsq(A(U[tr]), X[tr], rcond=None)
        Etr = X[tr]-A(U[tr])@coef
        Sig = np.cov(Etr, rowvar=False); Dg = np.diag(np.var(Etr,axis=0))
        mu_te = A(U[te])@coef
        # ---- NLL (bits/dim, raw log-ratio space) ----
        nll_g = gauss_nll_bits(X[te], mu_te, Sig)
        nll_i = gauss_nll_bits(X[te], mu_te, Dg)
        Xn = ((X[te]-meta["X_mean"])/meta["X_std"]).astype(np.float32)
        Un = ((U[te]-meta["U_mean"])/meta["U_std"]).astype(np.float32)
        lp_std = np.array(jax.vmap(lambda x,u: flow.log_prob(x, condition=u))(jnp.asarray(Xn), jnp.asarray(Un)))
        lp_raw = lp_std - np.sum(np.log(meta["X_std"]))
        nll_nf = float(-np.mean(lp_raw)/(k*np.log(2)))
        print(f"\n{name} (N_test={nte}) NLL bits/dim:  NF={nll_nf:.3f}  CondGauss={nll_g:.3f}  Indep={nll_i:.3f}")
        rows.append(dict(survey=name, nll_NF=nll_nf, nll_CondGauss=nll_g, nll_Indep=nll_i))
        # ---- doublet out-of-bounds fractions (data vs models) ----
        d_sii,d_oii = doublet_fracs(X[te], cols)
        # NF samples
        keys=jr.split(jr.key(SEED+7), len(te))
        Xs=[]
        for j in range(NDRAW):
            kk=jr.split(jr.key(SEED+7+j), len(te))
            s=jax.vmap(lambda a,u: flow.sample(a, sample_shape=(), condition=u))(kk, jnp.asarray(Un))
            Xs.append(np.array(s)*meta["X_std"]+meta["X_mean"])
        Xs=np.vstack(Xs); nf_sii,nf_oii=doublet_fracs(Xs, cols)
        mu_rep=np.repeat(mu_te,NDRAW,axis=0)
        Gs=mu_rep+rng.multivariate_normal(np.zeros(k),Sig,size=len(mu_rep)); g_sii,g_oii=doublet_fracs(Gs,cols)
        Is=mu_rep+rng.normal(size=mu_rep.shape)*np.sqrt(np.diag(Dg)); i_sii,i_oii=doublet_fracs(Is,cols)
        print(f"  [SII] out-of-bounds: data={d_sii:.3f} NF={nf_sii:.3f} Gauss={g_sii:.3f} Indep={i_sii:.3f}")
        print(f"  [OII] out-of-bounds: data={d_oii:.3f} NF={nf_oii:.3f} Gauss={g_oii:.3f} Indep={i_oii:.3f}")
        for lab,(d,nf,g,ii) in [("SII",(d_sii,nf_sii,g_sii,i_sii)),("OII",(d_oii,nf_oii,g_oii,i_oii))]:
            tail_rows.append(dict(survey=name, doublet=lab, data=d, NF=nf, CondGauss=g, Indep=ii))
    with open(REPO+"docs/benchmark_nll.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    with open(REPO+"docs/benchmark_tails.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(tail_rows[0])); w.writeheader(); [w.writerow(r) for r in tail_rows]
    print("\nWrote docs/benchmark_nll.csv and docs/benchmark_tails.csv")

if __name__ == "__main__":
    main()
