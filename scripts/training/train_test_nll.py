"""
A3: overfitting check. For each survey, retrain a flow with the production
hyperparameters on an 80% split and report the NLL (bits/dim, raw log-ratio space)
on the 80% train set vs the 20% held-out set. A negligible gap shows the
low-capacity flow is not overfitting, justifying Table 1 on the full sample.
"""
from pathlib import Path
import pickle
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx, optax
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
FLUX_SCALE=1e-17; SEED=0; EPOCHS=200; BATCH=2048; LR=3e-4; CLIP=1.0
SURVEYS = [("SDSS", BASE+"SDSS_main_training_data.fits", REPO+"nf_sdss_main_meta.pkl", "sdss"),
           ("DESI", BASE+"DESI_BGS_training_data.fits", REPO+"nf_desi_bgs_meta.pkl", "desi")]


def load_scalar_df(p):
    t=Table.read(p,hdu=1); return t[[n for n in t.colnames if len(t[n].shape)<=1]].to_pandas()

def add_loglha(df, survey):
    if survey=="sdss": z=df["Z_1"].to_numpy(float); ha=df["H_ALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    else: z=df["Z"].to_numpy(float); ha=df["HALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    m=np.isfinite(z)&(z>0)&np.isfinite(ha)&(ha>0); out=np.full(len(df),np.nan)
    out[m]=np.log10(ha[m])+np.log10(4*np.pi)+2*np.log10(cosmo.luminosity_distance(z[m]).to("cm").value)
    df=df.copy(); df["LOG_LHA"]=out; return df

def get_XU(df, meta):
    tcols=meta["resolved"]["target_cols"]; raw=[c[6:] if c.startswith("LOG10_") else c for c in tcols]
    ha=next(c for c in ["H_ALPHA_FLUX","HALPHA_FLUX"] if c in df.columns); logm=meta["resolved"]["logmstar_col"]
    F=np.column_stack([df[c].to_numpy(float) for c in raw]); fha=df[ha].to_numpy(float)
    u1=df[logm].to_numpy(float); u2=df["LOG_LHA"].to_numpy(float)
    good=np.all(F>0,axis=1)&(fha>0)&np.isfinite(fha)&np.all(np.isfinite(F),axis=1)&np.isfinite(u1)&np.isfinite(u2)
    return np.log10(F[good])-np.log10(fha[good])[:,None], np.column_stack([u1[good],u2[good]])

def train(key, Xn, Un, xdim):
    flow=block_neural_autoregressive_flow(key=key, base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    opt=optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(LR)); st=opt.init(eqx.filter(flow, eqx.is_inexact_array))
    @eqx.filter_jit
    def loss_fn(fl,x,u): return -jnp.mean(fl.log_prob(x, condition=u))
    @eqx.filter_jit
    def step(fl,s,x,u):
        l,g=eqx.filter_value_and_grad(loss_fn)(fl,x,u)
        up,s=opt.update(eqx.filter(g,eqx.is_inexact_array), s, params=eqx.filter(fl,eqx.is_inexact_array))
        return eqx.apply_updates(fl,up), s, l
    rng=np.random.default_rng(SEED+7); n=Xn.shape[0]
    for ep in range(1,EPOCHS+1):
        order=rng.permutation(n)
        for i in range(0,n,BATCH):
            idx=order[i:i+BATCH]
            flow,st,_=step(flow,st,jnp.asarray(Xn[idx]),jnp.asarray(Un[idx]))
    return flow

def nll_bits(flow, Xn, Un, Xstd, k):
    lp=np.array(jax.vmap(lambda x,u: flow.log_prob(x, condition=u))(jnp.asarray(Xn), jnp.asarray(Un)))
    return float(-np.mean(lp - np.sum(np.log(Xstd)))/(k*np.log(2)))

def main():
    for name, fits, metap, key in SURVEYS:
        meta=pickle.load(open(metap,"rb")); df=add_loglha(load_scalar_df(fits), key)
        X,U=get_XU(df, meta); N=len(X); k=X.shape[1]
        rng=np.random.default_rng(SEED); perm=rng.permutation(N); nte=int(0.2*N); te,tr=perm[:nte],perm[nte:]
        Xm,Xs=X[tr].mean(0),X[tr].std(0); Um,Us=U[tr].mean(0),U[tr].std(0)
        Xn=((X-Xm)/Xs).astype(np.float32); Un=((U-Um)/Us).astype(np.float32)
        flow=train(jr.key(SEED), Xn[tr], Un[tr], k)
        nll_tr=nll_bits(flow, Xn[tr], Un[tr], Xs, k); nll_te=nll_bits(flow, Xn[te], Un[te], Xs, k)
        print(f"{name}: N={N}  NLL_train={nll_tr:.4f}  NLL_test={nll_te:.4f}  gap={nll_te-nll_tr:+.4f} bits/dim", flush=True)

if __name__ == "__main__":
    main()
