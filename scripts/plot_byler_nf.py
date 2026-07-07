"""
Overlay normalizing-flow samples on the Byler grid in the dust-insensitive 3-D
space. Question: does the flow GENERATE the regions the grid cannot reach (the
AGN/composite wing) that real galaxies occupy? Reports the out-of-hull fraction
for the data and for NF samples (should match), and shows NF samples colored by
in/out of the grid's 3-D convex hull.
"""
import os
os.environ.setdefault("SPS_HOME", "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/FSPS")
import pickle
import numpy as np, pandas as pd
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.spatial import Delaunay, ConvexHull
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
FLUX_SCALE=1e-17; SEED=0; NSAMP=30000
S = dict(name="SDSS", fits=BASE+"SDSS_main_training_data.fits", eqx=REPO+"nf_sdss_main.eqx", meta=REPO+"nf_sdss_main_meta.pkl")
D = dict(name="DESI", fits=BASE+"DESI_BGS_training_data.fits", eqx=REPO+"nf_desi_bgs.eqx", meta=REPO+"nf_desi_bgs_meta.pkl")

def load_scalar_df(p):
    t=Table.read(p,hdu=1); return t[[n for n in t.colnames if len(t[n].shape)<=1]].to_pandas()
def add_loglha(df, survey):
    if survey=="sdss": z=df["Z_1"].to_numpy(float); ha=df["H_ALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    else: z=df["Z"].to_numpy(float); ha=df["HALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    m=np.isfinite(z)&(z>0)&np.isfinite(ha)&(ha>0); out=np.full(len(df),np.nan)
    out[m]=np.log10(ha[m])+np.log10(4*np.pi)+2*np.log10(cosmo.luminosity_distance(z[m]).to("cm").value)
    df=df.copy(); df["LOG_LHA"]=out; return df
def load_flow(fp, meta):
    xd=len(meta["resolved"]["out_cols"])
    return eqx.tree_deserialise_leaves(fp, block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed",0))), base_dist=Normal(jnp.zeros(xd)), cond_dim=2))
def idx_of(cols,*subs): return next(i for i,c in enumerate(cols) if any(s in c for s in subs))

def data_ratios(fits, metap, key):
    meta=pickle.load(open(metap,"rb")); df=add_loglha(load_scalar_df(fits), key)
    tcols=meta["resolved"]["target_cols"]; raw=[c[6:] if c.startswith("LOG10_") else c for c in tcols]
    hacol=next(c for c in ["H_ALPHA_FLUX","HALPHA_FLUX"] if c in df.columns); U=[c.upper() for c in raw]
    def col(sub):
        i=next(k for k,c in enumerate(U) if any(s in c for s in ([sub] if isinstance(sub,str) else sub))); return df[raw[i]].to_numpy(float)
    F={n:col(s) for n,s in dict(Hb=["H_BETA","HBETA"],NII="NII_6584",SII16=["SII_6716","SII_6717"],SII31="SII_6731",OIII="OIII_5007").items()}
    F["Ha"]=df[hacol].to_numpy(float); logm=df[meta["resolved"]["logmstar_col"]].to_numpy(float); ll=df["LOG_LHA"].to_numpy(float)
    good=np.all([np.isfinite(v)&(v>0) for v in F.values()],axis=0)&np.isfinite(logm)&np.isfinite(ll)
    F={k:v[good] for k,v in F.items()}
    R=pd.DataFrame(dict(nii_ha=np.log10(F["NII"]/F["Ha"]),oiii_hb=np.log10(F["OIII"]/F["Hb"]),sii_ha=np.log10((F["SII16"]+F["SII31"])/F["Ha"])))
    return R, np.column_stack([logm[good], ll[good]]), meta

def nf_ratios(flow, meta, Ucond):
    oc=[c.upper() for c in meta["resolved"]["out_cols"]]
    inii=idx_of(oc,"NII_6584"); ioiii=idx_of(oc,"OIII_5007"); ihb=idx_of(oc,"H_BETA","HBETA")
    is16=idx_of(oc,"SII_6716","SII_6717"); is31=idx_of(oc,"SII_6731")
    rng=np.random.default_rng(SEED); idx=rng.choice(len(Ucond), size=min(NSAMP,len(Ucond)), replace=False)
    Uc=Ucond[idx]; Un=((Uc-meta["U_mean"])/meta["U_std"]).astype(np.float32)
    keys=jr.split(jr.key(SEED+3), len(Uc))
    Xn=np.array(jax.vmap(lambda k,u: flow.sample(k, sample_shape=(), condition=u))(keys, jnp.asarray(Un)))
    X=Xn*meta["X_std"]+meta["X_mean"]
    return pd.DataFrame(dict(nii_ha=X[:,inii], oiii_hb=X[:,ioiii]-X[:,ihb],
                             sii_ha=np.log10(10**X[:,is16]+10**X[:,is31])))

grid=pd.read_csv(REPO+"docs/byler_grid.csv"); cols3=["nii_ha","oiii_hb","sii_ha"]
hull3=Delaunay(grid[cols3].to_numpy())
def kau(x): return np.where(x<0.05,0.61/(x-0.05)+1.30,-np.inf)
def kew(x): return np.where(x<0.47,0.61/(x-0.47)+1.19,-np.inf)
planes=[("nii_ha","oiii_hb",r"$\log$([N II]/H$\alpha$)",r"$\log$([O III]/H$\beta$)"),
        ("nii_ha","sii_ha", r"$\log$([N II]/H$\alpha$)",r"$\log$([S II]/H$\alpha$)"),
        ("oiii_hb","sii_ha",r"$\log$([O III]/H$\beta$)",r"$\log$([S II]/H$\alpha$)")]
plt.style.use(["science","no-latex"]); plt.rcParams.update({"axes.labelsize":13,"legend.fontsize":9,"axes.titlesize":11})

print("=== 3-D dust-insensitive out-of-hull: data vs NF samples ===")
for c in (S,D):
    Rdat, Ucond, meta = data_ratios(c["fits"], c["meta"], c["name"].lower())
    flow=load_flow(c["eqx"], meta)
    Rnf = nf_ratios(flow, meta, Ucond)
    d_out=(hull3.find_simplex(Rdat[cols3].to_numpy())>=0)==False
    n_out=(hull3.find_simplex(Rnf[cols3].to_numpy())>=0)==False
    print(f"  {c['name']}: data out={d_out.mean():.1%} (N={len(Rdat)}) ;  NF out={n_out.mean():.1%} (N={len(Rnf)})")
    fig,axes=plt.subplots(1,3,figsize=(16.5,5.3),constrained_layout=True)
    for ax,(cx,cy,lx,ly) in zip(axes,planes):
        gx,gy=grid[cx].to_numpy(),grid[cy].to_numpy(); ch=ConvexHull(np.column_stack([gx,gy]))
        for s in ch.simplices: ax.plot(np.column_stack([gx,gy])[s,0],np.column_stack([gx,gy])[s,1],color="0.5",lw=0.9)
        ax.scatter(Rnf[cx].to_numpy()[~n_out],Rnf[cy].to_numpy()[~n_out],s=3,alpha=0.12,color="0.6",rasterized=True)
        ax.scatter(Rnf[cx].to_numpy()[n_out],Rnf[cy].to_numpy()[n_out],s=4,alpha=0.30,color="#009E73",rasterized=True)
        ax.scatter(gx,gy,s=7,color="k",alpha=0.55,zorder=5)
        if (cx,cy)==("nii_ha","oiii_hb"):
            xs=np.linspace(-1.8,0.0,200); ax.plot(xs,kau(xs),"b--",lw=1.4)
            xk=np.linspace(-1.8,0.4,200); ax.plot(xk,kew(xk),"m-.",lw=1.4)
        ax.set_xlabel(lx); ax.set_ylabel(ly)
    axes[0].scatter([],[],color="0.6",label="NF sample, in hull"); axes[0].scatter([],[],color="#009E73",label=f"NF sample, outside ({n_out.mean():.0%})")
    axes[0].scatter([],[],color="k",label="Byler grid"); axes[0].legend(loc="lower left")
    fig.suptitle(f"{c['name']}: NF samples fill the grid gap  (NF out {n_out.mean():.0%}, data out {d_out.mean():.0%})",fontsize=13)
    for e in ("png","pdf"): fig.savefig(REPO+f"figs/byler_nf_{c['name'].lower()}.{e}",dpi=160,bbox_inches="tight")
    print(f"  Wrote figs/byler_nf_{c['name'].lower()}.png")
