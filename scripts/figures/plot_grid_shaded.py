"""
Byler grid coverage shown as a filled reachable REGION (not points), with per-panel
2-D membership so the in/out coloring matches exactly what is shaded. The region is
the occupancy of the dense grid in each plane (fine 2-D histogram, dilated + holes
filled) -> the continuous area the model can produce. Data (and, separately, NF
samples) are colored gray (inside the region) / bubblegum pink (outside). BPT panel
carries the Kauffmann03 / Kewley01 demarcations. Leads with the BPT plane.
"""
import os
os.environ.setdefault("SPS_HOME", "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/FSPS")
import pickle
import numpy as np, pandas as pd
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from scipy.ndimage import binary_dilation, binary_fill_holes
from flowjax.distributions import Normal
from flowjax.flows import block_neural_autoregressive_flow
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa
import cmasher as cmr
import sys
GRIDCSV = sys.argv[1] if len(sys.argv)>1 else "docs/byler_grid_dense.csv"
LABEL = sys.argv[2] if len(sys.argv)>2 else "byler"

BASE="/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"; REPO=BASE+"nebular_emission_model/"
BG=cmr.bubblegum; REGION_C=BG(0.30); OUT_C=BG(0.66); IN_C="0.55"
FLUX_SCALE=1e-17; SEED=0; NSAMP=30000
LIM=dict(nii_ha=(-1.8,0.4), oiii_hb=(-1.5,1.3), sii_ha=(-1.8,0.4))
S=dict(name="SDSS", fits=BASE+"SDSS_main_training_data.fits", eqx=REPO+"nf_sdss_main.eqx", meta=REPO+"nf_sdss_main_meta.pkl", key="sdss")
D=dict(name="DESI", fits=BASE+"DESI_BGS_training_data.fits", eqx=REPO+"nf_desi_bgs.eqx", meta=REPO+"nf_desi_bgs_meta.pkl", key="desi")

def load_scalar_df(p):
    t=Table.read(p,hdu=1); return t[[n for n in t.colnames if len(t[n].shape)<=1]].to_pandas()
def add_loglha(df,key):
    if key=="sdss": z=df["Z_1"].to_numpy(float); ha=df["H_ALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    else: z=df["Z"].to_numpy(float); ha=df["HALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    m=np.isfinite(z)&(z>0)&np.isfinite(ha)&(ha>0); o=np.full(len(df),np.nan)
    o[m]=np.log10(ha[m])+np.log10(4*np.pi)+2*np.log10(cosmo.luminosity_distance(z[m]).to("cm").value)
    df=df.copy(); df["LOG_LHA"]=o; return df
def load_flow(fp,meta):
    xd=len(meta["resolved"]["out_cols"]); return eqx.tree_deserialise_leaves(fp, block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed",0))),base_dist=Normal(jnp.zeros(xd)),cond_dim=2))
def idx_of(cols,*subs): return next(i for i,c in enumerate(cols) if any(s in c for s in subs))

def data_ratios(c):
    meta=pickle.load(open(c["meta"],"rb")); df=add_loglha(load_scalar_df(c["fits"]),c["key"])
    tcols=meta["resolved"]["target_cols"]; raw=[x[6:] if x.startswith("LOG10_") else x for x in tcols]; U=[x.upper() for x in raw]
    def col(sub):
        i=next(k for k,cc in enumerate(U) if any(s in cc for s in ([sub] if isinstance(sub,str) else sub))); return df[raw[i]].to_numpy(float)
    F={n:col(s) for n,s in dict(Hb=["H_BETA","HBETA"],NII="NII_6584",SII16=["SII_6716","SII_6717"],SII31="SII_6731",OIII="OIII_5007").items()}
    F["Ha"]=df[c_h:=next(x for x in ["H_ALPHA_FLUX","HALPHA_FLUX"] if x in df.columns)].to_numpy(float)
    logm=df[meta["resolved"]["logmstar_col"]].to_numpy(float); ll=df["LOG_LHA"].to_numpy(float)
    good=np.all([np.isfinite(v)&(v>0) for v in F.values()],axis=0)&np.isfinite(logm)&np.isfinite(ll); F={k:v[good] for k,v in F.items()}
    R=pd.DataFrame(dict(nii_ha=np.log10(F["NII"]/F["Ha"]),oiii_hb=np.log10(F["OIII"]/F["Hb"]),sii_ha=np.log10((F["SII16"]+F["SII31"])/F["Ha"])))
    return R, np.column_stack([logm[good],ll[good]]), meta
def nf_ratios(flow,meta,Ucond):
    oc=[x.upper() for x in meta["resolved"]["out_cols"]]
    inii=idx_of(oc,"NII_6584");ioiii=idx_of(oc,"OIII_5007");ihb=idx_of(oc,"H_BETA","HBETA");i16=idx_of(oc,"SII_6716","SII_6717");i31=idx_of(oc,"SII_6731")
    rng=np.random.default_rng(SEED); idx=rng.choice(len(Ucond),size=min(NSAMP,len(Ucond)),replace=False)
    Un=((Ucond[idx]-meta["U_mean"])/meta["U_std"]).astype(np.float32); keys=jr.split(jr.key(SEED+3),len(idx))
    X=np.array(jax.vmap(lambda k,u: flow.sample(k,sample_shape=(),condition=u))(keys,jnp.asarray(Un)))*meta["X_std"]+meta["X_mean"]
    return pd.DataFrame(dict(nii_ha=X[:,inii],oiii_hb=X[:,ioiii]-X[:,ihb],sii_ha=np.log10(10**X[:,i16]+10**X[:,i31])))

grid=pd.read_csv(GRIDCSV)
def region(cx,cy,step=0.03,dil=2):
    xe=np.arange(LIM[cx][0],LIM[cx][1]+step,step); ye=np.arange(LIM[cy][0],LIM[cy][1]+step,step)
    H,_,_=np.histogram2d(grid[cx],grid[cy],bins=[xe,ye]); m=binary_fill_holes(binary_dilation(H>0,iterations=dil)); return xe,ye,m
def inside(xe,ye,m,x,y):
    ix=np.clip(np.digitize(x,xe)-1,0,m.shape[0]-1); iy=np.clip(np.digitize(y,ye)-1,0,m.shape[1]-1)
    inb=(x>=xe[0])&(x<=xe[-1])&(y>=ye[0])&(y<=ye[-1]); return m[ix,iy]&inb
def kau(x): return np.where(x<0.05,0.61/(x-0.05)+1.30,np.nan)
def kew(x): return np.where(x<0.47,0.61/(x-0.47)+1.19,np.nan)
def cclip(fn,xlo,xhi,yl): x=np.linspace(xlo,xhi,600); y=fn(x); y[(y<yl[0])|(y>yl[1])]=np.nan; return x,y
planes=[("nii_ha","oiii_hb",r"$\log$([N II]/H$\alpha$)",r"$\log$([O III]/H$\beta$)"),
        ("nii_ha","sii_ha", r"$\log$([N II]/H$\alpha$)",r"$\log$([S II]/H$\alpha$)"),
        ("oiii_hb","sii_ha",r"$\log$([O III]/H$\beta$)",r"$\log$([S II]/H$\alpha$)")]
REG={ (cx,cy): region(cx,cy) for cx,cy,_,_ in planes }
plt.style.use(["science","no-latex"]); plt.rcParams.update({"axes.labelsize":13,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9,"axes.titlesize":12})

def make_fig(R, name, tag):
    fig,axes=plt.subplots(1,3,figsize=(16.5,5.3),constrained_layout=True); outpcts=[]
    rng=np.random.default_rng(1); idx=rng.choice(len(R),size=min(40000,len(R)),replace=False)
    for ax,(cx,cy,lx,ly) in zip(axes,planes):
        xe,ye,m=REG[(cx,cy)]; xc=0.5*(xe[:-1]+xe[1:]); yc=0.5*(ye[:-1]+ye[1:])
        out=~inside(xe,ye,m,R[cx].to_numpy(),R[cy].to_numpy()); outpcts.append(out.mean())
        ax.scatter(R[cx].to_numpy()[idx],R[cy].to_numpy()[idx],s=1.5,alpha=0.4,color="k",rasterized=True,zorder=1)
        ax.contourf(xc,yc,m.T.astype(float),levels=[0.5,1.5],colors=[REGION_C],alpha=0.30,zorder=3)  # region ON TOP so the boundary reads
        ax.contour(xc,yc,m.T.astype(float),levels=[0.5],colors=[BG(0.05)],linewidths=1.4,alpha=0.9,zorder=5)
        if (cx,cy)==("nii_ha","oiii_hb"):
            xk,yk=cclip(kau,-1.9,0.049,LIM["oiii_hb"]); ax.plot(xk,yk,"--",color="#333333",lw=1.3,zorder=6,label="Kauffmann03")
            xv,yv=cclip(kew,-1.9,0.469,LIM["oiii_hb"]); ax.plot(xv,yv,"-.",color="#333333",lw=1.3,zorder=6,label="Kewley01"); ax.legend(loc="lower left")
        ax.set_xlim(*LIM[cx]); ax.set_ylim(*LIM[cy]); ax.set_xlabel(lx); ax.set_ylabel(ly)
    h=[plt.matplotlib.patches.Patch(color=REGION_C,alpha=0.5,label=f"{LABEL} grid (reachable region)"),
       plt.Line2D([],[],marker="o",ls="",color="k",label=f"{name} {tag}")]
    axes[0].legend(handles=h,loc="upper left")
    fig.suptitle(f"{name}: {LABEL} grid reachable region vs {tag}",fontsize=13)
    for e in ("png","pdf"): fig.savefig(REPO+f"figs/{LABEL.lower()}_shaded_{tag.lower().replace(' ','')}_{name.lower()}.{e}",dpi=170,bbox_inches="tight")
    print(f"  {name} {tag}: out-of-region per panel = "+", ".join(f"{p:.0%}" for p in outpcts))

print("=== shaded-region per-panel coverage (dense grid) ===")
for c in (S,D):
    R,Uc,meta=data_ratios(c); make_fig(R,c["name"],"data")
    make_fig(nf_ratios(load_flow(c["eqx"],meta),meta,Uc),c["name"],"NF")
