"""
Dust-insensitive 3-D coverage of the data by the Byler+2017 grid, in
([N II]/Ha, [O III]/Hb, [S II]/Ha). In/out membership is computed in the FULL 3-D
space (so a point can lie within a 2-D projection of the grid yet be outside the
3-D envelope). No 2-D hull outline is drawn, to avoid implying a 2-D boundary.
"""
import os
os.environ.setdefault("SPS_HOME", "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/FSPS")
import pickle
import numpy as np, pandas as pd
from astropy.table import Table
from scipy.spatial import Delaunay
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa
import cmasher as cmr

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
BG = cmr.bubblegum
GRID_C, OUT_C, IN_C = BG(0.10), BG(0.66), "0.78"
S = dict(name="SDSS", fits=BASE+"SDSS_main_training_data.fits", meta=REPO+"nf_sdss_main_meta.pkl")
D = dict(name="DESI", fits=BASE+"DESI_BGS_training_data.fits", meta=REPO+"nf_desi_bgs_meta.pkl")
LIM = dict(nii_ha=(-1.8, 0.4), oiii_hb=(-1.5, 1.3), sii_ha=(-1.8, 0.4))

def load_scalar_df(p):
    t=Table.read(p,hdu=1); return t[[n for n in t.colnames if len(t[n].shape)<=1]].to_pandas()

def get_ratios(fits, metap):
    meta=pickle.load(open(metap,"rb")); df=load_scalar_df(fits)
    tcols=meta["resolved"]["target_cols"]; raw=[c[6:] if c.startswith("LOG10_") else c for c in tcols]
    hacol=next(c for c in ["H_ALPHA_FLUX","HALPHA_FLUX"] if c in df.columns); U=[c.upper() for c in raw]
    def col(sub):
        i=next(k for k,c in enumerate(U) if any(s in c for s in ([sub] if isinstance(sub,str) else sub)))
        return df[raw[i]].to_numpy(float)
    F={n: col(s) for n,s in dict(Hb=["H_BETA","HBETA"], NII="NII_6584",
        SII16=["SII_6716","SII_6717"], SII31="SII_6731", OIII="OIII_5007").items()}
    F["Ha"]=df[hacol].to_numpy(float)
    good=np.all([np.isfinite(v)&(v>0) for v in F.values()],axis=0)
    F={k:v[good] for k,v in F.items()}
    return pd.DataFrame(dict(nii_ha=np.log10(F["NII"]/F["Ha"]),
        oiii_hb=np.log10(F["OIII"]/F["Hb"]), sii_ha=np.log10((F["SII16"]+F["SII31"])/F["Ha"])))

grid=pd.read_csv(REPO+"docs/byler_grid.csv"); cols3=["nii_ha","oiii_hb","sii_ha"]
hull3=Delaunay(grid[cols3].to_numpy())
dat={c["name"]: get_ratios(c["fits"], c["meta"]) for c in (S,D)}

def kau(x): return np.where(x<0.05,0.61/(x-0.05)+1.30,np.nan)
def kew(x): return np.where(x<0.47,0.61/(x-0.47)+1.19,np.nan)
def curve(fn, xlo, xhi, ylim):
    x=np.linspace(xlo,xhi,600); y=fn(x); y[(y<ylim[0])|(y>ylim[1])]=np.nan; return x,y

planes=[("nii_ha","oiii_hb",r"$\log$([N II]/H$\alpha$)",r"$\log$([O III]/H$\beta$)"),
        ("nii_ha","sii_ha", r"$\log$([N II]/H$\alpha$)",r"$\log$([S II]/H$\alpha$)"),
        ("oiii_hb","sii_ha",r"$\log$([O III]/H$\beta$)",r"$\log$([S II]/H$\alpha$)")]
plt.style.use(["science","no-latex"])
plt.rcParams.update({"axes.labelsize":13,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9,"axes.titlesize":12})
rng=np.random.default_rng(0)
print("=== 3-D dust-insensitive coverage ===")
for nm in ("SDSS","DESI"):
    d=dat[nm]; out=hull3.find_simplex(d[cols3].to_numpy())<0
    nii_=d["nii_ha"].to_numpy(); agn=(nii_>=0.05)|(d["oiii_hb"].to_numpy()>kau(nii_))  # >=0.05 is right of the Kauffmann asymptote
    print(f"  {nm}: out={out.mean():.1%}; of out, {(agn&out).sum()/max(out.sum(),1):.0%} composite/AGN; SF-only out={(out&~agn).sum()/max((~agn).sum(),1):.1%}")
    fig,axes=plt.subplots(1,3,figsize=(16.5,5.3),constrained_layout=True)
    idx=rng.choice(len(d),size=min(30000,len(d)),replace=False)
    ii=idx[~out[idx]]; oo=idx[out[idx]]
    for ax,(cx,cy,lx,ly) in zip(axes,planes):
        ax.scatter(grid[cx],grid[cy],s=9,color=GRID_C,alpha=0.7,edgecolors="none",zorder=1,label="Byler grid")
        ax.scatter(d[cx].to_numpy()[ii],d[cy].to_numpy()[ii],s=2,alpha=0.06,color=IN_C,rasterized=True,zorder=2)
        ax.scatter(d[cx].to_numpy()[oo],d[cy].to_numpy()[oo],s=4,alpha=0.30,color=OUT_C,rasterized=True,zorder=3)
        if (cx,cy)==("nii_ha","oiii_hb"):
            xk,yk=curve(kau,-1.9,0.049,LIM["oiii_hb"]); ax.plot(xk,yk,"--",color="k",lw=1.4,label="Kauffmann03")
            xe,ye=curve(kew,-1.9,0.469,LIM["oiii_hb"]); ax.plot(xe,ye,"-.",color="k",lw=1.4,label="Kewley01")
        ax.set_xlim(*LIM[cx]); ax.set_ylim(*LIM[cy]); ax.set_xlabel(lx); ax.set_ylabel(ly)
    h=[plt.Line2D([],[],marker="o",ls="",color=GRID_C,label="Byler grid"),
       plt.Line2D([],[],marker="o",ls="",color=IN_C,label="galaxy, in grid hull"),
       plt.Line2D([],[],marker="o",ls="",color=OUT_C,label=f"galaxy, outside hull ({out.mean():.0%})")]
    axes[0].legend(handles=h,loc="upper left")
    fig.suptitle(f"{nm}: dust-insensitive 3-D coverage by the Byler+2017 grid  (membership from full 3-D)",fontsize=13)
    for e in ("png","pdf"): fig.savefig(REPO+f"figs/byler_3d_{nm.lower()}.{e}",dpi=170,bbox_inches="tight")
    print(f"  Wrote figs/byler_3d_{nm.lower()}.png")
