"""
Dust-insensitive coverage test: is the observed distribution contained in the
Byler+2017 grid's envelope in the 3-D space of the three dust-insensitive ratios
[N II]/Ha, [O III]/Hb, [S II]/Ha (no dereddening needed; 3 grid parameters -> a
3-D image that can fill a 3-D volume, so the test is well posed).
Reports the out-of-hull fraction, splits it by BPT class, and colors the data by
in/out of the 3-D hull. Kewley01 / Kauffmann03 curves drawn on the BPT panel.
"""
import os
os.environ.setdefault("SPS_HOME", "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/FSPS")
import pickle
import numpy as np, pandas as pd
from astropy.table import Table
from scipy.spatial import Delaunay, ConvexHull
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
S = dict(name="SDSS", fits=BASE+"SDSS_main_training_data.fits", meta=REPO+"nf_sdss_main_meta.pkl", c="#0072B2")
D = dict(name="DESI", fits=BASE+"DESI_BGS_training_data.fits", meta=REPO+"nf_desi_bgs_meta.pkl", c="#E69F00")

def load_scalar_df(p):
    t=Table.read(p,hdu=1); return t[[n for n in t.colnames if len(t[n].shape)<=1]].to_pandas()

def get_ratios(fits, metap):
    """Dust-insensitive ratios only -> no dereddening required."""
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

grid = pd.read_csv(REPO+"docs/byler_grid.csv")
cols3 = ["nii_ha","oiii_hb","sii_ha"]
hull3 = Delaunay(grid[cols3].to_numpy())
dat = {c["name"]: get_ratios(c["fits"], c["meta"]) for c in (S,D)}

def kau(x): return np.where(x<0.05, 0.61/(x-0.05)+1.30, -np.inf)   # Kauffmann03 (SF/composite)
def kew(x): return np.where(x<0.47, 0.61/(x-0.47)+1.19, -np.inf)   # Kewley01 (max starburst)

print("=== 3-D dust-insensitive ([NII]/Ha,[OIII]/Hb,[SII]/Ha) coverage ===")
outmask={}
for nm,d in dat.items():
    inside = hull3.find_simplex(d[cols3].to_numpy())>=0
    out = ~inside; outmask[nm]=out
    agn_comp = d["oiii_hb"].to_numpy() > kau(d["nii_ha"].to_numpy())   # above Kauffmann = composite/AGN
    sf = ~agn_comp
    print(f"  {nm}: out-of-hull = {out.mean():.1%}  (N={len(d)})")
    print(f"        of the out-of-hull galaxies, {(agn_comp&out).sum()/max(out.sum(),1):.0%} are composite/AGN (above Kauffmann03)")
    print(f"        out-of-hull among SF-only (below Kauffmann03) = {(out&sf).sum()/max(sf.sum(),1):.1%}")

# figure: 3 dust-insensitive planes; data colored in(gray)/out(red); grid + 2D hull; curves on BPT
planes=[("nii_ha","oiii_hb",r"$\log$([N II]/H$\alpha$)",r"$\log$([O III]/H$\beta$)"),
        ("nii_ha","sii_ha", r"$\log$([N II]/H$\alpha$)",r"$\log$([S II]/H$\alpha$)"),
        ("oiii_hb","sii_ha",r"$\log$([O III]/H$\beta$)",r"$\log$([S II]/H$\alpha$)")]
plt.style.use(["science","no-latex"])
plt.rcParams.update({"axes.labelsize":13,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9,"axes.titlesize":11})
rng=np.random.default_rng(0)
for nm,c in [("SDSS",S),("DESI",D)]:
    d=dat[nm]; out=outmask[nm]
    fig,axes=plt.subplots(1,3,figsize=(16.5,5.3),constrained_layout=True)
    for ax,(cx,cy,lx,ly) in zip(axes,planes):
        gx,gy=grid[cx].to_numpy(),grid[cy].to_numpy()
        ch=ConvexHull(np.column_stack([gx,gy]))
        for s in ch.simplices: ax.plot(np.column_stack([gx,gy])[s,0],np.column_stack([gx,gy])[s,1],color="0.5",lw=0.9)
        idx=rng.choice(len(d),size=min(30000,len(d)),replace=False)
        ii=idx[~out[idx]]; oo=idx[out[idx]]
        ax.scatter(d[cx].to_numpy()[ii],d[cy].to_numpy()[ii],s=2,alpha=0.05,color="0.6",rasterized=True)
        ax.scatter(d[cx].to_numpy()[oo],d[cy].to_numpy()[oo],s=3,alpha=0.20,color="#D55E00",rasterized=True)
        ax.scatter(gx,gy,s=7,color="k",alpha=0.55,zorder=5)
        if (cx,cy)==("nii_ha","oiii_hb"):
            xs=np.linspace(-1.8,0.0,200)
            ax.plot(xs,kau(xs),"b--",lw=1.5,label="Kauffmann03");
            xk=np.linspace(-1.8,0.4,200); ax.plot(xk,kew(xk),"g-.",lw=1.5,label="Kewley01")
            ax.legend(loc="lower left")
        ax.set_xlabel(lx); ax.set_ylabel(ly)
    ax=axes[0]
    ax.scatter([],[],color="0.6",label="in grid hull"); ax.scatter([],[],color="#D55E00",label=f"outside hull ({out.mean():.0%})")
    ax.scatter([],[],color="k",label="Byler grid"); ax.legend(loc="upper left")
    fig.suptitle(f"{nm}: dust-insensitive 3-D coverage by the Byler+2017 grid (out-of-hull galaxies in orange)",fontsize=13)
    for e in ("png","pdf"): fig.savefig(REPO+f"figs/byler_3d_{nm.lower()}.{e}",dpi=160,bbox_inches="tight")
    print(f"Wrote figs/byler_3d_{nm.lower()}.png")
