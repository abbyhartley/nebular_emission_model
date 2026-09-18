"""
Overlay the Byler+2017 FSPS/Cloudy grid on the observed data in three diagnostic
planes, outline the grid's convex-hull envelope, and quantify the fraction of real
galaxies that lie OUTSIDE the envelope (regions no grid model reaches). Convex hull
is conservative (it bridges the concave locus), so the reported uncovered fraction
is a lower bound. Data only here (NF added next).
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

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
FLUX_SCALE = 1e-17
S = dict(name="SDSS", fits=BASE+"SDSS_main_training_data.fits", meta=REPO+"nf_sdss_main_meta.pkl", c="#0072B2")
D = dict(name="DESI", fits=BASE+"DESI_BGS_training_data.fits", meta=REPO+"nf_desi_bgs_meta.pkl", c="#E69F00")

def load_scalar_df(p):
    t=Table.read(p,hdu=1); return t[[n for n in t.colnames if len(t[n].shape)<=1]].to_pandas()

# CCM/O'Donnell R_V=3.1 extinction curve, A_lambda/E(B-V), at the line wavelengths
KLAW = dict(Ha=2.530, Hb=3.609, NII=2.524, SII16=2.489, SII31=2.484,
            OII26=4.771, OII29=4.766, OIII=3.468)

def get_ratios(fits, metap, deredden=True):
    meta=pickle.load(open(metap,"rb")); df=load_scalar_df(fits)
    tcols=meta["resolved"]["target_cols"]; raw=[c[6:] if c.startswith("LOG10_") else c for c in tcols]
    hacol=next(c for c in ["H_ALPHA_FLUX","HALPHA_FLUX"] if c in df.columns)
    U=[c.upper() for c in raw]
    def col(sub):
        i=next(k for k,c in enumerate(U) if any(s in c for s in ([sub] if isinstance(sub,str) else sub)))
        return df[raw[i]].to_numpy(float)
    F={n: col(s) for n,s in dict(Hb=["H_BETA","HBETA"], NII="NII_6584", SII16=["SII_6716","SII_6717"],
        SII31="SII_6731", OII26="OII_3726", OII29="OII_3729", OIII="OIII_5007").items()}
    F["Ha"]=df[hacol].to_numpy(float)
    good=np.all([np.isfinite(v)&(v>0) for v in F.values()],axis=0)
    F={k:v[good] for k,v in F.items()}
    if deredden:
        # E(B-V) from Balmer decrement vs case-B 2.86; clip negatives (noise) to 0
        ebv = np.clip(2.5/(KLAW["Hb"]-KLAW["Ha"]) * np.log10((F["Ha"]/F["Hb"])/2.86), 0, None)
        for k in F:  # F_int = F_obs * 10^{0.4 E(B-V) k(lambda)}
            F[k] = F[k] * 10.0**(0.4*ebv*KLAW[k])
    return pd.DataFrame(dict(
        nii_ha=np.log10(F["NII"]/F["Ha"]), oiii_hb=np.log10(F["OIII"]/F["Hb"]),
        oii_ha=np.log10((F["OII26"]+F["OII29"])/F["Ha"]), sii_ha=np.log10((F["SII16"]+F["SII31"])/F["Ha"])))

grid = pd.read_csv(REPO+"docs/byler_grid.csv")
dat = {c["name"]: get_ratios(c["fits"], c["meta"]) for c in (S,D)}

# 4-D convex-hull coverage (full ratio space)
cols4 = ["nii_ha","oiii_hb","oii_ha","sii_ha"]
hull4 = Delaunay(grid[cols4].to_numpy())
print("=== fraction of galaxies OUTSIDE the Byler grid convex hull (lower bound) ===")
for nm, d in dat.items():
    inside = hull4.find_simplex(d[cols4].to_numpy()) >= 0
    print(f"  {nm}: 4-D uncovered = {(~inside).mean():.1%}  (N={len(d)})")

planes = [("nii_ha","oiii_hb", r"$\log$([N II]/H$\alpha$)", r"$\log$([O III]/H$\beta$)"),
          ("nii_ha","oii_ha",  r"$\log$([N II]/H$\alpha$)", r"$\log$([O II]/H$\alpha$)"),
          ("oiii_hb","oii_ha", r"$\log$([O III]/H$\beta$)", r"$\log$([O II]/H$\alpha$)")]
plt.style.use(["science","no-latex"])
plt.rcParams.update({"axes.labelsize":13,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9,"axes.titlesize":11})
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
rng=np.random.default_rng(0)
for ax,(cx,cy,lx,ly) in zip(axes, planes):
    gx,gy=grid[cx].to_numpy(), grid[cy].to_numpy()
    try:
        h2=Delaunay(np.column_stack([gx,gy]))
        from scipy.spatial import ConvexHull
        ch=ConvexHull(np.column_stack([gx,gy]))
        for s in ch.simplices: ax.plot(np.column_stack([gx,gy])[s,0], np.column_stack([gx,gy])[s,1], color="0.4", lw=1.0)
    except Exception as e:
        print("hull err", e); h2=None
    for c,d in [(S,dat["SDSS"]),(D,dat["DESI"])]:
        idx=rng.choice(len(d), size=min(25000,len(d)), replace=False)
        ax.scatter(d[cx].to_numpy()[idx], d[cy].to_numpy()[idx], s=2, alpha=0.05, color=c["c"], rasterized=True)
        upct = (h2.find_simplex(d[[cx,cy]].to_numpy())<0).mean() if h2 is not None else np.nan
        print(f"  {c['name']} {cx}-{cy}: out-of-hull = {upct:.1%}")
        ax.scatter([],[],color=c["c"],label=f"{c['name']} (out {upct:.0%})")
    ax.scatter(gx,gy,s=8,color="k",alpha=0.5,label="Byler grid",zorder=5)
    ax.set_xlabel(lx); ax.set_ylabel(ly); ax.legend(loc="best")
fig.suptitle("Byler+2017 FSPS/Cloudy grid vs dust-corrected line ratios (grid convex hull outlined)", fontsize=13)
for e in ("png","pdf"): fig.savefig(REPO+f"figs/byler_vs_data.{e}", dpi=160, bbox_inches="tight")
print("Wrote figs/byler_vs_data.png")
