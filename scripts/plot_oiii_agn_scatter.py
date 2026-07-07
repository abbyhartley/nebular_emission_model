"""
Illustrate how much of the residual [O III] cross-survey transfer scatter is
driven by AGN/composite excitation.
 Top row:   [O III] transfer residual (pred - true, log[OIII]/Ha) histograms split
            by NII-BPT class (SF / composite / AGN), one panel per direction.
 Bottom row: NII-BPT plane (log[NII]/Ha vs log[OIII]/Hb) hexbin-colored by the
            median |[O III] residual|, with Kauffmann03 / Kewley01 demarcations.
Also prints per-class scatter / bias / rho.
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa
try:
    import cmasher as cmr; CMAP = cmr.bubblegum
except Exception:
    CMAP = "magma"

BASE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/"
REPO = BASE + "nebular_emission_model/"
FLUX_SCALE = 1e-17; SEED = 0; N_MC = 50
S = dict(name="SDSS", fits=BASE+"SDSS_main_training_data.fits", eqx=REPO+"nf_sdss_main.eqx", meta=REPO+"nf_sdss_main_meta.pkl", key="sdss")
D = dict(name="DESI", fits=BASE+"DESI_BGS_training_data.fits", eqx=REPO+"nf_desi_bgs.eqx", meta=REPO+"nf_desi_bgs_meta.pkl", key="desi")
CLASS_COL = {"SF": "#0072B2", "composite": "#E69F00", "AGN": "#CC79A7"}


def load_scalar_df(p):
    t = Table.read(p, hdu=1); return t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()

def add_loglha(df, survey):
    if survey == "sdss": z=df["Z_1"].to_numpy(float); ha=df["H_ALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    else: z=df["Z"].to_numpy(float); ha=df["HALPHA_FLUX"].to_numpy(float)*FLUX_SCALE
    m=np.isfinite(z)&(z>0)&np.isfinite(ha)&(ha>0); out=np.full(len(df),np.nan)
    out[m]=np.log10(ha[m])+np.log10(4*np.pi)+2*np.log10(cosmo.luminosity_distance(z[m]).to("cm").value)
    df=df.copy(); df["LOG_LHA"]=out; return df

def load_flow(fp, meta):
    xdim=len(meta["resolved"]["out_cols"])
    tmpl=block_neural_autoregressive_flow(key=jr.key(int(meta.get("seed",0))), base_dist=Normal(jnp.zeros(xdim)), cond_dim=2)
    return eqx.tree_deserialise_leaves(fp, tmpl)

def get_XU(df, meta):
    tcols=meta["resolved"]["target_cols"]; raw=[c[6:] if c.startswith("LOG10_") else c for c in tcols]
    ha=next(c for c in ["H_ALPHA_FLUX","HALPHA_FLUX"] if c in df.columns); logm=meta["resolved"]["logmstar_col"]
    F=np.column_stack([df[c].to_numpy(float) for c in raw]); fha=df[ha].to_numpy(float)
    u1=df[logm].to_numpy(float); u2=df["LOG_LHA"].to_numpy(float)
    good=np.all(F>0,axis=1)&(fha>0)&np.isfinite(fha)&np.all(np.isfinite(F),axis=1)&np.isfinite(u1)&np.isfinite(u2)
    return np.log10(F[good])-np.log10(fha[good])[:,None], np.column_stack([u1[good],u2[good]]), [c.upper() for c in raw]

def idx_of(cols,*subs): return next(i for i,c in enumerate(cols) if any(s in c for s in subs))

def bpt(X, cols):
    nii=X[:, idx_of(cols,"NII_6584")]
    oiii=X[:, idx_of(cols,"OIII_5007")]-X[:, idx_of(cols,"H_BETA","HBETA")]
    ka=np.where(nii<0.05, 0.61/(nii-0.05)+1.30, -np.inf)
    ke=np.where(nii<0.47, 0.61/(nii-0.47)+1.19, -np.inf)
    sf=oiii<ka; agn=(oiii>ke)|(nii>=0.47); comp=(~sf)&(~agn)
    return dict(SF=sf, composite=comp, AGN=agn), nii, oiii

def sample_ratios(flow, meta, U, seed=0):
    Un=((U-meta["U_mean"])/meta["U_std"]).astype(np.float32); acc=np.zeros((len(U),len(meta["X_std"])))
    for j in range(N_MC):
        kk=jr.split(jr.key(seed+j), len(U))
        acc+=np.array(jax.vmap(lambda a,u: flow.sample(a,sample_shape=(),condition=u))(kk, jnp.asarray(Un)))
    return (acc/N_MC)*meta["X_std"]+meta["X_mean"]

def scat(r): p16,p84=np.percentile(r,[16,84]); return 0.5*(p84-p16)

metas={c["name"]:pickle.load(open(c["meta"],"rb")) for c in (S,D)}
flows={c["name"]:load_flow(c["eqx"],metas[c["name"]]) for c in (S,D)}
XU={c["name"]:get_XU(add_loglha(load_scalar_df(c["fits"]),c["key"]), metas[c["name"]]) for c in (S,D)}

res={}  # direction -> (nii, oiii_hb, residual, masks)
for tr, te in [(S,"DESI"), (D,"SDSS")]:
    Xt,Ut,cols=XU[te]; oi=idx_of(cols,"OIII_5007")
    pred=sample_ratios(flows[tr["name"]], metas[tr["name"]], Ut, seed=SEED+11)
    r=pred[:,oi]-Xt[:,oi]
    masks,nii,oiii_hb=bpt(Xt,cols)
    res[f"{tr['name']}->{te}"]=(nii,oiii_hb,r,masks)
    print(f"\n{tr['name']}->{te}  [OIII] residual by class:")
    for cl,mk in masks.items():
        print(f"  {cl:9s} N={int(mk.sum()):6d} ({mk.mean():4.0%})  scatter={scat(r[mk]):.3f}  bias={np.median(r[mk]):+.3f}  rho={spearmanr(pred[mk,oi],Xt[mk,oi]).correlation:.3f}")

plt.style.use(["science","no-latex"])
plt.rcParams.update({"axes.labelsize":13,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":10,"axes.titlesize":12})
fig, ax = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
dirs=list(res.keys())
xn=np.linspace(-1.6,0.3,200)
ka_c=0.61/(xn-0.05)+1.30; ke_c=0.61/(xn-0.47)+1.19
for j,dn in enumerate(dirs):
    nii,oiii_hb,r,masks=res[dn]
    a=ax[0,j]
    for cl,mk in masks.items():
        a.hist(r[mk], bins=np.linspace(-1.2,1.2,61), density=True, histtype="step", lw=2,
               color=CLASS_COL[cl], label=f"{cl} ({mk.mean():.0%}, $\\sigma$={scat(r[mk]):.2f})")
    a.axvline(0, color="k", ls=":", lw=1); a.set_xlabel(r"[O III] residual $\log L_{\rm pred}-\log L_{\rm true}$ [dex]")
    a.set_ylabel("normalized density"); a.set_title(dn.replace('->', r' $\rightarrow$ ')); a.legend()
    b=ax[1,j]
    hb=b.hexbin(nii, oiii_hb, C=np.abs(r), gridsize=55, reduce_C_function=np.median,
                cmap=CMAP, extent=(-1.6,0.4,-1.2,1.3), mincnt=8)
    m1=xn<0.05; m2=xn<0.47
    b.plot(xn[m1], ka_c[m1], "k--", lw=1.5, label="Kauffmann03")
    b.plot(xn[m2], ke_c[m2], "k-.", lw=1.5, label="Kewley01")
    b.set_xlabel(r"$\log$([N II]/H$\alpha$)"); b.set_ylabel(r"$\log$([O III]/H$\beta$)")
    b.set_xlim(-1.6,0.4); b.set_ylim(-1.2,1.3); b.legend(loc="lower left")
    cb=fig.colorbar(hb, ax=b, shrink=0.9); cb.set_label(r"median $|$[O III] residual$|$ [dex]")
fig.suptitle("Residual [O III] cross-survey scatter is concentrated in AGN/composite excitation", fontsize=13)
for e in ("png","pdf"): fig.savefig(REPO+f"figs/oiii_agn_scatter.{e}", dpi=180, bbox_inches="tight")
print("\nWrote figs/oiii_agn_scatter.png")
