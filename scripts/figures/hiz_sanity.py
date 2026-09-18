import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa
plt.style.use(["science","no-latex"]); plt.rcParams.update({"axes.labelsize":13,"xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":10})
LC="/oak/stanford/orgs/kipac/users/risahw/aemulus_lightcones/results/box017_v1"
CM="/oak/stanford/orgs/kipac/data/cosmos"
GFC="/oak/stanford/orgs/kipac/users" and "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs"
KE=41.27  # log L_Ha = log SFR + 41.27 (Kennicutt&Evans 2012, Chabrier)

# ---- Risa's COSMOS comparison shells ----
cc=json.load(open(LC+"/cosmos_compare.json"))
print("Risa cosmos_compare: area_cosmos=%.2f deg2, lc_n_used=%d, z_mode=%s"%(cc.get("cosmos_area_deg2",-1),cc.get("lc_n_used",-1),cc.get("z_mode")))
for s in cc.get("shells",[])[:8]:
    ks=s.get("smf_ks",s.get("ks",s.get("sm_ks")))
    print("  shell z[%s,%s]: keys=%s"%(s.get("z_lo"),s.get("z_hi"),[k for k in s.keys()][:10]))

# ---- load lightcone (obs_sm, obs_sfr, z_obs) ----
import pyarrow.parquet as pq
lc=pq.read_table(LC+"/desi_cosmos_z1.60_um_sm.parquet", columns=["z_obs","obs_sm","obs_sfr","ssfr"]).to_pandas()
z=lc["z_obs"].to_numpy(float); m=np.log10(np.clip(lc["obs_sm"].to_numpy(float),1,None))
sfr=lc["obs_sfr"].to_numpy(float); ls=np.where(sfr>0, np.log10(sfr), np.nan); lha=ls+KE
sf = (sfr>0) & (lc["ssfr"].to_numpy(float) > 1e-11)
print("lightcone: N=%d  logM* 1/50/99=%s  logSFR 1/50/99=%s"%(len(z),
      np.round(np.nanpercentile(m,[1,50,99]),2), np.round(np.nanpercentile(ls,[1,50,99]),2)))

# ---- COSMOS2020 (lp_mass_med, lp_SFR_med, lp_zBEST) ----
from astropy.io import fits
c=fits.open(CM+"/COSMOS2020_FARMER.fits", memmap=True)[1].data
cz=np.asarray(c["lp_zBEST"],float); cm_=np.asarray(c["lp_mass_med"],float); cs=np.asarray(c["lp_SFR_med"],float)
if np.nanmedian(cm_)>100: cm_=np.log10(np.clip(cm_,1,None))     # auto-detect log vs linear
if np.nanmedian(cs[np.isfinite(cs)&(cs>0)] if np.any(cs>0) else [1])>100: cs=np.log10(np.clip(cs,1e-30,None))
cok=np.isfinite(cz)&(cz>0)&np.isfinite(cm_)&(cm_>6)&(cm_<12.5)&np.isfinite(cs)&(cs>-5)&(cs<4)
print("COSMOS2020 clean N=%d  logM* 1/50/99=%s  logSFR 1/50/99=%s"%(cok.sum(),
      np.round(np.nanpercentile(cm_[cok],[1,50,99]),2), np.round(np.nanpercentile(cs[cok],[1,50,99]),2)))

# ---- DESI BGS ALT-B training domain (logM*, logL_Ha_obs) ----
from astropy.cosmology import Planck15 as cosmo
d=fits.open(GFC+"/DESI_BGS_training_data_ALTB.fits", memmap=True)[1].data
dz=np.asarray(d["Z"],float); dm=np.asarray(d["LOGM_COLOR"],float); dha=np.asarray(d["HALPHA_FLUX"],float)
ok=np.isfinite(dz)&(dz>0)&(dha>0)&np.isfinite(dm)
dl=np.full(dha.shape,np.nan)
dl[ok]=np.log10(dha[ok]*1e-17)+np.log10(4*np.pi)+2*np.log10(cosmo.luminosity_distance(dz[ok]).to("cm").value)
mlo,mhi=np.nanpercentile(dm[ok],[1,99]); llo,lhi=np.nanpercentile(dl[ok],[1,99])
print("DESI training domain (1-99pct): logM*=[%.2f,%.2f]  logL_Ha=[%.2f,%.2f]"%(mlo,mhi,llo,lhi))

# ---- domain overlap of lightcone (SF, using intrinsic L_Ha from SFR) ----
inbox = sf & (m>=mlo)&(m<=mhi)&(lha>=llo)&(lha<=lhi)
print("\nDOMAIN OVERLAP (lightcone SF galaxies inside DESI training box):")
print("  overall: %.1f%% of all lightcone; %.1f%% of SF lightcone"%(100*inbox.mean(), 100*inbox.sum()/max(sf.sum(),1)))
for zl,zh in [(0.1,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.1),(1.1,1.3),(1.3,1.6)]:
    sh=(z>=zl)&(z<zh); shsf=sh&sf
    print("  z[%.1f,%.1f]: N=%d  SF frac=%.2f  in-domain(of SF)=%.1f%%  medlogM*=%.2f medlogSFR=%.2f"%(
        zl,zh,sh.sum(), shsf.sum()/max(sh.sum(),1), 100*(inbox&sh).sum()/max(shsf.sum(),1),
        np.nanmedian(m[shsf]) if shsf.sum() else np.nan, np.nanmedian(ls[shsf]) if shsf.sum() else np.nan))

# ---- figure: (a) M* dist by z lc vs cosmos, (b) SFMS by z, (c) (logM*,logL_Ha) domain ----
fig,ax=plt.subplots(1,3,figsize=(17,5.2),constrained_layout=True)
zc=[(0.3,0.5),(0.7,0.9),(1.1,1.3)]; cols=["#0072B2","#009E73","#D55E00"]
for (zl,zh),col in zip(zc,cols):
    a=(z>=zl)&(z<zh); b=cok&(cz>=zl)&(cz<zh)
    bins=np.linspace(7,12,26)
    ax[0].hist(m[a],bins=bins,density=True,histtype="step",lw=2,color=col,label="LC %.1f-%.1f"%(zl,zh))
    ax[0].hist(cm_[b],bins=bins,density=True,histtype="step",lw=2,ls="--",color=col)
    # SFMS ridge
    mb=np.linspace(9,11.5,12)
    for src,mm,ss,style in [("lc",m[a&sf],ls[a&sf],"-"),("cos",cm_[b],cs[b],"--")]:
        med=[np.nanmedian(ss[(mm>=mb[i])&(mm<mb[i+1])]) if ((mm>=mb[i])&(mm<mb[i+1])).sum()>20 else np.nan for i in range(len(mb)-1)]
        ax[1].plot(0.5*(mb[:-1]+mb[1:]),med,style,color=col,lw=2)
ax[0].set_xlabel(r"$\log M_\star$"); ax[0].set_ylabel("normalized"); ax[0].set_title("M* dist: LC (solid) vs COSMOS2020 (dashed)"); ax[0].legend()
ax[1].set_xlabel(r"$\log M_\star$"); ax[1].set_ylabel(r"$\log$ SFR"); ax[1].set_title("SFMS: LC (solid) vs COSMOS2020 (dashed)")
# domain panel
ax[2].hexbin(dm[ok],dl[ok],gridsize=45,bins="log",cmap="Greys",mincnt=3)
ss2=sf & np.isfinite(m)&np.isfinite(lha)
idx=np.random.default_rng(0).choice(np.where(ss2)[0], size=min(40000,ss2.sum()), replace=False)
sc=ax[2].scatter(m[idx],lha[idx],c=z[idx],s=2,alpha=0.3,cmap="plasma",vmin=0,vmax=1.6)
ax[2].plot([mlo,mhi,mhi,mlo,mlo],[llo,llo,lhi,lhi,llo],"r-",lw=1.5)
ax[2].set_xlim(6.5,12); ax[2].set_ylim(37.5,43.5)
ax[2].set_xlabel(r"$\log M_\star$"); ax[2].set_ylabel(r"$\log L_{H\alpha}$ (intrinsic, from SFR)")
ax[2].set_title("DESI training (grey) vs lightcone SF (pts); red=train 1-99% box")
fig.colorbar(sc,ax=ax[2],label="lightcone z")
out=GFC+"/nebular_emission_model/figs_ALTB/hiz_sanity.png"
fig.savefig(out,dpi=180,bbox_inches="tight"); print("\nSaved:",out)
