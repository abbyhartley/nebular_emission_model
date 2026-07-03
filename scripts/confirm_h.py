"""
Confirm FastSpecFit's distance/h convention behind the ABSMAGs.
ABSMAG = m_app - DM_fsf - KCORR  =>  DM_fsf = m_app - KCORR - ABSMAG.
Use DECam r: m_r = 22.5 - 2.5 log10(FLUX_SYNTH_PHOTMODEL_R) [nanomaggies],
ABSMAG10_DECAM_R, KCORR10_DECAM_R. Compare DM_fsf to Planck15 vs H0=100 (h=1).
"""
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck15, FlatLambdaCDM

t = Table.read("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/fastspec_zall_combined_selected.fits", hdu=1)
zcol = "Z" if "Z" in t.colnames else ("Z_1" if "Z_1" in t.colnames else None)
print("z col:", zcol, " N:", len(t))
z = np.asarray(t[zcol], float)
fr = np.asarray(t["FLUX_SYNTH_PHOTMODEL_R"], float)     # nanomaggies (DECam r)
absr = np.asarray(t["ABSMAG10_DECAM_R"], float)
kr = np.asarray(t["KCORR10_DECAM_R"], float)

g = np.isfinite(z) & (z > 0.01) & np.isfinite(fr) & (fr > 0) & np.isfinite(absr) & np.isfinite(kr)
m_r = 22.5 - 2.5 * np.log10(fr[g])
DM_fsf = m_r - kr[g] - absr[g]
DM_p15 = Planck15.distmod(z[g]).value
DM_h1 = FlatLambdaCDM(H0=100, Om0=Planck15.Om0).distmod(z[g]).value

d_p15 = np.median(DM_fsf - DM_p15)
d_h1 = np.median(DM_fsf - DM_h1)
print(f"\nmedian (DM_fsf - DM_Planck15) = {d_p15:+.3f} mag")
print(f"median (DM_fsf - DM_H0=100 )  = {d_h1:+.3f} mag")
print(f"\n5*log10(h_Planck15) = 5*log10({Planck15.h:.4f}) = {5*np.log10(Planck15.h):+.3f} mag")
print("(If DM_fsf matches H0=100, FastSpecFit ABSMAGs are in h=1 units -> ~0.85 mag fainter than physical.)")
