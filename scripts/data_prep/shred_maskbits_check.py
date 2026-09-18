import numpy as np
from astropy.io import fits
G = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs"

# ALT-B DESI sample TARGETIDs
alt = fits.open(G+"/DESI_BGS_training_data_ALTB.fits", memmap=True)[1].data
tid_alt = np.asarray(alt["TARGETID"], dtype=np.int64)
n = len(tid_alt)

# zall-pix-iron ZCATALOG: TARGETID + MASKBITS (+ SURVEY/PROGRAM to isolate main/bright)
zf = fits.open(G+"/zall-pix-iron.fits", memmap=True)
hdu = None
for h in zf[1:]:
    try:
        if "MASKBITS" in h.columns.names and "TARGETID" in h.columns.names:
            hdu = h; break
    except Exception:
        pass
print("ZCATALOG cols incl:", [c for c in hdu.columns.names if c.upper() in ("TARGETID","MASKBITS","SURVEY","PROGRAM")])
d = hdu.data
tid_z = np.asarray(d["TARGETID"], np.int64)
mb_z = np.asarray(d["MASKBITS"]).astype(np.int64)
if "SURVEY" in hdu.columns.names and "PROGRAM" in hdu.columns.names:
    sv = np.char.strip(np.asarray(d["SURVEY"]).astype(str))
    pg = np.char.strip(np.asarray(d["PROGRAM"]).astype(str))
    keep = (sv == "main") & (pg == "bright")
    tid_z, mb_z = tid_z[keep], mb_z[keep]
    print("restricted zall to main/bright:", keep.sum(), "of", len(keep))

# dedupe by TARGETID, then lookup via sorted search
u, idx = np.unique(tid_z, return_index=True)
mb_u = mb_z[idx]
pos = np.searchsorted(u, tid_alt); pos = np.clip(pos, 0, len(u)-1)
matched = u[pos] == tid_alt
mb_alt = mb_u[pos][matched]
nm = matched.sum()
print(f"\nALT-B N = {n:,};  matched to zall MASKBITS = {nm:,} ({nm/n:.1%})")
print(f"MASKBITS == 0 (no imaging mask): {(mb_alt==0).sum():,} ({(mb_alt==0).mean():.2%})")
print(f"MASKBITS != 0 (any bit set):     {(mb_alt!=0).sum():,} ({(mb_alt!=0).mean():.2%})")
print("per-bit fraction of matched sample (Legacy DR9 MASKBITS):")
names = {1:"BRIGHT(star)",5:"ALLMASK_G",6:"ALLMASK_R",7:"ALLMASK_Z",8:"WISEM1",9:"WISEM2",
         10:"BAILOUT",11:"MEDIUM(star)",12:"GALAXY(SGA large gal)",13:"CLUSTER(globular)"}
for b in range(16):
    frac = (((mb_alt >> b) & 1) == 1).mean()
    if frac > 0:
        print(f"  bit {b:2d} (val {1<<b:6d}) {names.get(b,'?'):22s}: {frac:.3%}")
