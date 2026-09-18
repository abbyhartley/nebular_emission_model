from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

infile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")

# columns
z_col = "Z"
ha_flux_col = "HALPHA_FLUX"
logm_col = "LOGM_COLOR"

# SDSS line fluxes are in 1e-17 erg/s/cm^2
FLUX_SCALE = 1e-17

# if you want the *same subset* used for training (all 9 lines > 0), include these:
line_flux_cols = [
    "HBETA_FLUX",
    "HGAMMA_FLUX",
    "NII_6584_FLUX",
    "SII_6716_FLUX",
    "SII_6731_FLUX",
    "OII_3726_FLUX",
    "OII_3729_FLUX",
    "OIII_5007_FLUX",
]

t = Table.read(infile, hdu=1)
names = [name for name in t.colnames if len(t[name].shape) <= 1]
t = t[names]

z = np.asarray(t[z_col], dtype=float)
ha = np.asarray(t[ha_flux_col], dtype=float) * FLUX_SCALE
logm = np.asarray(t[logm_col], dtype=float)

# Base validity
mask = np.isfinite(z) & np.isfinite(ha) & (ha > 0) & np.isfinite(logm)

# Option: match the "all 9 lines > 0" training subset
for c in line_flux_cols:
    x = np.asarray(t[c], dtype=float) * FLUX_SCALE
    mask &= np.isfinite(x) & (x > 0)

z = z[mask]
ha = ha[mask]

# Compute log L(Ha)
dl_cm = cosmo.luminosity_distance(z).to("cm").value
logLHa = np.log10(ha) + np.log10(4*np.pi) + 2*np.log10(dl_cm)

p1, p50, p99 = np.percentile(logLHa, [1, 50, 99])
print("N used:", logLHa.size)
print("LOG_LHA percentiles (1/50/99):", float(p1), float(p50), float(p99))
print("LOG_LHA min/max:", float(logLHa.min()), float(logLHa.max()))

# Optional: how many are above some extreme thresholds?
for thr in [43, 44, 45, 46]:
    print(f"N(LOG_LHA > {thr}):", int(np.sum(logLHa > thr)), f"({np.mean(logLHa > thr):.3%})")
