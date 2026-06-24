# summarize_training_samples.py
from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

FLUX_SCALE = 1e-17  # both SDSS and DESI Halpha fluxes are in 1e-17 erg/s/cm^2

def log10_lha(z, ha_flux_1e17):
    """Return log10 L(Ha) [erg/s] from z and Halpha flux in 1e-17 units."""
    z = np.asarray(z, dtype=float)
    f = np.asarray(ha_flux_1e17, dtype=float) * FLUX_SCALE
    dl_cm = cosmo.luminosity_distance(z).to("cm").value
    return np.log10(f) + np.log10(4*np.pi) + 2*np.log10(dl_cm)

def summary(label, fits_path, *, z_col, ha_col, logm_col="LOGM_COLOR"):
    t = Table.read(fits_path, hdu=1)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    t = t[names]

    z = np.asarray(t[z_col], dtype=float)
    ha = np.asarray(t[ha_col], dtype=float)
    logm = np.asarray(t[logm_col], dtype=float)

    # require finite and positive Ha for luminosity
    m = np.isfinite(z) & np.isfinite(ha) & (ha > 0) & np.isfinite(logm)
    z = z[m]; ha = ha[m]; logm = logm[m]
    logLha = log10_lha(z, ha)

    def stats(x):
        return float(np.mean(x)), float(np.median(x))

    z_mean, z_med = stats(z)
    m_mean, m_med = stats(logm)
    l_mean, l_med = stats(logLha)

    print(f"\n=== {label} ===")
    print("File:", fits_path)
    print("N used:", len(z))
    print(f"z         mean/median: {z_mean:.5f} / {z_med:.5f}")
    print(f"logM*     mean/median: {m_mean:.3f} / {m_med:.3f}")
    print(f"logL(Ha)  mean/median: {l_mean:.3f} / {l_med:.3f}")

if __name__ == "__main__":
    desi_path = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/DESI_BGS_training_data.fits")
    sdss_path = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/SDSS_main_training_data.fits")

    summary("DESI BGS training sample", desi_path, z_col="Z", ha_col="HALPHA_FLUX")
    summary("SDSS Main training sample", sdss_path, z_col="Z_1", ha_col="H_ALPHA_FLUX")
