# build_altb_production.py
# Build the ALT-B production training samples (DESI + SDSS) via the EXISTING validated
# pipeline: write_filtered_fits_any with a custom SelectionConfig (continuum S/N>3,
# per-line S/N>3 on the 8 TARGET lines, Halpha left to IVAR>0 at training), z>0.05,
# ZWARN=0, GALAXY. Then replicate add_colorSM (LOGM_COLOR + DESI flux calibration +
# 0.1-99.9% mass trim). Non-destructive: writes *_ALTB files.
import sys
from pathlib import Path
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
sys.path.insert(0, str(REPO / "src"))
from normflow.selection import SelectionConfig, write_filtered_fits_any
from normflow.stellar_mass import log10_ml_r_from_gmr_sdss

BASE = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs")

# 8 TARGET lines (Halpha excluded -> only IVAR>0 enforced later, per the paper cut list)
DESI_TARGETS = ("OII_3726_FLUX", "OII_3729_FLUX", "HGAMMA_FLUX", "HBETA_FLUX",
                "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX")
SDSS_TARGETS = ("OII_3726_FLUX", "OII_3729_FLUX", "H_GAMMA_FLUX", "H_BETA_FLUX",
                "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6717_FLUX", "SII_6731_FLUX")
FLUX_CAL_DEX = {"OII_3726_FLUX": 0.0650, "OII_3729_FLUX": 0.0075, "HGAMMA_FLUX": 0.0676,
                "HBETA_FLUX": 0.0628, "HALPHA_FLUX": 0.0752, "OIII_5007_FLUX": 0.0502,
                "NII_6584_FLUX": 0.0449, "SII_6716_FLUX": 0.0635, "SII_6731_FLUX": 0.0545}
Msun_r, ZP = 4.64, 0.271


def masstrim_write(t, outfile, label):
    m = np.asarray(t["LOGM_COLOR"], float)
    finite = np.isfinite(m)
    lo, hi = np.percentile(m[finite], [0.1, 99.9])
    keep = finite & (m >= lo) & (m <= hi)
    t[keep].write(outfile, format="fits", overwrite=True)
    print(f"{label}: mass-trim [{lo:.3f},{hi:.3f}] kept {int(keep.sum()):,}/{len(t):,} -> {outfile}", flush=True)


def build_desi():
    sel = BASE / "fastspec_zall_combined_selected_ALTB.fits"
    cfg = SelectionConfig(survey="desi", z_min=0.05, snr_min=3.0, require_zwarn0=True,
                          require_spectype="GALAXY", require_line_snr=True, line_snr_min=3.0,
                          line_flux_cols=DESI_TARGETS)
    nsel, ntot = write_filtered_fits_any(BASE / "fastspec_zall_combined.fits", sel, cfg, verbose=False)
    print(f"DESI ALT-B selected {nsel:,}/{ntot:,}", flush=True)
    t = Table.read(sel, hdu=1)
    g01 = np.asarray(t["ABSMAG01_SDSS_G"], float); r01 = np.asarray(t["ABSMAG01_SDSS_R"], float)
    t["LOGM_COLOR"] = ((1.062 * (g01 - r01) - 0.555) + (-0.4 * (r01 - Msun_r)) + ZP).astype(np.float32)
    for col, dex in FLUX_CAL_DEX.items():
        t[col + "_ORIG"] = np.asarray(t[col], np.float32)
        t[col] = (np.asarray(t[col], float) * 10.0 ** dex).astype(np.float32)
    masstrim_write(t, BASE / "DESI_BGS_training_data_ALTB.fits", "DESI")


def build_sdss():
    sel = BASE / "mpa_rcsed2_combo_selected_ALTB.fits"
    cfg = SelectionConfig(survey="sdss", z_min=0.05, snr_min=3.0, require_zwarn0=True,
                          require_spectype="GALAXY", require_line_snr=True, line_snr_min=3.0,
                          line_flux_cols=SDSS_TARGETS)
    nsel, ntot = write_filtered_fits_any(BASE / "mpa_rcsed2_combo.fits", sel, cfg, verbose=False)
    print(f"SDSS ALT-B selected {nsel:,}/{ntot:,}", flush=True)
    t = Table.read(sel, hdu=1)
    z = np.asarray(t["Z_1"], float)
    Mg = np.asarray(t["corrmag_g"], float) - cosmo.distmod(z).value - np.asarray(t["kcorr_g"], float)
    Mr = np.asarray(t["corrmag_r"], float) - cosmo.distmod(z).value - np.asarray(t["kcorr_r"], float)
    logM = np.asarray(log10_ml_r_from_gmr_sdss(Mg - Mr), float) + (-0.4 * (Mr - Msun_r))
    t["LOGM_COLOR"] = logM.astype(np.float32)
    masstrim_write(t, BASE / "SDSS_main_training_data_ALTB.fits", "SDSS")


def main():
    print(">>> building DESI ALT-B", flush=True); build_desi()
    print(">>> building SDSS ALT-B", flush=True); build_sdss()
    print("=== builds complete ===", flush=True)


if __name__ == "__main__":
    main()
