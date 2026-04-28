# normflow is a python package that will contain all the functions I frequenctly reuse in this modeling project
# I'll import these functions as a package and call them in Sherlock
from .selection import DESISelectionConfig, desi_bgs_training_mask
from .stellar_mass import (absmag, restframe_color_gmr, desi_to_sdss_gmr, log10_ml_r_from_gmr_sdss, log10_stellar_mass_color)

__all__ = ["DESISelectionConfig", "desi_bgs_training_mask", "absmag",
    "restframe_color_gmr",
    "desi_to_sdss_gmr",
    "log10_ml_r_from_gmr_sdss",
    "log10_stellar_mass_color"]
