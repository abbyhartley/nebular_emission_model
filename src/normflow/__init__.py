# normflow is a python package that will contain all the functions I frequenctly reuse in this modeling project
# src/normflow/__init__.py

# Keep __init__ lightweight: don't hard-fail if optional submodules change.
# Users can still import from submodules directly.

from .stellar_mass import (
    desi_to_sdss_gmr,
    log10_ml_r_from_gmr_sdss,
    log10_stellar_mass_color,
)

__all__ = [
    "desi_to_sdss_gmr",
    "log10_ml_r_from_gmr_sdss",
    "log10_stellar_mass_color",
]

# Optional re-exports (do not crash if missing)
try:
    from .selection import DESISelectionConfig, desi_bgs_training_mask, write_filtered_fits_all_hdus
    __all__ += [
        "DESISelectionConfig",
        "desi_bgs_training_mask",
        "write_filtered_fits_all_hdus",
    ]
except Exception:
    # selection API may differ across versions; import from normflow.selection directly if needed
    pass
