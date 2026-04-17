# sample selection function for DESI BGS galaxies used to train the model
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

try:
    from astropy.table import Table
except ImportError:  # lets docs/import not hard-fail in some environments
    Table = None


DEFAULT_LINE_FLUX_COLS = ("OII_3726_FLUX", "OII_3729_FLUX",
    "HGAMMA_FLUX", "HBETA_FLUX", "HALPHA_FLUX",
    "OIII_5007_FLUX", "NII_6584_FLUX",
    "SII_6716_FLUX","SII_6731_FLUX")


@dataclass(frozen=True)
class DESISelectionConfig:
    z_min: float = 0.05
    require_spectype: str = "GALAXY"
    require_zwarn0: bool = True
    snr_min: float = 5.0
    snr_col: str = "SNR_R"  
    line_flux_cols: tuple[str, ...] = DEFAULT_LINE_FLUX_COLS


def _as_table(data_or_filename: Union["Table", str, Path]):
    if Table is None:
        raise ImportError("astropy is required to load FITS tables (pip install astropy).")
    if hasattr(data_or_filename, "colnames"):
        return data_or_filename
    return Table.read(str(data_or_filename))


def desi_bgs_training_mask(
    data_or_filename: Union["Table", str, Path],
    config: DESISelectionConfig = DESISelectionConfig(),
    *, return_table: bool = False):
    """
    Build a boolean mask for your DESI BGS fastspecfit training sample.

    Cuts:
      - SPECTYPE == 'GALAXY'
      - (optional) ZWARN == 0
      - Z > z_min
      - snr_col > snr_min
      - all emission line fluxes > 0 for config.line_flux_cols

    Parameters
    ----------
    data_or_filename : astropy.table.Table or path
        Fastspecfit combined table (or already-loaded Table).
    config : DESISelectionConfig
        Selection thresholds and column names.
    return_table : bool
        If True, return (mask, filtered_table). Else return mask only.

    Returns
    -------
    mask : np.ndarray[bool]
    (optional) filtered : astropy.table.Table
    """
    t = _as_table(data_or_filename)

    # Basic required columns check (fail fast with a useful error)
    required = ["SPECTYPE", "Z"]
    if config.require_zwarn0:
        required.append("ZWARN")
    required.append(config.snr_col)
    required += list(config.line_flux_cols)

    missing = [c for c in required if c not in t.colnames]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    mask = np.ones(len(t), dtype=bool)

    mask &= (np.asarray(t["SPECTYPE"]).astype(str) == config.require_spectype)
    mask &= (np.asarray(t["Z"]) > config.z_min)

    if config.require_zwarn0:
        mask &= (np.asarray(t["ZWARN"]) == 0)

    # Median S/N cut
    mask &= (np.asarray(t[config.snr_col]) > config.snr_min)

    # Emission line flux positivity cuts
    for col in config.line_flux_cols:
        mask &= (np.asarray(t[col]) > 0)

    if return_table:
        return mask, t[mask]
    return mask
