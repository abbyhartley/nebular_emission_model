# sample selection function for DESI BGS galaxies used to train the model
import numpy as np
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Union

try:
    from astropy.table import Table
except ImportError:  
    Table = None

DEFAULT_LINE_FLUX_COLS = ("OII_3726_FLUX", "OII_3729_FLUX",
    "HGAMMA_FLUX", "HBETA_FLUX", "HALPHA_FLUX",
    "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX")

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
    *,
    return_table: bool = False):
    """
    Build a boolean mask for your DESI BGS fastspecfit training sample.

    Cuts:
      - SPECTYPE == 'GALAXY'
      - (optional) ZWARN == 0
      - Z > z_min
      - snr_col > snr_min
      - all emission line fluxes > 0 for config.line_flux_cols
      - all of the above must be finite (no NaNs)

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

    # SPECTYPE cut
    spectype = np.asarray(t["SPECTYPE"]).astype(str)
    mask &= (spectype == config.require_spectype)

    # Z cut (+ finiteness)
    z = np.asarray(t["Z"])
    mask &= np.isfinite(z) & (z > config.z_min)

    # ZWARN cut
    if config.require_zwarn0:
        zwarn = np.asarray(t["ZWARN"])
        # zwarn is typically integer; finiteness check is harmless
        mask &= np.isfinite(zwarn) & (zwarn == 0)

    # S/N cut (+ finiteness)
    snr = np.asarray(t[config.snr_col])
    mask &= np.isfinite(snr) & (snr > config.snr_min)

    # Emission line flux positivity cuts (+ finiteness)
    for col in config.line_flux_cols:
        x = np.asarray(t[col])
        mask &= np.isfinite(x) & (x > 0)

    if return_table:
        return mask, t[mask]
    return mask

def write_filtered_fits(infile: Union["Table", str, Path],
    outfile: Union[str, Path],
    config: DESISelectionConfig = DESISelectionConfig()):
    """
    Apply desi_bgs_training_mask and write the filtered table to a FITS file.

    Parameters
    ----------
    infile : astropy.table.Table or path
        Input table or FITS filename.
    outfile : str or Path
        Output FITS filename.
    config : DESISelectionConfig
        Selection thresholds and column names.

    Returns
    -------
    n_selected : int
        Number of rows passing the selection.
    n_total : int
        Total number of rows in the input table.
    """
    mask, tsel = desi_bgs_training_mask(infile, config, return_table=True)
    tsel.write(str(outfile), overwrite=True)
    return int(mask.sum()), int(len(mask))
