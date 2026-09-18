"""
Sample selection functions for DESI BGS fastspecfit: 
- Per-table selection mask for training (finite checks, ZWARN==0, etc.)
- Multi-HDU FITS support: loop through all table HDUs, apply selection, and
  write a single stacked output FITS table with provenance column SRC_HDU
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union
import numpy as np

try:
    from astropy.table import Table, vstack
    from astropy.io import fits
except ImportError:  
    Table = None
    vstack = None
    fits = None


DEFAULT_LINE_FLUX_COLS = ("OII_3726_FLUX", "OII_3729_FLUX",
    "HGAMMA_FLUX", "HBETA_FLUX", "HALPHA_FLUX",
    "OIII_5007_FLUX", "NII_6584_FLUX", "SII_6716_FLUX", "SII_6731_FLUX")


@dataclass(frozen=True)
class DESISelectionConfig:
    z_min: float = 0.05
    require_zwarn0: bool = True
    snr_min: float = 5.0
    snr_col: str = "SNR_R" # Median signal-to-noise ratio per pixel in r camera.
    line_flux_cols: tuple[str, ...] = DEFAULT_LINE_FLUX_COLS

    # spectype cut applied only if this column exists in the table (since i want to generalize this to sdss later)
    spectype_col: str = "SPECTYPE"
    require_spectype: str | None = "GALAXY"


def _require_astropy():
    if Table is None or fits is None or vstack is None:
        raise ImportError("astropy is required (pip install astropy).")


def _as_table(data_or_filename: Union["Table", str, Path], *, hdu: int | None = None):
    """
    Convert input to an astropy Table.
    - If you pass a filename, you MUST specify hdu=... bc multi-HDU FITS otherwise just default to the first table HDU
    - For multi-HDU workflows, use write_filtered_fits_all_hdus()
    """
    _require_astropy()

    if hasattr(data_or_filename, "colnames"):
        return data_or_filename

    if hdu is None:
        raise ValueError("Input appears to be a filename. For safety with multi-HDU FITS, "
            "you must pass hdu=... (e.g. hdu=1), or use write_filtered_fits_all_hdus().")

    return Table.read(str(data_or_filename), hdu=hdu)


def desi_bgs_training_mask(
    data_or_filename: Union["Table", str, Path],
    config: DESISelectionConfig = DESISelectionConfig(),
    *,
    return_table: bool = False,
    hdu: int | None = None):
    """
    Build a boolean mask for a DESI BGS fastspecfit training sample (single Table)

    Cuts:
    - If config.spectype_col exists AND config.require_spectype is not None:
        spectype_col == require_spectype
    - Z finite and Z > z_min (0.05)
    - ZWARN finite and ZWARN == 0
    - SNR_R (or snr_col) finite and > snr_min
    - each emission line flux in line_flux_cols is finite and > 0 # WE MAY CHANGE THIS LATER!!!

    Parameters:
    data_or_filename : astropy.table.Table or path
        If path, you must supply hdu=...
    config : DESISelectionConfig
    return_table : bool
        If True, return (mask, filtered_table)
    hdu : int, optional
        Required if data_or_filename is a FITS filename

    Returns:
    mask : np.ndarray[bool]
    (optional) filtered : astropy.table.Table
    """
    t = _as_table(data_or_filename, hdu=hdu)

    required = ["Z", config.snr_col] # required columns
    if config.require_zwarn0:
        required.append("ZWARN")
    required += list(config.line_flux_cols)

    missing = [c for c in required if c not in t.colnames]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    mask = np.ones(len(t), dtype=bool)

    # SPECTYPE cut (only if column exists and requested)
    if config.require_spectype is not None and config.spectype_col in t.colnames:
        spectype = np.asarray(t[config.spectype_col]).astype(str)
        mask &= (spectype == config.require_spectype)

    # Z cut (+ finiteness)
    z = np.asarray(t["Z"])
    mask &= np.isfinite(z) & (z > config.z_min)

    # ZWARN cut (+ finiteness)
    if config.require_zwarn0:
        zwarn = np.asarray(t["ZWARN"])
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
    config: DESISelectionConfig = DESISelectionConfig(),
    *,
    hdu: int | None = None,
    add_src_hdu_col: bool = False):
    """
    Apply desi_bgs_training_mask and write the filtered table to a FITS file :D

    Use this for:
      - a single Table input, or
      - a FITS filename where you explicitly provide hdu=...

    For multi-HDU FITS, use write_filtered_fits_all_hdus().
    """
    _require_astropy()

    mask, tsel = desi_bgs_training_mask(infile, config, return_table=True, hdu=hdu)

    if add_src_hdu_col and hdu is not None:
        tsel = tsel.copy()
        tsel["SRC_HDU"] = np.full(len(tsel), int(hdu), dtype=np.int32)

    tsel.write(str(outfile), overwrite=True)
    return int(mask.sum()), int(len(mask))


def write_filtered_fits_all_hdus(infile: Union[str, Path],
    outfile: Union[str, Path],
    config: DESISelectionConfig = DESISelectionConfig(),
    *,
    start_hdu: int = 1,
    stop_hdu: int | None = None,
    verbose: bool = True,
    add_src_hdu_col: bool = True):
    """
    Loop over all table HDUs in a multi-HDU FITS file, apply selection per HDU,
    and write one stacked output FITS table.

    Parameters:
    infile : str or Path
        Multi-HDU FITS file
    outfile : str or Path
        Output FITS table containing selected rows from all HDUs
    config : DESISelectionConfig
    start_hdu : int
        First HDU index to consider (usually 1)
    stop_hdu : int, optional
        Stop before this HDU index (python slice semantics)
    verbose : bool
        Print per-HDU selection counts
    add_src_hdu_col : bool
        If True, add SRC_HDU column to the output to record origin HDU

    Returns:
    n_selected_total : int
    n_total : int
    """
    _require_astropy()

    infile = str(infile)
    outfile = str(outfile)

    selected_tables: list[Table] = []
    nsel_total = 0
    ntot_total = 0

    with fits.open(infile, memmap=True) as hdul:
        last = len(hdul) if stop_hdu is None else min(int(stop_hdu), len(hdul))

        for hdu in range(int(start_hdu), last):
            h = hdul[hdu]
            if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                continue

            t = Table.read(infile, hdu=hdu)
            ntot_total += len(t)

            mask, tsel = desi_bgs_training_mask(t, config, return_table=True)
            nsel = int(mask.sum())
            nsel_total += nsel

            if verbose:
                print(f"HDU {hdu}: selected {nsel} / {len(t)}")

            if nsel > 0:
                if add_src_hdu_col:
                    tsel = tsel.copy()
                    tsel["SRC_HDU"] = np.full(len(tsel), hdu, dtype=np.int32)
                selected_tables.append(tsel)

    if len(selected_tables) == 0:
        raise RuntimeError("Selection returned 0 rows across all table HDUs; nothing to write.")

    out = vstack(selected_tables, metadata_conflicts="silent")
    out.write(outfile, overwrite=True)

    return int(nsel_total), int(ntot_total)
~                                                
