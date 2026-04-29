"""
Sample selection utilities for DESI BGS / SDSS (MPA-JHU / RCSED2-like) tables.

Features
--------
- Versioned selection configs (choose different cut-sets by name)
- Survey-aware column mapping (DESI vs SDSS) with sensible defaults
- Works on:
    * a single astropy Table
    * a single-HDU FITS (table in HDU 1, typically)
    * a multi-HDU FITS (loops over all table HDUs)
- Writes one stacked output FITS table, optionally with provenance column SRC_HDU

Notes
-----
- The "survey" determines which columns are used for Z, ZWARN/Z_WARNING, SNR, and line fluxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union, Optional

import numpy as np

try:
    from astropy.table import Table, vstack
    from astropy.io import fits
except ImportError:
    Table = None
    vstack = None
    fits = None


Survey = Literal["desi", "sdss"]
SelectionVersion = Literal["v1"]


# ---- Default line flux column names per survey ----

DESI_LINE_FLUX_COLS = ("OII_3726_FLUX",
    "OII_3729_FLUX",
    "HGAMMA_FLUX",
    "HBETA_FLUX",
    "HALPHA_FLUX",
    "OIII_5007_FLUX",
    "NII_6584_FLUX",
    "SII_6716_FLUX",
    "SII_6731_FLUX")

SDSS_LINE_FLUX_COLS = ("OII_3726_FLUX",
    "OII_3729_FLUX",
    "H_GAMMA_FLUX",
    "H_BETA_FLUX",
    "H_ALPHA_FLUX",
    "OIII_5007_FLUX",
    "NII_6584_FLUX",
    "SII_6717_FLUX",
    "SII_6731_FLUX")


# ---- Column mapping per survey ----

SURVEY_COLMAP = {"desi": dict(z_col="Z",
        zwarn_col="ZWARN",
        snr_col="SNR_R",
        spectype_col="SPECTYPE",
        line_flux_cols=DESI_LINE_FLUX_COLS,
        # convenience (not used in cuts directly, but helpful)
        logm_col="LOGMSTAR",
        ha_flux_col="HALPHA_FLUX"),
    "sdss": dict(
        z_col="Z_1",
        zwarn_col="Z_WARNING",
        snr_col="SN_MEDIAN",   
        spectype_col="SPECTROTYPE", 
        line_flux_cols=SDSS_LINE_FLUX_COLS,
        logm_col="LGM_TOT_P50",
        ha_flux_col="H_ALPHA_FLUX")}


@dataclass(frozen=True)
class SelectionConfig:
    # which column mapping to use
    survey: Survey = "desi"

    # cut thresholds
    z_min: float = 0.05
    snr_min: float = 5.0
    require_zwarn0: bool = True # only use sources w/ trustworthy redshifts

    # Optional spectype cut (applied only if column exists and require_spectype is not None)
    require_spectype: Optional[str] = "GALAXY"

    # Column overrides (if None, use survey defaults)
    z_col: Optional[str] = None
    zwarn_col: Optional[str] = None
    snr_col: Optional[str] = None
    spectype_col: Optional[str] = None
    line_flux_cols: Optional[tuple[str, ...]] = None


def _require_astropy():
    if Table is None or fits is None or vstack is None:
        raise ImportError("astropy is required (pip install astropy).")


def _resolved_config(cfg: SelectionConfig) -> dict:
    """Resolve survey defaults + user overrides into a concrete mapping."""
    if cfg.survey not in SURVEY_COLMAP:
        raise ValueError(f"Unknown survey={cfg.survey!r}. Choose from {list(SURVEY_COLMAP)}")

    base = dict(SURVEY_COLMAP[cfg.survey])
    base["z_col"] = cfg.z_col or base["z_col"]
    base["zwarn_col"] = cfg.zwarn_col or base["zwarn_col"]
    base["snr_col"] = cfg.snr_col or base["snr_col"]
    base["spectype_col"] = cfg.spectype_col or base["spectype_col"]
    base["line_flux_cols"] = cfg.line_flux_cols or base["line_flux_cols"]

    return base


def get_selection_config(version: SelectionVersion = "v1", *, survey: Survey = "desi") -> SelectionConfig:
    """
    Versioned selection configs.

    Extend this over time: add 'v2', 'v3', etc. with different thresholds/cuts.
    """
    if version == "v1":
        return SelectionConfig(survey=survey,
            z_min=0.05,
            snr_min=5.0,
            require_zwarn0=True,
            require_spectype="GALAXY")
    raise ValueError(f"Unknown version={version!r}")


def _iter_table_hdus(filename: Union[str, Path]):
    """
    Yield (hdu_index, Table) for each table HDU in a FITS file.

    Works for both single-HDU-table files and multi-HDU-table files.
    """
    _require_astropy()
    filename = str(filename)

    with fits.open(filename, memmap=True) as hdul:
        for hdu in range(len(hdul)):
            h = hdul[hdu]
            if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                continue
            t = Table.read(filename, hdu=hdu)
            yield hdu, t


def training_mask(table: "Table",
    config: SelectionConfig,
    *,
    return_table: bool = False):
    """
    Build a boolean mask for a single astropy Table using survey-aware columns.
    """
    _require_astropy()
    col = _resolved_config(config)

    # Required columns
    required = [col["z_col"], col["snr_col"]] + list(col["line_flux_cols"])
    if config.require_zwarn0:
        required.append(col["zwarn_col"])

    missing = [c for c in required if c not in table.colnames]
    if missing:
        raise KeyError(f"Missing required columns for survey={config.survey}: {missing}")

    mask = np.ones(len(table), dtype=bool)

    # spectype cut
    spec_col = col["spectype_col"]
    if config.require_spectype is not None and spec_col in table.colnames:
        spectype = np.asarray(table[spec_col]).astype(str)
        mask &= (spectype == config.require_spectype)

    # Z cut
    z = np.asarray(table[col["z_col"]])
    mask &= np.isfinite(z) & (z > config.z_min)

    # ZWARN/Z_WARNING cut
    if config.require_zwarn0:
        zw = np.asarray(table[col["zwarn_col"]])
        mask &= np.isfinite(zw) & (zw == 0)

    # SNR cut
    snr = np.asarray(table[col["snr_col"]])
    mask &= np.isfinite(snr) & (snr > config.snr_min)

    # Line flux cut: require only H-alpha > 0
    ha_col = col.get("ha_flux_col", None)
    if ha_col is None or ha_col not in table.colnames:
        raise KeyError(f"Missing required H-alpha flux column for survey={config.survey}: {ha_col!r}")

    ha = np.asarray(table[ha_col])
    mask &= np.isfinite(ha) & (ha > 0)

    if return_table:
        return mask, table[mask]
    return mask


def write_filtered_fits_any(infile: Union[str, Path],
    outfile: Union[str, Path],
    config: SelectionConfig,
    *,
    verbose: bool = True,
    add_src_hdu_col: bool = True):
    """
    Apply selection to *all* table HDUs found in infile (1 or many), stack selected rows,
    and write to outfile.

    Returns
    -------
    n_selected_total, n_total
    """
    _require_astropy()

    selected_tables: list[Table] = []
    nsel_total = 0
    ntot_total = 0

    for hdu, t in _iter_table_hdus(infile):
        ntot_total += len(t)
        mask, tsel = training_mask(t, config, return_table=True)
        nsel = int(mask.sum())
        nsel_total += nsel

        if verbose:
            print(f"HDU {hdu}: selected {nsel} / {len(t)}")

        if nsel > 0:
            if add_src_hdu_col:
                tsel = tsel.copy()
                tsel["SRC_HDU"] = np.full(len(tsel), int(hdu), dtype=np.int32)
            selected_tables.append(tsel)

    if len(selected_tables) == 0:
        raise RuntimeError("Selection returned 0 rows across all table HDUs; nothing to write.")

    out = vstack(selected_tables, metadata_conflicts="silent")
    out.write(str(outfile), overwrite=True)

    return int(nsel_total), int(ntot_total)
