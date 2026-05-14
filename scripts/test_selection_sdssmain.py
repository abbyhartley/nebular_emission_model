import sys
from pathlib import Path

REPO = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model").resolve()
sys.path.insert(0, str(REPO / "src"))   # <-- critical

from normflow.selection import get_selection_config, write_filtered_fits_any

infile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo.fits")
outfile = Path("/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/mpa_rcsed2_combo_selected_test.fits")

cfg = get_selection_config("v1", survey="sdss")

nsel, ntot = write_filtered_fits_any(
    infile,
    outfile,
    cfg,
    verbose=True,
)

print(f"TOTAL selected {nsel} / {ntot}")
print(f"Wrote: {outfile}")
