"""Run tengri's convert_cue_weights.py under NumPy-2 compat shims, since the
upstream (numpy<2-era) `cue` package uses removed aliases like np.in1d."""
import numpy as np
# restore removed NumPy 2.0 aliases used by upstream cue
if not hasattr(np, "in1d"): np.in1d = np.isin
if not hasattr(np, "row_stack"): np.row_stack = np.vstack
for nm, val in dict(float_=np.float64, int_=np.int64, complex_=np.complex128,
                    unicode_=np.str_, bool=bool, object=object, str=str, int=int).items():
    if not hasattr(np, nm):
        try:
            setattr(np, nm, val)
        except Exception:
            pass
import runpy, sys
CUE = "/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/cue_repo/src/cue/data"
sys.argv = ["convert_cue_weights.py", "--cue-dir", CUE, "--output", "data/cue_weights.npz"]
runpy.run_path("scripts/convert_cue_weights.py", run_name="__main__")
