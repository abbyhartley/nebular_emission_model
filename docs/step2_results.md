# Step 2 results — decomposing the cross-survey shift (review 2.4 / 2.1 / 2.6)

Canonical sample = strict per-line S/N (DESI 130,382 / SDSS 57,935).

## Test A — redshift decomposition (review 2.4)
Control box (common to both surveys): logM* in [9.60,10.00], logL_Ha in [40.56,41.06].
In box: SDSS 7,568; DESI 3,513.

Within-survey ratio vs z (at fixed M*, L_Ha) -- ratios DO evolve with z:
  SDSS:  d log([NII]/Ha)/dz = -1.61 ;  d log([OIII]/Hb)/dz = +1.28
  DESI:  d log([NII]/Ha)/dz = -1.18 ;  d log([OIII]/Hb)/dz = +2.10

SDSS - DESI offset at matched (M*, L_Ha):
  [NII]/Ha : full-box -0.174 dex ; z-overlap[0.07,0.10] -0.193 dex
  [OIII]/Hb: full-box +0.126 dex ; z-overlap[0.07,0.10] +0.151 dex

=> Ratios evolve with z, but the SDSS-DESI offset does NOT shrink when restricted to
matched redshift (it is slightly larger). The cross-survey shift is therefore a genuine
survey-dependent effect, NOT an artifact of the different z-distributions. The curves run
parallel and vertically offset (see fig).
Figure: /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/plotting/zdecomp_ratio_vs_z.png

## Test B — M* zero-point sensitivity (review 2.6)
Add a constant delta to the TEST survey's log M* before conditioning; sweep delta.

In-survey controls peak exactly at delta=0 (validates the method):
  SDSS->SDSS best delta = 0.00 ;  DESI->DESI best delta = 0.00

Cross-survey: transfer improves MONOTONICALLY out to the +/-0.60 grid edge (no minimum
in the physically plausible range):
  SDSS->DESI:  delta 0.00 -> +0.60 :  NLL_raw -0.661 -> -1.222 ;  RMSE 0.361 -> 0.207 ;  rho 0.380 -> 0.704
  DESI->SDSS:  delta 0.00 -> -0.60 :  NLL_raw -1.352 -> -1.585 ;  RMSE 0.259 -> 0.185 ;  rho 0.516 -> 0.758
  (at a plausible |delta|=0.15: SDSS->DESI NLL -0.661->-0.809 ; DESI->SDSS -1.352->-1.427)

Interpretation (IMPORTANT - state carefully): the cross-survey offset lies along the
mass-metallicity direction in ratio space, so a single M* shift coherently raises [NII]/Ha
AND lowers [OIII]/Hb -- exactly the direction needed to match the other survey. Hence the
shift is strongly DEGENERATE with the survey-relative M* zero-point. This is NOT evidence
that masses are wrong by 0.6 dex (that would be a 4x mass error); rather, the data cannot
cleanly separate an intrinsic shift from a stellar-mass calibration offset. The "shift at
fixed M*" claim is fragile: M* is not consistently 'fixed' across surveys.

## Net synthesis
- The cross-survey shift is real (survives matched-z; in-survey NLL/floor are clean).
- BUT its interpretation as intrinsic astrophysics is undercut by a strong degeneracy with
  the cross-survey color-mass zero-point (review 2.1 item 3, 2.6). 
- Action: report the M/L-color relation scatter (TODO: compute from calibration sample),
  soften any 'intrinsic' language, and add the M* zero-point degeneracy as a key caveat.

## Writing-ready sentences
Results (new paragraph): "Restricting to a common (M*, L_Ha) control box, the observed
[NII]/Ha and [OIII]/Hb ratios evolve with redshift within both surveys, yet the SDSS-DESI
offset persists (and slightly grows) when matched in redshift, indicating the cross-survey
shift is not an artifact of the differing redshift distributions."
Discussion (new caveat): "The cross-survey shift is strongly degenerate with the
survey-relative stellar-mass zero-point: applying a constant offset to the test-survey
log M* improves transfer monotonically (e.g. SDSS->DESI rho 0.38->0.70 at +0.6 dex), while
in-survey performance is maximized at zero offset. Because the offset lies along the
mass-metallicity direction, we cannot cleanly separate an intrinsic survey shift from a
color-mass calibration difference; we therefore interpret the shift as an upper bound on
intrinsic survey differences at fixed (M*, L_Ha)."

## TODO (remaining 2.6 item)
Compute and report the scatter of the recalibrated log(M/L_r)-color relation (~0.1-0.15 dex
expected) from the MPA-JHU calibration sample; the relation coefficients (1.062, -0.555) are
hard-coded in src/normflow/stellar_mass.py but the fit residual scatter is not yet reported.
