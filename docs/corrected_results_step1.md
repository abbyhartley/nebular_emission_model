# Corrected Results — Step 1 (canonical = strict per-line S/N sample)

Canonical sample (decision 2026-06-24): STRICT per-line S/N>5 on all 9 lines.
On-disk training/eval files = DESI 130,382 / SDSS 57,935 galaxies.
Current flows (nf_desi_bgs.eqx, nf_sdss_main.eqx) already match this sample.

## Corrected Table 1 (replaces draft Table 1; all values on the 130k/58k sample)

Point metrics (RMSE/bias/scatter/NMAD) are identical in abs and ratio space (the
log L_Ha term cancels in the residual). Spearman rho is reported on the RATIOS
(review 2.5a). NLL_raw is the standardization-independent density on raw log-ratios
and is the only cross-flow-comparable NLL (review 2.5b); NLL_norm (per-flow
standardization, old style) is shown only for continuity and is NOT comparable across flows.

| Train -> Test | NLL_raw (bits/dim, common) | NLL_norm (per-flow) | RMSE (dex) | Bias (dex) | Scatter (dex) | NMAD (dex) | Spearman rho (ratio) |
|---|---|---|---|---|---|---|---|
| SDSS -> SDSS | -1.844 | 0.859 | 0.151 | -0.001 | 0.106 | 0.095 | 0.820 |
| SDSS -> DESI | -0.661 | 2.042 | 0.362 | +0.005 | 0.292 | 0.268 | 0.380 |
| DESI -> DESI | -1.736 | 0.925 | 0.177 | -0.002 | 0.122 | 0.109 | 0.765 |
| DESI -> SDSS | -1.352 | 1.309 | 0.260 | +0.093 | 0.168 | 0.158 | 0.515 |

Key changes vs the draft table:
- rho drops sharply on ratios (esp. cross-survey: SDSS->DESI 0.79 -> 0.38; DESI->SDSS 0.81 -> 0.52).
- On the common scale, each NATIVE flow beats the FOREIGN one on its own test set
  (SDSS test: -1.844 < -1.352; DESI test: -1.736 < -0.661) -- resolving the draft
  table's paradox where the foreign flow looked "better".

## Balmer-decrement floor frac(R<2.86) [review 2.10]
| Distribution | frac(R<2.86) |
|---|---|
| Observed SDSS / DESI | 0.65% / 2.6% |
| NF in-survey SDSS->SDSS / DESI->DESI | 0.9% / 2.9% |
| NF cross SDSS->DESI / DESI->SDSS | 15.5% / 10.7% |
In-survey NF rate ~ observed (1-3%); cross-survey excess is genuine (~5-15x).

## Corrected sample-selection numbers (rewrite of paper ll.249-254)
DESI BGS parent 6,445,926 -> global S/N>5 1,495,353 -> per-line S/N>5 (all 9) + finite color-mass: 130,382. Median z = 0.116 (was 0.16).
SDSS main parent 293,845 -> global S/N>5 210,092 -> per-line S/N>5 (all 9): 57,935. Median z = 0.072 (was 0.08).

## Open issues to address (not blocking)
- 2.1 justification: the per-line S/N>5 on all 9 lines is severe (DESI 5.6x, SDSS 2.6x
  reduction vs loose). Must justify or soften (it preferentially keeps high-EW/SF galaxies).
- SPECTYPE/GALAXY cut: the DESI parent fastspec_zall_combined.fits has NO 'SPECTYPE'
  column, so selection.py SILENTLY SKIPS the GALAXY cut for DESI (paper cut #1).
  SDSS SPECTROTYPE is present and applied. Need to confirm DESI galaxy purity another way.
