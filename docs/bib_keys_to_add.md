# New \cite keys introduced during the review-response edits

These keys are referenced in the updated manuscript and must exist in refs.bib
before compiling (otherwise they render as bold "??"). Grouped by the pass that
added them. All optional pop-cosmos/GalSBI bibcodes are now RESOLVED.

## Step 1 (metrics + framing fixes)
- `2001ApJ...556..121K`  — Kewley et al. 2001 (ApJ 556, 121): the "maximum starburst"
  BPT demarcation. Replaces the prior mis-citation of 2006MNRAS.372..961K.
- `2023ascl.soft08005M` — Moustakas et al. 2023, FastSpecFit (ASCL). DESI line catalog.
- `decao2020bnaf`        — De Cao, Titov & Aziz 2020 (UAI), Block Neural Autoregressive
  Flow — the architecture actually used. (mnemonic key; set to the real bibcode if preferred.)
- `MPAJHU`               — confirm this key is defined in refs.bib (used in the SDSS
  section + data-availability). Brinchmann+2004 / Kauffmann+2003 / Tremonti+2004 are
  already cited via 2004MNRAS.351.1151B / 2003MNRAS.341...33K / 2004ApJ...613..898T.

## Step 2b (doublet check + aperture)
- `2006agna.book.....O`  — Osterbrock & Ferland 2006, AGN^2 (doublet density limits).
- `2005PASP..117..227K`  — Kewley, Jansen & Geller 2005 (PASP 117, 227): aperture effects.

## Step 2c (writing cluster)
- `2019ApJ...873..111I`  — Ivezic et al. 2019 (Rubin/LSST).
- `2015arXiv150303757S`  — Spergel et al. 2015 (Roman/WFIRST).
- `2011arXiv1110.3193L`  — Laureijs et al. 2011 (Euclid).
- `2020ApJS..249....5A`  — Alsing et al. 2020, Speculator.

## Step 2c optional (pop-cosmos / GalSBI) — RESOLVED bibcodes
- `alsing2024popcosmos`     -> bibcode 2024ApJS..274...12A ; arXiv 2402.00935
  Alsing et al. 2024, ApJS 274, 12, "pop-cosmos: A Comprehensive Picture of the Galaxy
  Population from COSMOS Data".
- `thorp2024popcosmos`      -> bibcode 2024ApJ...975..145T ; arXiv 2406.19437
  Thorp et al. 2024, ApJ 975, 145, "pop-cosmos: Scaleable Inference of Galaxy Properties
  and Redshifts with a Data-driven Population Model".
- `fischbacher2024galsbi`   -> bibcode 2024arXiv241208701F ; arXiv 2412.08701
  Fischbacher, Kacprzak, Tortorelli et al. 2024, "GalSBI: Phenomenological galaxy
  population model for cosmology using simulation-based inference".
  (NOTE: `tortorelli2024galsbi` was DROPPED -- Tortorelli is a co-author on the Fischbacher
  GalSBI paper, not a separate reference; the Conclusions now cite GalSBI with the single
  Fischbacher key.)

## Remaining \plum to fill by hand (cannot be auto-resolved)
- Acknowledgments: real NSF GRFP grant number + Stanford Graduate Fellowship wording.
- Full required DESI and SDSS collaboration acknowledgment boilerplate.
- \software line: NumPy / SciPy / JAX citations (optional).
