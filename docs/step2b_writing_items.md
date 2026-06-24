# Step 2b — writing items 2.3 (aperture), 2.7 (noise), 2.8 (doublet check)

## 2.8 Doublet-ratio physical-bounds check (scripts/check_doublet_ratios.py)
Density-sensitive doublets bounded by atomic physics (Osterbrock & Ferland 2006, T_e~1e4 K):
  [S II] 6716/6731 in [0.44 (high-n_e), 1.45 (low-n_e)]
  [O II] 3729/3726 in [0.35 (high-n_e), 1.47 (low-n_e)]
~1e5 samples per flow. Fraction OUTSIDE band (csv: docs/doublet_ratio_bounds.csv):

[S II] 6716/6731   (frac<0.44 / frac>1.45)
  Observed SDSS      0.00% / 31.99%
  Observed DESI      0.01% / 21.59%
  NF SDSS->SDSS      0.02% / 33.42%
  NF DESI->DESI      0.05% / 22.28%
  NF SDSS->DESI      0.07% / 24.86%
  NF DESI->SDSS      0.09% / 21.51%

[O II] 3729/3726   (frac<0.35 / frac>1.47)
  Observed SDSS      0.08% / 12.58%
  Observed DESI      0.00% / 20.70%
  NF SDSS->SDSS      0.12% / 13.09%
  NF DESI->DESI      0.02% / 20.67%
  NF SDSS->DESI      0.23% /  8.29%
  NF DESI->SDSS      0.13% / 20.37%

KEY TAKEAWAYS:
- High-density limit essentially never violated (<0.25% everywhere): no spurious high-n_e tail.
- Out-of-band is almost entirely on the LOW-density side (ratio above upper limit), i.e.
  measurement noise on faint doublet components scattering past the soft n_e->0 limit.
- In-survey NF reproduces the OBSERVED out-of-band fraction to within ~1%
  (OII: 13.1 vs 12.6 SDSS; 20.7 vs 20.7 DESI). So the "split-OII heavy tails" are in the
  DATA (noise), not a flow artifact. The flow faithfully reproduces the noise-convolved
  doublet distribution and does not invent unphysical density structure.
- Cross-survey fractions track the documented domain shift (e.g. SDSS->DESI OII 8.3% vs
  DESI data 20.7%) but show no catastrophic unphysical pileup.

## Writing-ready LaTeX

### 2.8 -> new paragraph at end of Results sec:ratio_comparisons
As a physical-consistency check we examine the two density-sensitive doublets,
[S\,II]\,$\lambda\lambda6716,6731$ and [O\,II]\,$\lambda\lambda3729,3726$, whose intrinsic
ratios are bounded by the low- and high-electron-density limits set by atomic physics
([S\,II]\,$6716/6731\in[0.44,1.45]$ and [O\,II]\,$3729/3726\in[0.35,1.47]$ at
$T_e\sim10^4$\,K; \citealt{2006agna.book.....O}). Drawing $\sim\!10^5$ samples from each
flow, we find that the in-survey flows reproduce the observed out-of-band fractions to
within $\sim\!1\%$: for [O\,II] the fraction exceeding the low-density limit is $13.1\%$
(flow) versus $12.6\%$ (data) for SDSS and $20.7\%$ versus $20.7\%$ for DESI, and for
[S\,II] it is $33\%$ versus $32\%$ (SDSS) and $22\%$ versus $22\%$ (DESI); the high-density
limit is essentially never violated ($<0.25\%$ in all cases). These out-of-band fractions
are inherited from measurement noise on the faint individual doublet components in the
training data---they are present in the observations themselves---rather than being an
artifact of the flow, which reproduces the noise-convolved doublet distribution. The heavy
tails in the split [O\,II] components therefore do not reflect unphysical model behavior.

### 2.7 + 2.3 -> new Discussion subsection "Caveats and scope" (before Future work)
Our flows learn the \emph{noise-convolved} distribution of observed line ratios rather
than the intrinsic, error-free distribution: the training targets are measured fluxes, so
the learned density already incorporates measurement uncertainty. This is directly visible
in the density-sensitive doublets (\S\ref{sec:ratio_comparisons}), whose tails beyond the
atomic-physics limits are reproduced because they are present in the noisy data. A practical
consequence is that part of the measured cross-survey shift may arise from the differing
per-line signal-to-noise of the two surveys rather than from intrinsic astrophysics.
\citet{2024MNRAS.531.1454K} address this by injecting observational noise into their
predictions to match the observed scatter; an explicit, survey-specific noise model that
deconvolves the intrinsic density from the measurement kernel is a natural extension of our
framework, which we defer to future work.

A further uncontrolled contributor is the differing spectroscopic aperture: SDSS uses
$3''$ fibers whereas DESI uses $1.5''$ fibers, so at fixed redshift the two surveys sample
different physical scales and, through radial gradients in ionization and metallicity,
potentially different line ratios \citep[e.g.][]{2005PASP..117..227K}. Because DESI's
smaller fibers are partly compensated by its higher median redshift, the median physical
aperture is comparable between the two selected samples ($\sim$ a few kpc); we therefore
expect aperture differences to be a sub-dominant, though non-zero, contributor to the
measured cross-survey shift.

## New bib keys needed
- 2006agna.book.....O  (Osterbrock & Ferland 2006, AGN2)
- 2005PASP..117..227K  (Kewley, Jansen & Geller 2005, aperture effects)
