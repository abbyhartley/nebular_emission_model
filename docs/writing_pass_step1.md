# Writing Pass — ready-to-paste LaTeX (grounded in corrected step-1 results)

Canonical sample = strict per-line S/N (DESI 130,382 / SDSS 57,935).

---
## A. Corrected Table 1 (replace the current \begin{table*} block)

```latex
\begin{table*}[t]
\centering
\caption{In-survey and cross-survey performance for the SDSS-trained and DESI-trained
normalizing flows, evaluated on the per-line S/N$>5$ samples (SDSS: 57{,}935; DESI: 130{,}382).
Point metrics use Monte Carlo averaging ($n_{\rm mc}=50$) and are computed for the log
flux ratios $\log(L_{\rm line}/L_{H\alpha})$; RMSE, bias (median residual), scatter
($\tfrac{1}{2}[p_{84}-p_{16}]$) and NMAD ($1.4826\,\mathrm{MAD}$) are in dex. Spearman
$\rho$ is computed on the log ratios (the model output) rather than on absolute
luminosities, which otherwise inherit a trivial correlation through the shared
conditioning variable $\log L_{H\alpha}$. NLL is the negative log-likelihood per
dimension (bits) of the true 8-D log flux-ratio vector, evaluated on the unnormalized
ratios so that values are comparable across flows; lower is better and values may be
negative for continuous densities.}
\label{tab:flow_metrics}
\begin{tabular}{lcccccc}
\hline
Train $\rightarrow$ Test & NLL (bits/dim) & RMSE (dex) & Bias (dex) & Scatter (dex) & NMAD (dex) & Spearman $\rho$ \\
\hline
SDSS $\rightarrow$ SDSS & $-1.844$ & 0.151 & $-0.001$ & 0.106 & 0.095 & 0.820 \\
SDSS $\rightarrow$ DESI & $-0.661$ & 0.362 & $+0.005$ & 0.292 & 0.268 & 0.380 \\
DESI $\rightarrow$ DESI & $-1.736$ & 0.177 & $-0.002$ & 0.122 & 0.109 & 0.765 \\
DESI $\rightarrow$ SDSS & $-1.352$ & 0.260 & $+0.093$ & 0.168 & 0.158 & 0.515 \\
\hline
\end{tabular}
\end{table*}
```
NOTE: DESI->SDSS bias is +0.093 dex on the strict 130k/58k sample (identical in abs and
ratio space, since residuals are identical). The draft value +0.009 was from the old
loose 733k sample, so it changes along with the rest of the table.

---
## B. Section 4.1 (Summary statistics) — replacement paragraphs

Replace the NLL paragraph:
> On a common (standardization-independent) scale, each flow assigns a higher likelihood
> to its native survey than the foreign flow does: on the SDSS test set the SDSS-trained
> flow reaches NLL $=-1.84$ bits\,dim$^{-1}$ versus $-1.35$ for the DESI-trained flow, and
> on the DESI test set the DESI-trained flow reaches $-1.74$ versus $-0.66$. This ordering
> indicates a genuine cross-survey shift in the (selected, observed-frame) conditional
> density rather than stochastic prediction noise. We note that NLL values normalized per
> flow are \emph{not} comparable across flows, since each flow standardizes its targets
> independently.

Replace the rho sentence (was "0.784 <= rho <= 0.901"):
> Computed on the log flux ratios (the quantity the flow models), Spearman $\rho$ ranges
> from $0.82$ (SDSS in-survey) and $0.76$ (DESI in-survey) down to $0.52$ (DESI$\rightarrow$SDSS)
> and $0.38$ (SDSS$\rightarrow$DESI), with the weakest per-line transfer for [O\,III]\,$\lambda5007$
> (down to $\rho\simeq 0$ cross-survey). These ratio-space correlations are substantially
> lower than the absolute-luminosity values, which are inflated by the shared $\log L_{H\alpha}$
> conditioning. We therefore do not compare our $\rho$ directly to \citet{2024MNRAS.531.1454K},
> whose rank correlations are computed for equivalent widths.

---
## C. Sample selection (rewrite of ll. 249-254)

> After the spectral-type and redshift cuts (\S\,...) the BGS parent of 6{,}445{,}926 objects
> is reduced to 1{,}495{,}353 by the continuum signal-to-noise requirement; imposing
> S/N$>5$ on each of the eight target lines (and H$\alpha$) yields a final BGS sample of
> 130{,}382 galaxies. The same cuts applied to the SDSS main sample (parent 293{,}845)
> give 57{,}935 galaxies. The resulting DESI and SDSS samples have median redshifts of
> 0.116 and 0.072 respectively.

Reconcile cut list: enumerated cut #3 (per-line S/N$>5$ on all eight lines) is what we
apply; ensure the prose funnel above no longer implies a single median-S/N step ending at 733k.

---
## D. Abstract / Conclusions softening (review 2.1)

Abstract (replace ll. ~17-19):
> In cross-survey transfer (SDSS$\leftrightarrow$DESI), performance degrades in a
> line-dependent manner. Likelihood metrics and Balmer-decrement behavior indicate a shift
> in the \emph{selected, observed-frame} conditional distribution between surveys at fixed
> $(M_\star, L_{H\alpha})$, beyond stochastic prediction noise; we caution that this measured
> shift convolves intrinsic differences with survey-dependent selection, aperture,
> redshift-distribution, and measurement-noise effects that we do not fully disentangle.

Conclusions (replace ll. ~499-501): analogous wording -- replace "genuine survey-dependent
shift in the conditional distribution" with "shift in the selected, observed-frame
conditional distribution", and add the same one-sentence list of uncontrolled contributors.

---
## E. Section 2.1 -- justify the per-line S/N cut (new sentence after the cut list)

> Requiring S/N$>5$ on all eight lines is a deliberately conservative cut that preferentially
> retains higher-equivalent-width, star-forming galaxies. Because our target application is the
> generation of emission-line-galaxy mocks -- a population already biased toward star-forming,
> high-luminosity systems -- this selection is well matched to the intended use, at the cost of
> a strong (5.6$\times$ for DESI, 2.6$\times$ for SDSS) reduction relative to a continuum-S/N-only
> selection, and a selection function that differs between surveys (see \S\,results, cross-survey).

---
## F. Fig. 2 Kewley 2001 vs 2006 fix

The "maximum starburst" curve is \citet{2001ApJ...556..121K} (Kewley et al. 2001), NOT
\citet{2006MNRAS.372..961K} (which is the Seyfert/LINER classification work). In the Fig. 2
caption and main text, change the maximum-starburst attribution to Kewley et al. 2001, and
add 2001ApJ...556..121K to refs.bib. Keep \citet{2003MNRAS.346.1055K} (Kauffmann) for the
empirical SF/composite division.

---
## G. Section 3.1 -- Khederlarian positioning paragraph (new)

> The closest prior work is \citet{2024MNRAS.531.1454K}, who model the same eight DESI/BGS
> FastSpecFit lines. Two differences define our contribution. First, they predict a
> deterministic conditional mean and explicitly name normalizing flows as a future extension;
> our flow \emph{is} that density model, directly producing the intrinsic scatter and
> covariance they must approximate by adding noise to point predictions. Second, they condition
> on the continuum shape (11 flux ratios plus a luminosity), whereas we condition on
> $(M_\star, L_{H\alpha})$ alone -- the appropriate inputs for adding lines to halo/SHAM/
> UniverseMachine-type catalogs that provide stellar mass and star formation but no continuum.
> We model the joint 8-D density (covariance by construction) rather than per-line marginals.

---
## H. Citations to add / fix in refs.bib
- Ivezic et al. 2019 (LSST), Spergel et al. 2015 (Roman), Laureijs et al. 2011 (Euclid) -- currently inline \plum{} only.
- Moustakas et al. 2023, FastSpecFit -- ADS 2023ascl.soft08005M.
- BNAF architecture: De Cao, Titov & Aziz 2020 (UAI) -- the flow you actually use; currently only Dinh+2016 (RealNVP) is cited.
- Fix malformed MPA-JHU entry -> cite Brinchmann et al. 2004, Kauffmann et al. 2003, Tremonti et al. 2004.
- Kewley et al. 2001 -- 2001ApJ...556..121K (Fig. 2 fix above).
- Optional positioning: Alsing+2020 (Speculator, architecture only), Alsing+2024 / Thorp+2025 (pop-cosmos), Fischbacher+2025 / Tortorelli+2025 (GalSBI).
