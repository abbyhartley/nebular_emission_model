# Red-Team Review — Triage & Response Plan

Response to the Claude-generated red-team review of *An Empirical Model for Galaxy
Nebular Emission* (Hartley, Cooray & Wechsler). Triage split into writing-only vs
coding/analysis. Priorities: **Must** (affects headline claims/numbers), **Should**,
**Optional/Future**.

## Two critiques CONFIRMED against the code
- **§2.5 Spearman rho is inflated.** In `scripts/eval_nf_summarystats_NLL_*.py`, `rho`
  is computed on *absolute* log-luminosities (`ratio + logLHa`). Because `logLHa` is a
  conditioning input shared by prediction and truth, rho is dominated by the trivial
  L_Ha-L_Ha correlation (e.g. Hbeta rho=0.988). Must report rho/RMSE on the **ratios**
  (the actual model output). Also invalidates the Khederlarian rho comparison (theirs is
  on EWs).
- **§2.5 NLL not on a common scale across flows.** `nll_bits_per_dim` normalizes with
  each flow's OWN `meta["X_mean"]/["X_std"]`, so SDSS-flow and DESI-flow bits/dim are not
  comparable. Fix: evaluate density on the raw (unnormalized) ratio by adding the
  standardization log-Jacobian `sum(log X_std)`, OR state the caveat explicitly.

## Writing-only (no code)
| # | Item | Priority | How to address |
|---|------|----------|----------------|
| 2.1 | Soften "genuine distribution shift at fixed (M*,L_Ha)" -> "shift in the selected, observed-frame conditional distribution" | Must | Reword abstract ll.17-19 + conclusions ll.499-501; list uncontrolled contributors (selection, aperture, z-mix, M* zero-point, AGN fraction). |
| 3.1 | Real positioning paragraph vs Khederlarian et al. 2024 (closest prior work) | Must | density-vs-point-estimate, joint-vs-marginal; (M*,L_Ha) is the right conditioning for halo/SHAM/UM mocks lacking continuum. Cleanest novelty statement. |
| 4 | Fig.2 mislabel: Kewley 2001 vs 2006 (CONFIRMED in .tex) | Must | "Maximum starburst" curve = Kewley et al. 2001 (ApJ 556,121); text/caption/bib currently cite 2006MNRAS.372..961K (Seyfert/LINER work). Fix all three. |
| 4 | Missing references | Must | Add Ivezic+2019(LSST), Spergel+2015(Roman), Laureijs+2011(Euclid), Moustakas+2023 FastSpecFit (2023ascl.soft08005M), BNAF De Cao/Titov/Aziz 2020. Fix malformed MPA-JHU cite -> Brinchmann+2004/Kauffmann+2003/Tremonti+2004. |
| 3.4 | Cite the architecture actually used (BNAF), not just Dinh+2017 (RealNVP) | Must | Cite BNAF; state use of block_neural_autoregressive_flow. |
| 4 | Fig.1 caption still has \plum{} placeholder; Figs 3/4 lack takeaways | Should | Replace placeholder with (a)-(d) descriptions; add one-line takeaways. |
| 2.9 | Conditioning on OBSERVED L_Ha undercuts intrinsic->observed forward-model framing | Should | Add scope-boundary paragraph: model is observed-frame; mock use needs separate SFR->dust-attenuated-fiber-Ha step. |
| 2.7 | Learned density is noise-convolved; engage Khederlarian noise-injection result | Should | State scope (observed-frame; differential S/N contributes to shift). Full noise modeling -> future work. |
| 3.5 | Intro critiques single instantaneous tracers but model conditions on instantaneous L_Ha + M* | Should | Soften ll.72-78 or acknowledge M* adds only integrated-SFH info; recent-burst sensitivity still absent. |
| 3.2 | Frame Speculator (Alsing+2020) as architectural precedent only (no line accuracy) | Optional | One sentence in future-work. |
| 3.3 | Cite pop-cosmos & GalSBI in forward-model framing | Optional | Alsing+2024, Thorp+2025, Fischbacher+2025, Tortorelli+2025. |
| 2.3 | Aperture systematics | Optional | Acknowledge + cite Kewley, Jansen & Geller 2005. Physical apertures similar in median (~4 kpc) so figure optional. |
| 4 | Abstract has no numbers; acknowledgments placeholder | Should | Add 2-4 quantified anchors; real grant numbers + DESI/SDSS required-ack text + data/code availability (repo now public). |

## Coding / analysis (run on Sherlock via sbatch)
| # | Item | Priority | Effort | Plan |
|---|------|----------|--------|------|
| 2.5a | Report rho and RMSE on the RATIOS, not absolute luminosities | Must | Small | Compute metrics on (logL_pred - logLHa) vs (logL_true - logLHa). Add ratio columns to CSV. |
| 2.5b | Reconcile/standardize NLL across flows + add robust stats | Must | Small-Med | Evaluate density on raw ratios (add sum(log X_std)) for a common scale; add NMAD + median residual alongside RMSE. |
| 2.10 | In-survey Balmer-decrement floor rate (baseline for cross-survey 15.5/10.7%) | Must | Small | Extend check_balmer_dec_from_flows.py to also report in-survey sub-2.86 fraction. |
| 2.6 | Report M/L-color scatter + M* zero-point sensitivity test | High | Small/Med | Scatter of recalibrated relation; perturb M* zero-point and re-measure cross-survey shift (tests 2.1). |
| 2.8 | Doublet-ratio physical-bounds check on flow samples | High | Small | Check generated [OII]3729/3726 in ~0.35-1.5 and [SII]6717/6731 in ~0.44-1.45. Tests "split-OII heavy tails" hypothesis. |
| 2.4/2.1 | [NII]/Ha and [OIII]/Hb offset vs redshift (decomposition test) | High | Med | Bin SDSS<->DESI residuals by z; show whether the shift tracks z-mismatch. |
| 4 | Add Balmer-decrement distribution figure (in/cross, 2.86 floor) | Should | Small | Use existing numbers + plotstyle. |
| 2.5c | Diagnose [OII]/[SII] heavy tails: astrophysics vs catastrophic measurements | Should | Med | Cross-tab large-residual objects vs S/N and DESI pipeline flags. |
| 3.4 | Hyperparameter table for appendix | Should | Trivial | BNAF (pin FlowJax layer/block defaults), Adam, lr=3e-4, clip-global-norm=1.0, batch=2048, epochs=200, seed=0, per-feature standardization; note autoregressive variable ordering. |
| 2.2/2.7 | Excitation-class mixture / noise-injection layer | Optional/Future | Large | Keep as future-work unless quick test shows in-survey impact. |

## Pushback (avoid over-correcting)
- 2.8 full [OII] re-modeling and 2.2 in-survey excitation conditioning: treat as
  optional/future. Do the cheap doublet-ratio CHECK; address rest in discussion.
- 2.3 aperture figure: citation + sentence likely sufficient (reviewer concedes apertures
  are similar in the median).

## Suggested sequence
1. Cheap code fixes that change the tables (2.5a, 2.5b, 2.10) — do BEFORE finalizing prose.
2. One decomposition test (2.4 offsets-vs-z) + M* zero-point test (2.6).
3. Writing pass (framing, Khederlarian paragraph, citations, Fig.2 fix, housekeeping).
