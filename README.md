# An Empirical Model for Galaxy Nebular Emission

Conditional normalizing flows that learn the **joint distribution of eight bright optical
emission line ratios** given two global galaxy properties: stellar mass and Hα luminosity.

Most nebular emission models predict one line at a time, or return a single deterministic
line ratio per galaxy. This model returns the full 8-dimensional conditional density
`p(x | M*, L_Hα)`, so the line-to-line covariance and the intrinsic scatter come out of the
model rather than being bolted on afterwards. That makes it usable both for **evaluating a
likelihood** on observed spectra and for **painting realistic emission lines onto mock
catalogs** that supply stellar masses and star formation rates but no spectra
(halo-occupation, SHAM, UniverseMachine).

Two flows are provided, trained separately on **84,651 SDSS Main Galaxy Sample** and
**501,751 DESI Bright Galaxy Survey** galaxies. Once both surveys are placed on a common
stellar-mass and flux scale, either flow predicts the other survey's line ratios nearly as
well as its own, so the learned density is close to survey-independent.

**Modeled lines** (targets are `log10(L_line / L_Hα)`):
Hβ, Hγ, [N II] λ6584, [S II] λ6717, [S II] λ6731, [O II] λ3726, [O II] λ3729, [O III] λ5007.

## Install

```bash
git clone https://github.com/abbyhartley/nebular_emission_model.git
cd nebular_emission_model
pip install -e ".[plots]"
```

The published models were produced with `flowjax` 17.2.1 and `jax` 0.5.2. FlowJAX changes
its API across major versions, so the pin `flowjax>=17.2,<18` matters for loading the
supplied weights.

## Quickstart

```python
import numpy as np
from normflow.io import load_flow, sample_line_ratios, line_names

flow, meta = load_flow("desi")          # or "sdss"

logmstar = np.array([ 9.5, 10.0, 10.5])  # log10(M*/Msun)
loglha   = np.array([40.8, 41.0, 41.2])  # log10(L_Halpha / erg s^-1)

x = sample_line_ratios(flow, meta, logmstar, loglha, seed=0)   # (3, 8)
print(line_names(meta))
print(x)                                 # log10(L_line / L_Halpha)
```

To recover absolute line luminosities, add the conditioning Hα luminosity back:

```python
log_L_line = loglha[:, None] + x
```

`normflow.io.log_prob(flow, meta, x, logmstar, loglha)` evaluates the conditional density of
observed ratios, which is what the cross-survey comparisons in the paper use.

Generating 10⁶ galaxies takes roughly 41 s on a single CPU core — about three orders of
magnitude faster than an equivalent FSPS+Cloudy forward model using precomputed
photoionization tables.

## Scope — please read before using

The flows learn the **noise-convolved distribution of observed line ratios** for an
**emission-line-selected** population. Specifically:

- The training samples require S/N > 3 in the continuum and in *all eight* lines, so the
  learned density is `p(x | M*, L_Hα, all eight lines detected)`. It should be applied to
  emission-line-selected populations like its training set, **not** to a complete galaxy
  sample, and not to weak-line, low-sSFR, or quiescent systems.
- The density includes each survey's measurement uncertainty. For generating synthetic
  catalogs that resemble what a survey observes, this is the right object; it is not the
  intrinsic, error-free density.
- Conditioning is on **observed** Hα luminosity, not intrinsic SFR. Painting lines onto a
  simulation requires a survey-specific map from intrinsic star formation to the
  dust-attenuated, aperture-limited Hα luminosity that survey would measure. That map is
  deliberately outside the model, which is what keeps the model survey-agnostic.
- Validated to z ≈ 1 against DESI–COSMOS with essentially no bias in the Balmer lines and
  [O III]/Hβ. The one known systematic is the [O II] doublet, increasingly under-predicted
  with redshift (≈0.04 dex at z ~ 0.1 to ≈0.10 dex at z ~ 1) because the ionized ISM evolves
  at fixed (M*, L_Hα).

## Repository layout

```
models/            trained flows used in the paper (nf_{sdss,desi}_ALTB.eqx + _meta.pkl)
src/normflow/      importable library
  io.py              load a flow, sample, evaluate log-density
  train_NF.py        train_line_ratio_flow(): trains a flow from a dataframe
  selection.py       sample selection cuts
  stellar_mass.py    colour-derived stellar masses
scripts/
  data_prep/         catalog prep, crossmatch, flux calibration, colour masses, selection
  training/          training entry points + Slurm wrappers
  evaluation/        metrics, NLL, Balmer decrement, doublet limits, PIT, covariance
  figures/           every script that makes a paper figure
  highz/             Appendix C: DESI–COSMOS redshift extrapolation
  extras/            UniverseMachine comparison, Byler/Cue photoionization grids
results/           evaluation tables (CSV) backing the numbers in the paper
docs/              methods notes
```

Scripts use **absolute `/oak/...` paths** to the DESI and SDSS catalogs, which are not
redistributable here. Point them at your own copies (see `data/README.md`).

## Reproducing the paper figures

`scripts/figures/submit_figs_altb.sh` is the master driver and runs most of these in
sequence.

| Fig | Output file | Script |
|-----|-------------|--------|
| 1 | `fig1_flowchart_assembled_v6.png` | `scripts/figures/fig1_flowchart_assembled_v6.py` |
| 2 | `completeness_coverage_hist.png` | `scripts/figures/completeness_hist_v2.py` (`completeness_hist_v3.py` = shared-axis revision) |
| 3 | `balmer_decrement_dist.png` | `scripts/figures/plot_balmer_decrement_ALTB.py` |
| 4 | `corner_ratios_NFs_and_data.png` | `scripts/figures/corner_overlay_ratios_sdss_desi_and_flows_v2_ALTB.py` |
| 5 | `corr_matrices_data_nf_diff.png` | `scripts/figures/corr_matrices_data_and_diff.py` |
| 6 | `corner8_sdss_data_vs_nf.png` | `scripts/figures/corner8_sdss_data_vs_nf.py` |
| 7 | `cross_survey_transfer_invcmap.png` | `scripts/figures/cross_survey_lums_fig1_pretty_ALTB.py` (via `submit_transfer_invcmap.sh`) |
| 8 | `balmer_conditioning_scatter.png` | `scripts/evaluation/test_balmer_conditioning_ALTB.py` |
| 9 | `hiz_lum_1to1.png` | `scripts/highz/hiz_lum_1to1.py` |

Table 1 is produced by `scripts/evaluation/eval_altb_table1.py`.

> **Note on the `_ALTB` scripts.** Several figure scripts ending in `_ALTB.py` are generated
> by `sed` from a base script by the corresponding `submit_*_fix.sh` wrapper, which
> substitutes the model and catalog filenames. Both the base and generated scripts are
> committed so the published figures are reproducible, but if you modify a base script you
> must re-run its wrapper to regenerate the `_ALTB` variant. Making the paths configurable
> instead of `sed`-substituted is a known cleanup.

## Data

The DESI and SDSS catalogs this model is trained on are public but too large to ship here:

- **DESI BGS** emission lines: `FastSpecFit` value-added catalog (Moustakas et al. 2023)
- **SDSS MGS** emission lines and derived properties: MPA–JHU catalog (DR8)
- **DESI–COSMOS** (Appendix C validation): Ratajczak et al. 2026

See `data/README.md` for the specific files and columns the scripts expect.

## Citation

If you use this model, please cite:

> Hartley, A. I., Cooray, S., & Wechsler, R. H. (2026), *An Empirical Model for Galaxy
> Nebular Emission*, in preparation.

## License

MIT — see [LICENSE](LICENSE).
