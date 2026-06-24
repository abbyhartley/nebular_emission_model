# CLAUDE.md — nebular_emission_model

Context doc for Claude Code sessions. I (Claude) run on Abby's local Mac and reach this
repo over SSH (`ssh sherlock`). Read this file first at the start of each session.

## Goal
Train **conditional normalizing flows** to learn the joint (covariant) distribution of
**8 bright optical emission-line ratios** relative to Halpha, conditioned on two global
galaxy observables: **stellar mass (logM*)** and **Halpha luminosity (logL_Ha)**.
Architecture: **block neural autoregressive flow** built with **FlowJAX** (JAX/Equinox).
Trained/tested on **DESI BGS** (FastSpecFit) and **SDSS Main Galaxy Sample** (MPA-JHU).
A paper is in progress ("An Empirical Model for Galaxy Nebular Emission").

The 8 modeled lines (targets are log10(L_line / L_Ha)):
Hbeta, Hgamma, [NII]6584, [SII]6716, [SII]6731, [OII]3726, [OII]3729, [OIII]5007.
Conditioning vector u = (logM*, logL_Ha). Recover absolute lum: logL_line = logL_Ha + ratio.

## Environment & cluster
- Host: Stanford **Sherlock** HPC. Group: `cyaolai` / `oak_cyaolai`. Partition: **serc**.
- Conda env: **`desi_nf`** (`source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate desi_nf`).
- Module: `python/3.12.1`. Key libs: jax, equinox, flowjax, astropy, numpy, pandas.
- NEVER run heavy compute on the login node — submit via `sbatch` (see Workflow).

## Key paths
- Repo: `/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model`
- Data lives ONE LEVEL UP in `.../gfc_NFs/` (FITS files, gitignored):
  - DESI training: `gfc_NFs/DESI_BGS_training_data.fits`
  - SDSS training: `gfc_NFs/SDSS_main_training_data.fits`
- Trained models (in repo root): `nf_desi_bgs.eqx` + `nf_desi_bgs_meta.pkl`,
  `nf_sdss_main.eqx` + `nf_sdss_main_meta.pkl` (`.eqx` = Equinox serialized flow;
  `meta.pkl` = normalization stats X_mean/X_std/U_mean/U_std + resolved column map).

## Package: src/normflow
- `stellar_mass.py` — color-derived stellar mass: `desi_to_sdss_gmr`,
  `log10_ml_r_from_gmr_sdss`, `log10_stellar_mass_color`. Unifies M* across surveys via
  (g-r) color (log(M/L_r)=1.062(g-r)-0.555; DESI->SDSS color transform).
- `selection.py` — sample-selection cuts (DESISelectionConfig, masks, FITS writers).
- `train_NF.py` — `train_line_ratio_flow(...)`: the core FlowJAX training routine.

## Scripts (scripts/)
- Train: `train_nf_desi.py`, `train_nf_sdss.py` (read training FITS, write .eqx + meta).
- Selection: `select_desibgs_training_sample.py`, `select_sdssmain_training_sample.py`.
- Eval (NLL + summary stats): `eval_nf_summarystats_NLL_desi.py`,
  `eval_nf_summarystats_NLL_sdss.py`, `..._sdss_on_desi_mag_limited.py`.
- Cross-survey / diagnostics: `check_balmer_dec_from_flows.py`,
  `fit_all_flux_mapping_desi_to_sdss.py`, `match_sdss_desi_compare_line_*.py`.
- `scripts/plotting/` — paper figures (BPT, corner ratio overlays, cross-survey transfer).
- `scripts/UM_comp/` — UniverseMachine comparison / OII luminosity-function application.

## Workflow (run a job)
1. Edit the relevant submit script to call the desired python script:
   - `submit.sh` (repo root) or `scripts/submit.sh` — uncomment the target `python ...` line.
2. `sbatch submit.sh` (run from the dir whose `logs/` you want output in).
3. Monitor: `squeue -u hartley1`; logs in `logs/<job>_<jobid>.{out,err}`.
4. SLURM template: partition serc, 1 node, 4 cpus, 50G mem, 12:00:00 walltime.

## Data conventions
- DESI fluxes are LINEAR in units of 1e-17 erg/s/cm^2 (FLUX_SCALE=1e-17); SDSS differs — check meta.
- Quality cuts: Redrock type=GALAXY, z>0.05 & ZWARN=0, S/N>5 on all 8 lines, Halpha IVAR>0.
- Stellar mass: use unified COLOR-derived M* (`LOGM_COLOR`), not native spectral M*.
- DESI [SII] uses 6716 (paper text says 6717 — same line).

## Git
- Remote: `git@github.com:abbyhartley/nebular_emission_model.git` (public).
- NOTE: many recent scripts/models are uncommitted/untracked locally on Sherlock and
  NOT yet pushed to GitHub. The Sherlock copy is the source of truth — trust it over GitHub.
- `.gitignore` excludes *.fits, data/, results/, *.png/pdf/npy/npz.

## Session etiquette for Claude
- Confirm connection with a cheap command before heavy work.
- Keep big I/O and training inside `sbatch` jobs, never the login node.
- Update this file when paths, env, or the model design change.
