#!/bin/bash
#SBATCH --job-name=xfer_invcmap
#SBATCH --output=logs/xfer_invcmap_%j.out
#SBATCH --error=logs/xfer_invcmap_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:30:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/figures
echo "=== in-survey (inverted cmap) ==="
python in_survey_lums_fig_pretty_ALTB.py
echo "=== cross-survey (inverted cmap) ==="
python cross_survey_lums_fig1_pretty_ALTB.py
echo "=== DONE ==="
