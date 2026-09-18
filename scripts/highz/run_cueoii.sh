#!/bin/bash
#SBATCH --job-name=cueoii
#SBATCH --output=logs/cueoii_%j.out
#SBATCH --error=logs/cueoii_%j.err
#SBATCH --partition=serc
#SBATCH -N 1 -n 1 --cpus-per-task=8 --mem=48G --time=1:00:00
source $HOME/miniconda3/etc/profile.d/conda.sh
ROOT=/oak/stanford/groups/cyaolai/AbbyHartley/conda
export TMPDIR=$ROOT/tmp PIP_CACHE_DIR=$ROOT/pip_cache
conda activate $ROOT/envs/tengri_env
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/tengri_repo
python /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/hiz_cue_oii_prediction.py
echo "=== JOB DONE ==="
