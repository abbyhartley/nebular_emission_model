#!/bin/bash
#SBATCH --job-name=cueagn
#SBATCH --output=/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/logs/cueagn_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/logs/cueagn_%j.err
#SBATCH --partition=serc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
source $HOME/miniconda3/etc/profile.d/conda.sh
ROOT=/oak/stanford/groups/cyaolai/AbbyHartley/conda
export TMPDIR=$ROOT/tmp PIP_CACHE_DIR=$ROOT/pip_cache
conda activate $ROOT/envs/tengri_env
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/tengri_repo
python gen_cue_agngrid.py
echo DONE
