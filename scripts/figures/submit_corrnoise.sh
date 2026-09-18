#!/bin/bash
#SBATCH --job-name=corrnoise
#SBATCH --output=logs/corrnoise_%j.out
#SBATCH --error=logs/corrnoise_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/figures
python corr_noise_diagnosis.py
echo "=== DONE ==="
