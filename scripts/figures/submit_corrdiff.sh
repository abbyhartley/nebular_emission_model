#!/bin/bash
#SBATCH --job-name=corrdiff
#SBATCH --output=logs/corrdiff_%j.out
#SBATCH --error=logs/corrdiff_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/figures
python corr_matrices_data_and_diff.py
echo "=== DONE ==="
