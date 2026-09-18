#!/bin/bash
#SBATCH --job-name=comphist2
#SBATCH --output=logs/comphist2_%j.out
#SBATCH --error=logs/comphist2_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/figures
python completeness_hist_v2.py
echo "=== DONE ==="
