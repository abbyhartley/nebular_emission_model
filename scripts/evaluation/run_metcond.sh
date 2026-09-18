#!/bin/bash
#SBATCH --job-name=metcond
#SBATCH --output=plotting/logs/metcond_%j.out
#SBATCH --error=plotting/logs/metcond_%j.err
#SBATCH --partition=serc
#SBATCH -N 1 -n 1 --cpus-per-task=8 --mem=64G --time=6:00:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python test_metallicity_conditioning_ALTB.py
echo "=== JOB DONE ==="
