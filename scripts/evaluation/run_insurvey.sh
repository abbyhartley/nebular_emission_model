#!/bin/bash
#SBATCH --job-name=insurv
#SBATCH --output=plotting/logs/insurv_%j.out
#SBATCH --error=plotting/logs/insurv_%j.err
#SBATCH --partition=serc
#SBATCH -N 1 -n 1 --cpus-per-task=8 --mem=64G --time=0:30:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python insurvey_perline_desi.py
echo "=== JOB DONE ==="
