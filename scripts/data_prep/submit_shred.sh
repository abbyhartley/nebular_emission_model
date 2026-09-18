#!/bin/bash
#SBATCH --job-name=shredchk
#SBATCH --output=logs/shredchk_%j.out
#SBATCH --error=logs/shredchk_%j.err
#SBATCH --partition=serc
#SBATCH -N 1 -n 1 --cpus-per-task=4 --mem=64G --time=0:30:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python shred_maskbits_check.py
echo "=== DONE ==="
