#!/bin/bash
#SBATCH --job-name=tailrec
#SBATCH --output=logs/tailrec_%j.out
#SBATCH --error=logs/tailrec_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/figures
python selection_tail_recovery.py
echo "=== DONE ==="
