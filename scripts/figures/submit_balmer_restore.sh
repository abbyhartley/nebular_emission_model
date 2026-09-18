#!/bin/bash
#SBATCH --job-name=balmrest
#SBATCH --output=logs/balmrest_%j.out
#SBATCH --error=logs/balmrest_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/figures
echo ">>> restoring STRICT figs/balmer_decrement_dist (original script, strict flows)"
python plot_balmer_decrement.py
echo "=== DONE ==="
