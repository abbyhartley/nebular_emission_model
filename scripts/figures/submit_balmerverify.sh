#!/bin/bash
#SBATCH --job-name=balmerverify
#SBATCH --output=logs/balmerverify_%j.out
#SBATCH --error=logs/balmerverify_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/figures
echo "=== S/N by line (Part A) ==="
python snr_by_line.py
echo "=== corr matrices dump (Part B data) ==="
python corr_analysis_altb.py
echo "=== DONE ==="
