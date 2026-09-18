#!/bin/bash
#SBATCH --job-name=fig1v5
#SBATCH --output=logs/fig1v5_%j.out
#SBATCH --error=logs/fig1v5_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/figures
python fig1_flowchart_assembled_v5.py
echo "=== DONE ==="
