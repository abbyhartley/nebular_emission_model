#!/bin/bash
#SBATCH --job-name=buildaltb
#SBATCH --output=logs/buildaltb_%j.out
#SBATCH --error=logs/buildaltb_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python build_altb_production.py
echo "=== DONE ==="
