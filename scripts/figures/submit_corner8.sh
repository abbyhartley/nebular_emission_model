#!/bin/bash
#SBATCH --job-name=corner8
#SBATCH --output=logs/corner8_%j.out
#SBATCH --error=logs/corner8_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/figures
python corner8_sdss_data_vs_nf.py
echo "=== DONE ==="
