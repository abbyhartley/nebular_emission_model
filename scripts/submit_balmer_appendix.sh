#!/bin/bash
#SBATCH --job-name=balapp
#SBATCH --output=logs/balapp_%j.out
#SBATCH --error=logs/balapp_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
echo "===== IN-SURVEY SDSS Balmer conditioning (oracle + two-stage) ====="
python test_balmer_conditioning.py
echo "===== CROSS-SURVEY Balmer conditioning (external R) on recalibrated data ====="
python test_balmer_cond_cross.py
echo "===== DONE ====="
