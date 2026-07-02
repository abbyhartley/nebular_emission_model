#!/bin/bash
#SBATCH --job-name=balmer_cond
#SBATCH --output=logs/balmer_cond_%j.out
#SBATCH --error=logs/balmer_cond_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model
python scripts/test_balmer_conditioning.py
