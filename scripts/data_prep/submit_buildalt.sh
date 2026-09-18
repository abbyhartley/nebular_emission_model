#!/bin/bash
#SBATCH --job-name=buildalt
#SBATCH --output=logs/buildalt_%j.out
#SBATCH --error=logs/buildalt_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --time=1:30:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python build_alt_desi_samples.py
echo "=== DONE ==="
