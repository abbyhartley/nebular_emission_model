#!/bin/bash
#SBATCH --job-name=shaded
#SBATCH --output=logs/shaded_%j.out
#SBATCH --error=logs/shaded_%j.err
#SBATCH --partition=serc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
module purge
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python plot_byler_shaded.py
echo DONE
