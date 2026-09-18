#!/bin/bash
#SBATCH --job-name=cueshade
#SBATCH --output=logs/cueshade_%j.out
#SBATCH --error=logs/cueshade_%j.err
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
python plot_grid_shaded.py /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/docs/cue_grid.csv cue
echo DONE
