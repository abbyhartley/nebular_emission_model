#!/bin/bash
#SBATCH --job-name=color_test
#SBATCH --output=logs/color_test_%j.out
#SBATCH --error=logs/color_test_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=12:00:00
module purge
module load python/3.12.1
# ---- ACTIVATE CONDA ENV ----
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model

python add_colorSM_to_desifits.py
