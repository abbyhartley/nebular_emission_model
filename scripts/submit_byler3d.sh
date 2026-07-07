#!/bin/bash
#SBATCH --job-name=byler3d
#SBATCH --output=logs/byler3d_%j.out
#SBATCH --error=logs/byler3d_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
module purge
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python plot_byler_3d.py
echo DONE
