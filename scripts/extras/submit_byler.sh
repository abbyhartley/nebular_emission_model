#!/bin/bash
#SBATCH --job-name=byler
#SBATCH --output=logs/byler_%j.out
#SBATCH --error=logs/byler_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
export SPS_HOME=/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/FSPS
module purge
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python gen_byler_grid.py
echo DONE
