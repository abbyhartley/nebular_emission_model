#!/bin/bash
#SBATCH --job-name=bylerfig
#SBATCH --output=logs/bylerfig_%j.out
#SBATCH --error=logs/bylerfig_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
module purge
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python plot_byler_3d.py
python plot_byler_nf.py
echo DONE
