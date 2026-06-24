#!/bin/bash
#SBATCH --job-name=balmerflr
#SBATCH --output=logs/balmerflr_%j.out
#SBATCH --error=logs/balmerflr_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python check_balmer_dec_from_flows.py
