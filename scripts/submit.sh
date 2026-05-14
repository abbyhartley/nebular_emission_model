#!/bin/bash
#SBATCH --job-name=desiNF
#SBATCH --output=logs/desiNF_%j.out
#SBATCH --error=logs/desiNF_%j.err
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
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts

python eval_nf_summarystats_NLL_sdss.py
