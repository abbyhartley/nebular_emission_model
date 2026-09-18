#!/bin/bash
#SBATCH --job-name=evaltab1
#SBATCH --output=logs/evaltab1_%j.out
#SBATCH --error=logs/evaltab1_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python eval_altb_table1.py
echo "=== DONE ==="
