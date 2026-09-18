#!/bin/bash
#SBATCH --job-name=cont3
#SBATCH --output=logs/cont3_%j.out
#SBATCH --error=logs/cont3_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/figures
python defend_selection_cont3.py
echo "=== DONE ==="
