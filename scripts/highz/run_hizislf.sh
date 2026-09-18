#!/bin/bash
#SBATCH --job-name=hizislf
#SBATCH --output=plotting/logs/hizislf_%j.out
#SBATCH --error=plotting/logs/hizislf_%j.err
#SBATCH --partition=serc
#SBATCH -N 1 -n 1 --cpus-per-task=8 --mem=96G --time=1:00:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
python hiz_insurvey_lf.py
echo "=== JOB DONE ==="
