#!/bin/bash
#SBATCH --job-name=benchp2
#SBATCH --output=logs/benchp2_%j.out
#SBATCH --error=logs/benchp2_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
echo "=== HARDWARE ==="; lscpu | grep -E "Model name"; echo
python bench_paper2.py
echo "=== DONE ==="
