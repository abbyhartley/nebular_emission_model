#!/bin/bash
#SBATCH --job-name=step2
#SBATCH --output=logs/step2_%j.out
#SBATCH --error=logs/step2_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
echo "===== Z-DECOMPOSITION ====="
python test_zdecomp.py
echo ""
echo "===== M* ZERO-POINT ====="
python test_mstar_zeropoint.py
