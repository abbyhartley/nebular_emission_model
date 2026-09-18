#!/bin/bash
#SBATCH --job-name=fig5p50
#SBATCH --output=logs/fig5p50_%j.out
#SBATCH --error=logs/fig5p50_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
echo "===== Figure 5: correlation matrices ====="
cd plotting && python plot_correlation_matrices.py && cd ..
echo "===== Balmer p50 values ====="
python compute_balmer_p50.py
echo "===== DONE ====="
