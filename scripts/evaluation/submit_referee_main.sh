#!/bin/bash
#SBATCH --job-name=refmain
#SBATCH --output=logs/refmain_%j.out
#SBATCH --error=logs/refmain_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
echo "===== A1 PIT/coverage ====="; python pit_calibration.py
echo "===== A2 NLL + tails baseline ====="; python benchmark_nll_tails.py
echo "===== A2b correlation benchmark (rerun, recalibrated DESI) ====="; python tier1_covariance_benchmark.py
echo "===== A4 AGN fractions + SF-only [OIII] ====="; python agn_fraction_oiii.py
echo "===== DONE ====="
