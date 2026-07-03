#!/bin/bash
#SBATCH --job-name=reeval
#SBATCH --output=logs/reeval_%j.out
#SBATCH --error=logs/reeval_%j.err
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
echo "===== eval_nf_metrics_corrected ====="
python eval_nf_metrics_corrected.py
echo "===== test_mstar_zeropoint ====="
python test_mstar_zeropoint.py
echo "===== test_zdecomp ====="
python test_zdecomp.py
echo "===== DONE ====="
