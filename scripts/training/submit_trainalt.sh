#!/bin/bash
#SBATCH --job-name=trainalt
#SBATCH --output=logs/trainalt_%j.out
#SBATCH --error=logs/trainalt_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
GFC=/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs
echo ">>> training ALT-A"; time python train_alt.py $GFC/DESI_ALT_A.fits $GFC/nebular_emission_model/nf_desi_ALT_A.eqx $GFC/nebular_emission_model/nf_desi_ALT_A_meta.pkl
echo ">>> training ALT-B"; time python train_alt.py $GFC/DESI_ALT_B.fits $GFC/nebular_emission_model/nf_desi_ALT_B.eqx $GFC/nebular_emission_model/nf_desi_ALT_B_meta.pkl
echo "=== DONE ==="
