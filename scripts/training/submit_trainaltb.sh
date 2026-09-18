#!/bin/bash
#SBATCH --job-name=trainaltb
#SBATCH --output=logs/trainaltb_%j.out
#SBATCH --error=logs/trainaltb_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
GFC=/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs
REPO=$GFC/nebular_emission_model
echo ">>> DESI ALT-B (501k)"; time python train_alt.py      $GFC/DESI_BGS_training_data_ALTB.fits  $REPO/models/nf_desi_ALTB.eqx $REPO/models/nf_desi_ALTB_meta.pkl
echo ">>> SDSS ALT-B (85k)";  time python train_alt_sdss.py $GFC/SDSS_main_training_data_ALTB.fits $REPO/models/nf_sdss_ALTB.eqx $REPO/models/nf_sdss_ALTB_meta.pkl
echo "=== DONE ==="
