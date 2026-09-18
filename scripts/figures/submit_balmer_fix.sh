#!/bin/bash
#SBATCH --job-name=balmerfix
#SBATCH --output=logs/balmerfix_%j.out
#SBATCH --error=logs/balmerfix_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:30:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
REPO=/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model
PLOT=$REPO/scripts/figures; OUT=$REPO/figs_ALTB
sed -e 's/nf_desi_bgs/nf_desi_ALTB/g' -e 's/nf_sdss_main/nf_sdss_ALTB/g' \
    -e 's/DESI_BGS_training_data\.fits/DESI_BGS_training_data_ALTB.fits/g' \
    -e 's/SDSS_main_training_data\.fits/SDSS_main_training_data_ALTB.fits/g' \
    -e 's/"legend.fontsize": 13,/"legend.fontsize": 9,/' \
    -e 's/loc="upper right",/loc="upper right", fontsize=8, labelspacing=0.3, handlelength=1.4,/' \
    -e 's/title_fontsize=13/title_fontsize=9/' \
    $PLOT/plot_balmer_decrement.py > $PLOT/plot_balmer_decrement_ALTB.py
cd $OUT
echo ">>> balmer dec (fixed legend)"; python $PLOT/plot_balmer_decrement_ALTB.py || echo FAIL_balmer
ls -la $OUT/balmer_decrement_dist.png; echo "=== DONE ==="
