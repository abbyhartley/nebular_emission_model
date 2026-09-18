#!/bin/bash
#SBATCH --job-name=flowfix
#SBATCH --output=logs/flowfix_%j.out
#SBATCH --error=logs/flowfix_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
REPO=/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model
PLOT=$REPO/scripts/figures; OUT=$REPO/figs_ALTB; mkdir -p $OUT
s=flowchart_inset_plots
sed -e 's/nf_desi_bgs/nf_desi_ALTB/g' -e 's/nf_sdss_main/nf_sdss_ALTB/g' \
    -e 's/DESI_BGS_training_data\.fits/DESI_BGS_training_data_ALTB.fits/g' \
    -e 's/SDSS_main_training_data\.fits/SDSS_main_training_data_ALTB.fits/g' \
    -e 's/cond_dim=2,/cond_dim=2, inverter=_ROBUST_INV,/g' -e 's/cond_dim=2$/cond_dim=2, inverter=_ROBUST_INV/g' \
    -e 's/^N_MC = 50/N_MC = 20/' -e 's/^N_PLOT = 200_000/N_PLOT = 50_000/' \
    $PLOT/$s.py > $PLOT/${s}_ALTBfix.py
python $REPO/scripts/patch_inverter.py $PLOT/${s}_ALTBfix.py
grep -E "^N_MC|^N_PLOT" $PLOT/${s}_ALTBfix.py
cd $OUT
echo ">>> flowchart (fixed, reduced volume)"; time python $PLOT/${s}_ALTBfix.py || echo "FAIL flowchart"
ls -la $OUT/inset_*.png; echo "=== DONE ==="
