#!/bin/bash
#SBATCH --job-name=figsfix
#SBATCH --output=logs/figsfix_%j.out
#SBATCH --error=logs/figsfix_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
REPO=/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model
PLOT=$REPO/scripts/figures
OUT=$REPO/figs_ALTB
mkdir -p $OUT

sed_all () {
  sed -e 's/nf_desi_bgs/nf_desi_ALTB/g' -e 's/nf_sdss_main/nf_sdss_ALTB/g' \
      -e 's/DESI_BGS_training_data\.fits/DESI_BGS_training_data_ALTB.fits/g' \
      -e 's/SDSS_main_training_data\.fits/SDSS_main_training_data_ALTB.fits/g' \
      -e 's/cond_dim=2,/cond_dim=2, inverter=_ROBUST_INV,/g' \
      -e 's/cond_dim=2$/cond_dim=2, inverter=_ROBUST_INV/g' "$1"
}
for s in flowchart_inset_plots corner_overlay_ratios_sdss_desi_and_flows_v2 ; do
  sed_all $PLOT/$s.py > $PLOT/${s}_ALTBfix.py
  python $REPO/scripts/patch_inverter.py $PLOT/${s}_ALTBfix.py
done

cd $OUT
echo ">>> flowchart insets (fixed)"; python $PLOT/flowchart_inset_plots_ALTBfix.py                        || echo "FAIL flowchart"
echo ">>> corner (fixed)";          python $PLOT/corner_overlay_ratios_sdss_desi_and_flows_v2_ALTBfix.py  || echo "FAIL corner"
[ -f corner_filled_data_plus_nf_contours_scienceplots.png ] && cp corner_filled_data_plus_nf_contours_scienceplots.png corner_ratios_NFs_and_data.png
echo "=== OUTPUTS ==="; ls -la $OUT/inset_*.png $OUT/corner_*.png 2>/dev/null
echo "=== DONE ==="
