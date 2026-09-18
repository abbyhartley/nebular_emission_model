#!/bin/bash
#SBATCH --job-name=figsaltb
#SBATCH --output=logs/figsaltb_%j.out
#SBATCH --error=logs/figsaltb_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf

REPO=/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model
PLOT=$REPO/scripts/figures
SCR=$REPO/scripts/evaluation
OUT=$REPO/figs_ALTB
mkdir -p $OUT

# uniform input-path repoint: strict production files -> ALT-B files
sed_in () {
  sed -e 's/nf_desi_bgs/nf_desi_ALTB/g' \
      -e 's/nf_sdss_main/nf_sdss_ALTB/g' \
      -e 's/DESI_BGS_training_data\.fits/DESI_BGS_training_data_ALTB.fits/g' \
      -e 's/SDSS_main_training_data\.fits/SDSS_main_training_data_ALTB.fits/g' "$1"
}

# cwd-relative-output plotting scripts: input repoint only (run from $OUT)
for s in flowchart_inset_plots corner_overlay_ratios_sdss_desi_and_flows_v2 \
         cross_survey_lums_fig1_pretty in_survey_lums_fig_pretty ; do
  sed_in $PLOT/$s.py > $PLOT/${s}_ALTB.py
done
# absolute figs/ output plotting scripts: also redirect figs/ -> figs_ALTB/
for s in plot_balmer_decrement plot_correlation_matrices ; do
  sed_in $PLOT/$s.py | sed 's|figs/|figs_ALTB/|g' > $PLOT/${s}_ALTB.py
done
# balmer conditioning (in scripts/, trains internally): input repoint + "figs" -> "figs_ALTB"
sed_in $SCR/test_balmer_conditioning.py | sed 's|"figs"|"figs_ALTB"|g' > $SCR/test_balmer_conditioning_ALTB.py

cd $OUT
echo ">>> flowchart insets";    python $PLOT/flowchart_inset_plots_ALTB.py                          || echo "FAIL flowchart"
echo ">>> corner";              python $PLOT/corner_overlay_ratios_sdss_desi_and_flows_v2_ALTB.py   || echo "FAIL corner"
echo ">>> cross-survey";        python $PLOT/cross_survey_lums_fig1_pretty_ALTB.py                  || echo "FAIL crosssurvey"
echo ">>> in-survey";           python $PLOT/in_survey_lums_fig_pretty_ALTB.py                      || echo "FAIL insurvey"
echo ">>> balmer decrement";    python $PLOT/plot_balmer_decrement_ALTB.py                          || echo "FAIL balmerdec"
echo ">>> correlation";         python $PLOT/plot_correlation_matrices_ALTB.py                      || echo "FAIL corr"
echo ">>> balmer conditioning"; python $SCR/test_balmer_conditioning_ALTB.py                        || echo "FAIL balmercond"

# rename to requested figure names
cd $OUT
[ -f corner_filled_data_plus_nf_contours_scienceplots.png ] && cp corner_filled_data_plus_nf_contours_scienceplots.png corner_ratios_NFs_and_data.png
[ -f cross_survey_transfer_rows_cmasher_bubblegum_fullrange.png ] && cp cross_survey_transfer_rows_cmasher_bubblegum_fullrange.png cross_survey_transfer.png
echo "=== OUTPUT FILES ==="; ls -la $OUT/*.png 2>/dev/null
echo "=== DONE ==="
