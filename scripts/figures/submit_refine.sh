#!/bin/bash
#SBATCH --job-name=refine
#SBATCH --output=logs/refine_%j.out
#SBATCH --error=logs/refine_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
REPO=/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model
PLOT=$REPO/scripts/figures; OUT=$REPO/figs_ALTB; mkdir -p $OUT

sed_in () {
  sed -e 's/nf_desi_bgs/nf_desi_ALTB/g' -e 's/nf_sdss_main/nf_sdss_ALTB/g' \
      -e 's/DESI_BGS_training_data\.fits/DESI_BGS_training_data_ALTB.fits/g' \
      -e 's/SDSS_main_training_data\.fits/SDSS_main_training_data_ALTB.fits/g' "$1"
}
# percentile axis limits for in_survey + cross_survey (clip outlier stretch)
for s in in_survey_lums_fig_pretty cross_survey_lums_fig1_pretty ; do
  sed_in $PLOT/$s.py \
    | sed 's/return float(np.min(v)), float(np.max(v))/return float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))/' \
    > $PLOT/${s}_ALTB.py
done
# balmer decrement: shrink legend so it clears the peak
sed_in $PLOT/plot_balmer_decrement.py \
  | sed -e 's/"legend.fontsize": 13,/"legend.fontsize": 9,/' \
        -e 's/loc="upper right",/loc="upper right", fontsize=8, title_fontsize=9, labelspacing=0.3, handlelength=1.4,/' \
  > $PLOT/plot_balmer_decrement_ALTB.py
# flowchart: input repoint + inverter + reduced volume + percentile square_hexbin limits
sed_in $PLOT/flowchart_inset_plots.py \
  | sed -e 's/cond_dim=2,/cond_dim=2, inverter=_ROBUST_INV,/g' -e 's/cond_dim=2$/cond_dim=2, inverter=_ROBUST_INV/g' \
        -e 's/^N_MC = 50/N_MC = 20/' -e 's/^N_PLOT = 200_000/N_PLOT = 50_000/' \
        -e 's|lo = np.min(np.concatenate(\[x, y\]))|lo = np.percentile(np.concatenate([x, y]), 0.5)|' \
        -e 's|hi = np.max(np.concatenate(\[x, y\]))|hi = np.percentile(np.concatenate([x, y]), 99.5)|' \
  > $PLOT/flowchart_inset_plots_ALTBfix.py
python $REPO/scripts/patch_inverter.py $PLOT/flowchart_inset_plots_ALTBfix.py

cd $OUT
echo ">>> in-survey";     time python $PLOT/in_survey_lums_fig_pretty_ALTB.py    || echo FAIL_insurvey
echo ">>> cross-survey";  time python $PLOT/cross_survey_lums_fig1_pretty_ALTB.py || echo FAIL_crosssurvey
echo ">>> balmer dec";    time python $PLOT/plot_balmer_decrement_ALTB.py         || echo FAIL_balmer
echo ">>> flowchart";     time python $PLOT/flowchart_inset_plots_ALTBfix.py      || echo FAIL_flowchart
[ -f cross_survey_transfer_rows_cmasher_bubblegum_fullrange.png ] && cp cross_survey_transfer_rows_cmasher_bubblegum_fullrange.png cross_survey_transfer.png
echo "=== DONE ==="; ls -la $OUT/in_survey_lineratios.png $OUT/cross_survey_transfer.png $OUT/balmer_decrement_dist.png $OUT/inset_0*.png 2>/dev/null
