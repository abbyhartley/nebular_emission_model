#!/bin/bash
#SBATCH --job-name=refine2
#SBATCH --output=logs/refine2_%j.out
#SBATCH --error=logs/refine2_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
REPO=/oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model
PLOT=$REPO/scripts/figures; SCR=$REPO/scripts; OUT=$REPO/figs_ALTB; mkdir -p $OUT
sed_in () {
  sed -e 's/nf_desi_bgs/nf_desi_ALTB/g' -e 's/nf_sdss_main/nf_sdss_ALTB/g' \
      -e 's/DESI_BGS_training_data\.fits/DESI_BGS_training_data_ALTB.fits/g' \
      -e 's/SDSS_main_training_data\.fits/SDSS_main_training_data_ALTB.fits/g' "$1"
}
# in-survey + cross-survey: per-panel robust limits (hex_panel self-computes) + no axis sharing
for s in in_survey_lums_fig_pretty cross_survey_lums_fig1_pretty ; do
  sed_in $PLOT/$s.py > $PLOT/${s}_ALTB.py
  python $SCR/patch_perpanel.py $PLOT/${s}_ALTB.py
done
# balmer: legend entries -> 10 (match title), title -> 10; output to figs_ALTB
sed_in $PLOT/plot_balmer_decrement.py \
  | sed -e 's/"legend.fontsize": 13,/"legend.fontsize": 10,/' \
        -e 's/loc="upper right",/loc="upper right", fontsize=10, labelspacing=0.3, handlelength=1.4,/' \
        -e 's/title_fontsize=13/title_fontsize=10/' \
        -e 's|figs/|figs_ALTB/|g' \
  > $PLOT/plot_balmer_decrement_ALTB.py
# flowchart: same robust fence in square_hexbin + inverter + reduced volume
sed_in $PLOT/flowchart_inset_plots.py \
  | sed -e 's/cond_dim=2,/cond_dim=2, inverter=_ROBUST_INV,/g' -e 's/cond_dim=2$/cond_dim=2, inverter=_ROBUST_INV/g' \
        -e 's/^N_MC = 50/N_MC = 20/' -e 's/^N_PLOT = 200_000/N_PLOT = 50_000/' \
        -e 's|lo = np.min(np.concatenate(\[x, y\]))|_xy=np.concatenate([x,y]); _xy=_xy[np.isfinite(_xy)]; _q1,_q3=np.percentile(_xy,[25,75]); _fe=3.0*(_q3-_q1); _kp=_xy[(_xy>=_q1-_fe)\&(_xy<=_q3+_fe)]; lo=float(_kp.min())|' \
        -e 's|hi = np.max(np.concatenate(\[x, y\]))|hi=float(_kp.max())|' \
  > $PLOT/flowchart_inset_plots_ALTBfix.py
python $SCR/patch_inverter.py $PLOT/flowchart_inset_plots_ALTBfix.py

cd $OUT
echo ">>> in-survey";    time python $PLOT/in_survey_lums_fig_pretty_ALTB.py     || echo FAIL_insurvey
echo ">>> cross-survey"; time python $PLOT/cross_survey_lums_fig1_pretty_ALTB.py || echo FAIL_crosssurvey
echo ">>> balmer";       time python $PLOT/plot_balmer_decrement_ALTB.py          || echo FAIL_balmer
echo ">>> flowchart";    time python $PLOT/flowchart_inset_plots_ALTBfix.py       || echo FAIL_flowchart
[ -f cross_survey_transfer_rows_cmasher_bubblegum_fullrange.png ] && cp cross_survey_transfer_rows_cmasher_bubblegum_fullrange.png cross_survey_transfer.png
echo "=== DONE ==="; ls -la $OUT/in_survey_lineratios.png $OUT/cross_survey_transfer.png $OUT/balmer_decrement_dist.png $OUT/inset_0*.png 2>/dev/null
