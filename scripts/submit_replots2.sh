#!/bin/bash
#SBATCH --job-name=replot2
#SBATCH --output=logs/replot2_%j.out
#SBATCH --error=logs/replot2_%j.err
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
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/plotting
echo "===== balmer decrement dist ====="
python plot_balmer_decrement.py
echo "===== corner overlay (data + both flows) ====="
python corner_overlay_ratios_sdss_desi_and_flows_v2.py
echo "===== cross-survey transfer (bubblegum) ====="
python cross_survey_lums_fig1_pretty.py
echo "===== DONE ====="
