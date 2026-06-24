#!/bin/bash
#SBATCH --job-name=flowchart
#SBATCH --output=logs/flowchart_%j.out
#SBATCH --error=logs/flowchart_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=12:00:00
module purge
module load python/3.12.1
# ---- ACTIVATE CONDA ENV ----
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/plotting

python flowchart_inset_plots.py
# python cross_survey_balmer_vs_conditions.py
# python compare_parent_vs_selected_sdss_desi_pretty.py
# python plot_delta_logFha_sdss_minus_desi_vs_logFha_desi.py
# python corner_overlay_ratios_sdss_desi_and_flows_v2.py 
# python cross_survey_lums_fig1_pretty.py
