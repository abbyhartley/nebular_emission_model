#!/bin/bash
#SBATCH --job-name=recolor_all
#SBATCH --output=logs/recolor_all_%j.out
#SBATCH --error=logs/recolor_all_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
module purge; module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts/figures
echo "=== insets v4 ==="; python flowchart_inset_plots_ALTB_v4.py
echo "=== assembled v4 ==="; python fig1_flowchart_assembled_v4.py
echo "=== in-survey ==="; python in_survey_lums_fig_pretty_ALTB.py
echo "=== cross-survey ==="; python cross_survey_lums_fig1_pretty_ALTB.py
echo "=== DONE ==="
