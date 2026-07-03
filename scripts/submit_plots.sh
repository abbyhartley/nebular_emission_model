#!/bin/bash
#SBATCH --job-name=replots
#SBATCH --output=logs/replots_%j.out
#SBATCH --error=logs/replots_%j.err
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
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts
echo "=== regenerate crossmatch CSV (new calibrated DESI file) ==="
python crossmatch_sdss_desi.py
echo "=== phase1 flux compare ==="
python phase1_flux_compare.py
echo "=== phase2 mass+zdecomp (same-galaxy) ==="
python phase2_mass_zdecomp_xm.py
cd plotting
echo "=== plot_zdecomp_styled ==="
python plot_zdecomp_styled.py
echo "=== plot_data_distributions ==="
python plot_data_distributions.py
echo "=== plot_mstar_rho_sweep (MC, long) ==="
python plot_mstar_rho_sweep.py
echo "=== DONE ==="
