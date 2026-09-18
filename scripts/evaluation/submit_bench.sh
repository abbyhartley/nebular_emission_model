#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts

echo "=== HARDWARE ==="
echo "SLURM cpus-per-task: $SLURM_CPUS_PER_TASK"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
echo

echo "=== NF BENCHMARK ==="
python bench_nf.py
echo

echo "=== SPS+CLOUDY (FSPS) BENCHMARK ==="
python bench_fsps.py
echo "=== DONE ==="
