#!/bin/bash
#SBATCH --job-name=benchp
#SBATCH --output=logs/benchp_%j.out
#SBATCH --error=logs/benchp_%j.err
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
module purge
module load python/3.12.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate desi_nf
cd /oak/stanford/groups/cyaolai/AbbyHartley/gfc_NFs/nebular_emission_model/scripts

echo "=== HARDWARE ==="
lscpu | grep -E "Model name"
echo "SLURM cpus-per-task: $SLURM_CPUS_PER_TASK"
echo

# Single-core pinning is done inside bench_paper.py via os.sched_setaffinity
# (to a core from the job's own cpuset), which is robust to whatever cores SLURM assigns.
python bench_paper.py
echo "=== DONE ==="
