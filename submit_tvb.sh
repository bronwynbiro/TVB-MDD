#!/bin/bash
# Job array: one task = one (subject, run). UNTESTED on Fir, so run one task first (see steps below).
#SBATCH --account=def-YOURSUPERVISOR      # your supervisor's allocation account (placeholder)
#SBATCH --job-name=tvbmdd
#SBATCH --time=01:00:00                   # about 15-20 min per run at 150 epochs / 0.02 step on my test core; padded
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G                          # my test peaked at ~0.8 GB
#SBATCH --array=0-99%50                   # 100 tasks = 20 subjects x 5 runs, at most 50 running at once
#SBATCH --output=logs/%x_%A_%a.out

NRUNS=5                                   # runs per subject; must match the array size above (20 x NRUNS tasks)
SUB=$(( SLURM_ARRAY_TASK_ID / NRUNS ))
RUN=$(( SLURM_ARRAY_TASK_ID % NRUNS ))

module load python/3.10
source $HOME/tvb310/bin/activate
export OMP_NUM_THREADS=1 MPLBACKEND=Agg

python run_one.py --sub $SUB --run $RUN \
    --data $HOME/tvb_mdd/data \
    --out  $HOME/scratch/tvb_results \
    --epochs 150 --step 0.02
