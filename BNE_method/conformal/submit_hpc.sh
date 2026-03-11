#!/bin/bash
#SBATCH --job-name=conformal_bne
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --mem=80000
#SBATCH --output=conformal_%A_%a.out
#SBATCH --array=0-11

# ============================================================
# Conformalized BNE — Simulation Sweep
#
# Runs 12 jobs: 3 sample sizes x 4 seeds
# Usage:  sbatch submit_hpc.sh
# ============================================================

# Configuration
SAMPLE_SIZES=(250 500 1000)
SEEDS=(0 42 123 456)

# Map SLURM_ARRAY_TASK_ID -> (n_train, seed)
N_SIZES=${#SAMPLE_SIZES[@]}
SIZE_IDX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))

N_TRAIN=${SAMPLE_SIZES[$SIZE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Job ${SLURM_ARRAY_TASK_ID}: n_train=${N_TRAIN}, seed=${SEED}"

# Activate conda environment (adjust name if needed)
source activate BNE

# Run
cd "$(dirname "$0")"
python run_simulation.py --n_train ${N_TRAIN} --seed ${SEED}
