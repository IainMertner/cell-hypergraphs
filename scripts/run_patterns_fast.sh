#!/bin/bash -l
# SMOKE TEST for stage 2. NOT A RESULT -- do not report numbers from this.
#
# 3 runs/arm at 10 epochs against patience 20, so early stopping never engages
# and every run is truncated mid-convergence. Useful for confirming the pipeline
# completes on real data and for measuring seconds-per-run; useless as evidence.
#
# CPU on purpose, so it starts immediately.
#
#     qsub scripts/run_patterns_fast.sh
#     qsub scripts/run_patterns.sh        # the real one

#$ -N pat_fast
#$ -l h_rt=2:0:0
#$ -l mem=8G
#$ -pe smp 4
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/patterns_fast.out
#$ -e /home/ucabim3/Scratch/logs/patterns_fast.err

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }
mkdir -p /home/ucabim3/Scratch/logs
echo "=== $(date) on $(hostname) ==="

# match torch threads to the slots SGE granted
export OMP_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
echo "slots: ${NSLOTS:-1}"

SEEDS="${SEEDS:-1}"
EPOCHS="${EPOCHS:-10}"
FOLDS="${FOLDS:-3}"
TASK="${TASK:-pattern4}"
RPB="${RPB:-4}"           # smaller groups measured faster on CPU

echo "SMOKE TEST -- NOT A RESULT"
echo "  task=$TASK seeds=$SEEDS folds=$FOLDS epochs=$EPOCHS rpb=$RPB"

python -u train_patterns.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --labels /home/ucabim3/Scratch/til_indices.csv \
    --task "$TASK" \
    --regions-per-batch "$RPB" \
    --epochs "$EPOCHS" \
    --folds "$FOLDS" \
    --seeds "$SEEDS"

echo
echo "REMINDER: smoke test only. Report nothing from this file."
echo "=== done: $(date) ==="
