#!/bin/bash -l
# STAGE 2, the defensible configuration. Use run_patterns_fast.sh for a smoke
# test and do not report its numbers.
#
# NEVER run train_patterns.py on a login node. Always qsub.
#
#     qsub scripts/run_patterns.sh
#
# --seeds 10 x 5 folds = 50 runs/arm. Each seed is a fresh patient->fold
# assignment as well as a fresh init, so the spread reflects which patients land
# in test -- the dominant variance at this n.
#
# --epochs 150 with patience 20: the cap must not be what ends a run. Training
# is full-batch, so an epoch is ONE optimiser step, and a low cap can penalise
# whichever arm converges slower.
#
# h_rt/mem are sized for the job, not the queue. A run killed at the wall
# produces nothing; if it overruns, cut SEEDS rather than raise these.

#$ -N patterns
#$ -l h_rt=12:0:0
#$ -l mem=48G
#$ -l gpu=1
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/patterns.out
#$ -e /home/ucabim3/Scratch/logs/patterns.err

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }
mkdir -p /home/ucabim3/Scratch/logs
echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || echo "WARNING: no GPU visible -- this will not finish in the walltime"

SEEDS="${SEEDS:-10}"
EPOCHS="${EPOCHS:-150}"
FOLDS="${FOLDS:-5}"
TASK="${TASK:-pattern4}"
RPB="${RPB:-16}"          # region grouping: speed only, cannot change results

echo "PROPER RUN | task=$TASK seeds=$SEEDS folds=$FOLDS epochs=$EPOCHS rpb=$RPB"
echo "  -> $((SEEDS * FOLDS)) runs/arm"

python -u train_patterns.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --labels /home/ucabim3/Scratch/til_indices.csv \
    --task "$TASK" \
    --regions-per-batch "$RPB" \
    --epochs "$EPOCHS" \
    --folds "$FOLDS" \
    --seeds "$SEEDS"

echo "=== done: $(date) ==="
