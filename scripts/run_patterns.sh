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
# 16G not 48G: 48 was set when regions-per-batch was 16 and training was
# full-batch. A 48G single-slot request is harder for the scheduler to
# place than the GPU, and jobs sat unqueued for hours because of it.
#$ -l mem=16G
# No GPU. Capacity-matched models are ~10^4 parameters over a few thousand
# nodes per region, which a CPU handles in minutes -- but a gpu=1 request
# queued for hours to days behind real GPU work: the wait dwarfed the run.
# Add -l gpu=1 on the command line if a sweep ever outgrows this.
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/patterns.$JOB_ID.out
#$ -e /home/ucabim3/Scratch/logs/patterns.$JOB_ID.err

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }
mkdir -p /home/ucabim3/Scratch/logs
echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "CPU only"

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
