#!/bin/bash -l
# =============================================================================
# STAGE 2: train the MIL pattern classifier off the precomputed graph cache.
#
# CPU BY DEFAULT, and that is usually the faster route end-to-end. The workload
# is thousands of SMALL graphs, so GPU utilisation is poor and most of the time
# goes on kernel-launch overhead -- while a gpu=1 request queues behind every
# segmentation job on the cluster. A CPU job starts almost immediately. Waiting
# 14h for a GPU to save an hour of compute is a bad trade.
#
# train_patterns.py defaults --device to cuda-if-available, so it picks CPU on
# its own here and CUDA automatically when a GPU is granted.
#
# SUBMIT (CPU, schedules fast):
#     qsub scripts/run_patterns.sh
#
# SUBMIT (GPU, only worth it once the queue is quiet):
#     qsub -l gpu=1 scripts/run_patterns.sh
#
# NOTE: SGE spools a COPY of this script at submit time, so editing it does not
# affect an already-queued job. Change something? qdel and resubmit.
# =============================================================================

#$ -N patterns
#$ -l h_rt=12:0:0
#$ -l mem=16G
#$ -pe smp 4
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/patterns.out
#$ -e /home/ucabim3/Scratch/logs/patterns.err

# Source from the repo, not a copy in $HOME, and abort if it is not there.
ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/segmentation/cellvit_env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }
mkdir -p /home/ucabim3/Scratch/logs
echo "=== $(date) on $(hostname) ==="
# only meaningful if a GPU was actually granted
command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi --query-gpu=name --format=csv,noheader \
    || echo "no GPU allocated -- running on CPU"

# --seeds 3 for a first pass: 3 seeds x 5 folds = 15 runs/arm, enough to see
# whether an arm clears the control. Raise to 10 for the confirmatory run once
# the cost per run is known -- seeds multiply runtime linearly.
python -u train_patterns.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --labels /home/ucabim3/Scratch/til_indices.csv \
    --task pattern4 \
    --seeds 3

echo "=== done: $(date) ==="
