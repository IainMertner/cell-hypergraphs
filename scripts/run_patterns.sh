#!/bin/bash -l
# =============================================================================
# STAGE 2: train the MIL pattern classifier off the precomputed graph cache.
#
# GPU, and measured worth it: one slide's forward+backward is 97ms vs 17ms
# (pw-knn) and 258ms vs 24ms (hg-knn) -- 5.7x and 10.7x. hg-knn gains most
# because DeepSetsHyperConv scatters over ~720k incidences per slide, which is
# exactly what a GPU is for. The whole sweep is ~30-60min on GPU vs ~10h on CPU.
#
# REQUEST SHAPE is tuned for placement, not just for power:
#   mem=32G       on ONE slot -- no -pe. Requesting N cores means N FREE CORES
#                 ON THE SAME GPU HOST, which is a far harder constraint than the
#                 same memory on a single slot. An otherwise-identical job with
#                 -pe smp 4 sat unplaced for 24h while a CPU job asking DOUBLE
#                 the memory started immediately.
#   h_rt=3:0:0    short jobs backfill into gaps long ones cannot reach
#
# Deliberately NO `-ac allow=...`. That narrows eligibility to one node class,
# and free GPUs were observed on both E and L nodes (4 and 3 respectively) --
# restricting would shrink the reachable pool, not widen it. Jobs here have
# already run on both classes with no allow= flag, so nothing needs unlocking.
#
# train_patterns.py defaults --device to cuda-if-available, so this still works
# (just slower) if no GPU is granted.
#
# SUBMIT:  qsub scripts/run_patterns.sh
#
# NOTE: SGE spools a COPY of this script at submit time, so editing it does not
# affect an already-queued job. Change something? qdel and resubmit.
# =============================================================================

#$ -N patterns
#$ -l h_rt=3:0:0
#$ -l mem=32G
#$ -l gpu=1
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

# --seeds 3 x 5 folds = 15 runs/arm. Affordable on GPU (~30-60min total); this
# would be ~10h on CPU. The progress log prints seconds-per-run, so size the
# confirmatory sweep (--seeds 10) from that measured number.
#
# regions-per-batch left at its default 16. A smaller value (4) measured faster
# on CPU due to cache locality, but on GPU fewer/larger kernel launches are
# usually better -- and it cannot change results either way, since region
# boundaries are never split.
python -u train_patterns.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --labels /home/ucabim3/Scratch/til_indices.csv \
    --task pattern4 \
    --seeds 3

echo "=== done: $(date) ==="
