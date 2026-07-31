#!/bin/bash -l
# =============================================================================
# STAGE 1: precompute all region graphs for every slide, once.
# CPU job -- no GPU. Run this once; afterwards run_patterns.sh loads the cache
# and runs in minutes.
#
# *** NEVER run precompute_graphs.py directly on a login node. *** It is a
# multi-hour, multi-core job over every slide. Login nodes are shared and their
# watchdog will kill it and flag the account. ALWAYS qsub.
#
# SUBMIT -- default arms (pw-knn hg-knn):
#     qsub scripts/precompute.sh
#
# SUBMIT -- adding an arm. Existing slides are TOPPED UP with just the new arm
# rather than rebuilt, so this costs one arm, not the whole cache:
#     qsub -v ARMS="pw-knn hg-knn hg-radius" scripts/precompute.sh
# =============================================================================

#$ -N precompute
#$ -l h_rt=6:0:0          # generous; all arms x all regions x ~99 slides
#$ -l mem=16G             # per slot
#$ -pe smp 4
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/precompute.out
#$ -e /home/ucabim3/Scratch/logs/precompute.err

# Source from the repo, not a copy in $HOME, and abort if it is not there --
# otherwise the job runs on whatever python is on the default PATH and fails
# obscurely later. See the same guard in segmentation/cellvit_chunked.sh.
ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/segmentation/cellvit_env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
mkdir -p /home/ucabim3/Scratch/logs

echo "=== $(date) on $(hostname) ==="

# Arms default to train_patterns.py's DEFAULT_ARMS (pw-knn hg-knn). Override
# with -v ARMS="..." at submit time so this file does not need editing.
ARMS="${ARMS:-}"
echo "arms: ${ARMS:-<default>}"

# -u = unbuffered, so per-slide progress appears in the log live rather than
# only when the job ends.
python -u precompute_graphs.py \
    --cache-root /home/ucabim3/Scratch/cellvit_out \
    --out /home/ucabim3/Scratch/graph_cache \
    ${ARMS:+--arms $ARMS}

echo "=== done: $(date) ==="
