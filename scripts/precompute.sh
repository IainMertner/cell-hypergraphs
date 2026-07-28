#!/bin/bash -l
# =============================================================================
# STAGE 1: precompute all region graphs for every slide, once.
# CPU job -- no GPU. Run this once; afterwards run_patterns.sh loads the cache
# and runs in minutes.
#
# SUBMIT:
#     qsub precompute.sh
# =============================================================================

#$ -N precompute
#$ -l h_rt=6:0:0          # generous; all arms x all regions x ~99 slides
#$ -l mem=16G             # per slot
#$ -pe smp 4
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/precompute.out
#$ -e /home/ucabim3/Scratch/logs/precompute.err

source /home/ucabim3/cellvit_env.sh
mkdir -p /home/ucabim3/Scratch/logs

echo "=== $(date) on $(hostname) ==="

# -u = unbuffered, so per-slide progress appears in the log live rather than
# only when the job ends. Arms default to pw-knn + hg-knn -- the one comparison
# the project is trying to answer. To add an arm later, rerun this with
# `--arms hg-radius`: existing slides are topped up with just the new arm rather
# than rebuilt, so it costs one arm, not the whole cache.
python -u precompute_graphs.py \
    --cache-root /home/ucabim3/Scratch/cellvit_out \
    --out /home/ucabim3/Scratch/graph_cache

echo "=== done: $(date) ==="
