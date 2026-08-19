#!/bin/bash -l
# STAGE 1: build every arm's region graphs for every slide, once.
#
# NEVER run precompute_graphs.py on a login node -- it is a multi-hour job over
# every slide. Always qsub.
#
#     qsub scripts/precompute.sh
#     qsub -v ARMS="pw-knn hg-knn hg-radius" scripts/precompute.sh
#     qsub -v RADIUS=25 -v OUT=$SC/graph_cache_r25 scripts/precompute.sh
#
# RADIUS changes a GEOMETRY_KEY, so it needs its own --out: a cache cannot mix
# radii, and the cohort fingerprint differs, so radii are separate experiments
# that combine_results.py will refuse to pool. That is the point.
#
# Adding an arm tops up existing slides rather than rebuilding them.

#$ -N precompute
#$ -l h_rt=6:0:0
#$ -l mem=16G             # per slot
#$ -pe smp 4
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/precompute.$JOB_ID.out
#$ -e /home/ucabim3/Scratch/logs/precompute.$JOB_ID.err

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
mkdir -p /home/ucabim3/Scratch/logs

echo "=== $(date) on $(hostname) ==="
ARMS="${ARMS:-}"
LIMIT="${LIMIT:-}"                 # first N slides only, for a measurement run
OUT="${OUT:-/home/ucabim3/Scratch/graph_cache}"
TOP_N="${TOP_N:-}"                 # regions per slide (seeded random sample)
RADIUS="${RADIUS:-}"               # hg-radius/pw-radius radius in microns
K="${K:-}"                         # neighbours for pw-knn/hg-knn
echo "arms: ${ARMS:-<default>} | limit: ${LIMIT:-none} | out: $OUT"
echo "radius: ${RADIUS:-<default>} | k: ${K:-<default>} | top_n: ${TOP_N:-<default>}"

python -u precompute_graphs.py \
    --cache-root /home/ucabim3/Scratch/cellvit_out \
    --out "$OUT" \
    ${ARMS:+--arms $ARMS} \
    ${TOP_N:+--top-n $TOP_N} \
    ${RADIUS:+--hg-radius-um $RADIUS} \
    ${K:+--k $K} \
    ${LIMIT:+--limit $LIMIT}

echo "=== done: $(date) ==="
