#!/bin/bash -l
# Write a graph cache containing only the constructions a sweep uses.
#
# 253 slides of torch.load/torch.save is minutes of solid I/O -- qsub it, do not
# run it on a login node.
#
#     qsub scripts/export_cache.sh
#     qsub -v ARMS="pw-knn hg-knn",OUT=/home/ucabim3/Scratch/graph_cache_knn \
#          scripts/export_cache.sh
#
# Measured on this cohort: the two knn constructions are ~66% of each file, so
# exporting just the radius pair takes the cache from 50GB to roughly 17GB.
# That is the difference between an overnight transfer and an afternoon one.

#$ -N export_cache
#$ -l h_rt=4:0:0
#$ -l mem=16G
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/export_cache.$JOB_ID.out
#$ -e /home/ucabim3/Scratch/logs/export_cache.$JOB_ID.err

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
mkdir -p /home/ucabim3/Scratch/logs

SRC="${SRC:-/home/ucabim3/Scratch/graph_cache}"
OUT="${OUT:-/home/ucabim3/Scratch/graph_cache_radius}"
ARMS="${ARMS:-pw-radius hg-radius}"

echo "=== $(date) on $(hostname) ==="
echo "src  $SRC"
echo "out  $OUT"
echo "arms $ARMS"
df -h "$(dirname "$OUT")" | tail -1

python -u export_cache.py --in "$SRC" --out "$OUT" --arms $ARMS

echo "=== done: $(date) ==="
du -sh "$SRC" "$OUT"
