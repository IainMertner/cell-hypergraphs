#!/bin/bash -l
# Masked cell-type prediction (train_masked.py). Per-cell supervision, no MIL --
# measures the encoders only.
#
# NEVER run train_masked.py on a login node. Always qsub.
#
#     qsub scripts/run_masked.sh
#     qsub -v FEATURES=none,ARMS="pw-knn hg-radius@sum2",REGIONS=20 scripts/run_masked.sh
#
# The paired test is off unless COMPARE_TO names a reference arm:
#     qsub -l h_rt=2:0:0 -v REGIONS=50,ARMS="pw-radius@gin+deg hg-radius@deepsets2",\
#     COMPARE_TO=pw-radius@gin+deg,CAPACITY_REF=hg-radius@deepsets2 scripts/run_masked.sh
#
# CPU-only: the models are small enough that a GPU buys minutes and costs an
# hour in the queue. Add -l gpu=1 on the command line if you ever need one.

#$ -N masked
#$ -l h_rt=0:20:0
#$ -l mem=8G
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/masked.out
#$ -e /home/ucabim3/Scratch/logs/masked.err

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }
mkdir -p /home/ucabim3/Scratch/logs

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "CPU only"

ARMS="${ARMS:-pw-knn hg-knn hg-radius}"
REGIONS="${REGIONS:-10}"
SEEDS="${SEEDS:-3}"
EPOCHS="${EPOCHS:-300}"
FEATURES="${FEATURES:-morph}"      # none | type | morph | both
# both off unless set. COMPARE_TO runs the paired test against that arm;
# CAPACITY_REF pins the parameter target so runs stay comparable to each other
COMPARE_TO="${COMPARE_TO:-}"
CAPACITY_REF="${CAPACITY_REF:-}"

echo "arms=$ARMS regions=$REGIONS seeds=$SEEDS epochs=$EPOCHS features=$FEATURES"
echo "compare-to=${COMPARE_TO:-<none>} capacity-ref=${CAPACITY_REF:-<default>}"

python -u train_masked.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --arms $ARMS \
    --regions "$REGIONS" \
    --seeds "$SEEDS" \
    --epochs "$EPOCHS" \
    --features "$FEATURES" \
    ${COMPARE_TO:+--compare-to "$COMPARE_TO"} \
    ${CAPACITY_REF:+--capacity-ref "$CAPACITY_REF"}

echo "=== done: $(date) ==="
