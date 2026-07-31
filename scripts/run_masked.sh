#!/bin/bash -l
# =============================================================================
# DIAGNOSTIC: masked cell-type prediction (train_masked.py).
#
# *** NEVER run train_masked.py directly on a login node. *** ALWAYS qsub.
#
# Per-CELL supervision, so a handful of regions gives thousands of labels --
# unlike the slide-level task, sample size is not the constraint. No MIL, no
# attention pooling: this measures the ENCODERS and nothing else.
#
# SUBMIT:
#     qsub scripts/run_masked.sh
#     qsub -v ARMS="pw-knn hg-radius hg-radius@sum",REGIONS=20 scripts/run_masked.sh
# =============================================================================

#$ -N masked
#$ -l h_rt=2:0:0
#$ -l mem=32G
#$ -l gpu=1
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/masked.out
#$ -e /home/ucabim3/Scratch/logs/masked.err

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/segmentation/cellvit_env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }
mkdir -p /home/ucabim3/Scratch/logs

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || echo "no GPU -- will be slow"

ARMS="${ARMS:-pw-knn hg-knn hg-radius}"
REGIONS="${REGIONS:-10}"
SEEDS="${SEEDS:-3}"
EPOCHS="${EPOCHS:-300}"
# morph = label NOT in the input (non-circular, the one to believe).
# type/both put the one-hot label in the features, reducing the task to reading
# a neighbourhood type histogram -- which favours sum aggregation for free.
FEATURES="${FEATURES:-morph}"   # none|type|morph|both

echo "arms=$ARMS regions=$REGIONS seeds=$SEEDS epochs=$EPOCHS features=$FEATURES"

python -u train_masked.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --arms $ARMS \
    --regions "$REGIONS" \
    --seeds "$SEEDS" \
    --epochs "$EPOCHS" \
    --features "$FEATURES"

echo "=== done: $(date) ==="
