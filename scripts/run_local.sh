#!/bin/bash
# Run a pattern sweep on one machine, sequentially. The local counterpart of
# run_patterns_array.sh, which needs SGE and hardcodes Myriad paths.
#
#     bash scripts/run_local.sh <tag> <first> <last> [VAR=VALUE ...]
#
#     bash scripts/run_local.sh til 1 50 FOLDS=10 TASK=arrangement \
#         LABELS=$SHARED/til_indices.csv \
#         ARMS="pw-radius@gin hg-radius@deepsets2" \
#         GRAPH_CACHE=$SHARED/gc_final RESULTS=$SHARED/results
#
# Task ids map to (seed, fold) exactly as FOLD_SPLIT does there, so a run split
# across machines -- or half here and half on Myriad -- reassembles in
# combine_results.py without knowing where anything ran.
#
# Resumable: a task whose result JSON exists is skipped, so re-running after an
# interruption costs nothing. Point several machines at one RESULTS only if
# they are on DIFFERENT tag/task combinations; there is no claim protocol here.

set -uo pipefail

TAG="${1:?usage: run_local.sh <tag> <first> <last> [VAR=VALUE ...]}"
FIRST="${2:?need a first task id}"
LAST="${3:?need a last task id}"
shift 3
for kv in "$@"; do export "$kv"; done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${RESULTS:-$REPO/results}"
FOLDS="${FOLDS:-10}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-$(( EPOCHS / 10 > 20 ? EPOCHS / 10 : 20 ))}"
TASK="${TASK:-pattern4}"
LABELS="${LABELS:?set LABELS=/path/to/labels.csv}"
GRAPH_CACHE="${GRAPH_CACHE:?set GRAPH_CACHE=/path/to/cache}"
OUT_DIR="$RESULTS/$TAG"
mkdir -p "$OUT_DIR"

command -v python >/dev/null || { echo "FATAL: no python -- source env.sh" >&2; exit 1; }
echo "$TAG | tasks $FIRST-$LAST | folds $FOLDS | cache $GRAPH_CACHE"
echo "results -> $OUT_DIR"

for T in $(seq "$FIRST" "$LAST"); do
    SEED=$(( (T - 1) / FOLDS ))
    FOLD=$(( (T - 1) % FOLDS ))
    NAME="${TASK}${LABEL_COL:+_$LABEL_COL}_seed${SEED}_fold${FOLD}"
    if [ -s "$OUT_DIR/$NAME.json" ]; then
        echo "  [$T] $NAME -- already done"
        continue
    fi
    echo
    echo "=== [$T/$LAST] seed $SEED fold $FOLD  $(date +%H:%M:%S) ==="
    python -u "$REPO/train_patterns.py" \
        --graph-cache "$GRAPH_CACHE" \
        --labels "$LABELS" \
        --task "$TASK" \
        --folds "$FOLDS" \
        --seed "$SEED" \
        --fold "$FOLD" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --regions-per-batch "${RPB:-16}" \
        ${ARMS:+--arms $ARMS} \
        ${LABEL_COL:+--label-col "$LABEL_COL"} \
        ${MIN_CLASS:+--min-class $MIN_CLASS} \
        ${PW_LAYERS:+--pw-layers $PW_LAYERS} \
        ${HIDDEN:+--hidden $HIDDEN} \
        ${LR:+--lr $LR} \
        ${BATCH:+--batch-size $BATCH} \
        ${REGION_DIM:+--region-dim $REGION_DIM} \
        ${ATT_DIM:+--att-dim $ATT_DIM} \
        ${AB_SKIP:+--abundance-skip} \
        ${PROGRESS:+--progress-every $PROGRESS} \
        --save-results "$OUT_DIR/$NAME.json"
done

echo
echo "done. combine with:  python combine_results.py $OUT_DIR"
