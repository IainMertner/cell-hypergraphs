#!/bin/bash -l
# STAGE 2 split by seed, one seed per array task.
#
# _make_folds is deterministic in (cohort, folds, seed), so seed N computed here
# is bit-identical to seed N computed inline -- splitting costs nothing and buys
# easier placement plus resilience (a failure loses one seed, not the sweep).
#
# NEVER run train_patterns.py on a login node. Always qsub.
#
#     qsub -t 1-3 scripts/run_patterns_array.sh
#     python combine_results.py ~/Scratch/results/
#
# LEARNING CURVE: set SIZES and the task index selects a cohort size instead of
# a seed. Each point gets its own cohort fingerprint, so combine_results.py
# refuses to pool them -- read them separately.
#
#     qsub -t 1-5 -v SIZES="40 60 80 100 113",RESULT_TAG=curve scripts/run_patterns_array.sh

#$ -N pat_seed
#$ -l h_rt=1:0:0
#$ -l mem=48G
#$ -l gpu=1
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/pat_seed.$TASK_ID.out
#$ -e /home/ucabim3/Scratch/logs/pat_seed.$TASK_ID.err

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }

RESULTS=/home/ucabim3/Scratch/results
mkdir -p /home/ucabim3/Scratch/logs "$RESULTS"

SEED=$(( SGE_TASK_ID - 1 ))       # SGE tasks are 1-based, seeds 0-based

SUBSAMPLE=""
if [ -n "${SIZES:-}" ]; then
    SIZE_ARR=($SIZES)
    IDX=$(( SGE_TASK_ID - 1 ))
    if [ "$IDX" -ge "${#SIZE_ARR[@]}" ]; then
        echo "FATAL: task $SGE_TASK_ID has no entry in SIZES=($SIZES)" >&2
        exit 1
    fi
    SUBSAMPLE="--subsample ${SIZE_ARR[$IDX]}"
    SEED="${CURVE_SEED:-0}"
    echo "LEARNING CURVE: task $SGE_TASK_ID -> ${SIZE_ARR[$IDX]} slides, seed $SEED"
fi

EPOCHS="${EPOCHS:-150}"
# patience must scale with the cap: training is full-batch, so an epoch is one
# optimiser step and patience 20 ends a run around step 40 whatever the cap is
PATIENCE="${PATIENCE:-$(( EPOCHS / 10 > 20 ? EPOCHS / 10 : 20 ))}"
FOLDS="${FOLDS:-5}"
TASK="${TASK:-pattern4}"
RPB="${RPB:-16}"
RESULT_TAG="${RESULT_TAG:-}"
# empty = train_patterns.py's DEFAULT_ARMS. arm@agg picks the aggregation layer;
# the cache is keyed by construction only.
#     ARMS="pw-knn hg-knn hg-radius"        ARMS="pw-knn hg-knn hg-knn@sum"
ARMS="${ARMS:-}"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || echo "WARNING: no GPU visible -- this will not finish in the walltime"
echo "task $SGE_TASK_ID -> SEED $SEED | task=$TASK folds=$FOLDS" \
     "epochs=$EPOCHS patience=$PATIENCE arms=${ARMS:-<default>}"

# RESULT_TAG keeps a different configuration in its own directory
OUT_DIR="$RESULTS${RESULT_TAG:+/$RESULT_TAG}"
mkdir -p "$OUT_DIR"
RESULT_NAME="${TASK}_seed${SEED}"
[ -n "$SUBSAMPLE" ] && RESULT_NAME="${TASK}_n${SIZE_ARR[$(( SGE_TASK_ID - 1 ))]}_seed${SEED}"

python -u train_patterns.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --labels /home/ucabim3/Scratch/til_indices.csv \
    --task "$TASK" \
    --regions-per-batch "$RPB" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --folds "$FOLDS" \
    --seed "$SEED" \
    ${ARMS:+--arms $ARMS} \
    $SUBSAMPLE \
    --save-results "$OUT_DIR/${RESULT_NAME}.json"

echo "=== done: $(date) ==="
echo "when all tasks finish:  python combine_results.py $OUT_DIR"
