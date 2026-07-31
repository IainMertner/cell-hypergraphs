#!/bin/bash -l
# =============================================================================
# STAGE 2, SPLIT ACROSS ARRAY TASKS. One seed per task.
#
# This is run_patterns.sh's work divided by seed. It sacrifices nothing:
# _make_folds is a deterministic function of (cohort, folds, seed) and the
# cohort comes from sorted(glob(...)), so seed 1 computed here is bit-identical
# to seed 1 computed inline. Arms stay paired run-for-run.
#
# WHY BOTHER
#   placement   ONE seed is a third of the monolith's work, so h_rt is a third
#               of its 12h (plus headroom) -- 5h. That is the whole point: this
#               is a genuinely smaller request, not the same request relabelled.
#               Array tasks are also scheduled independently, so they trickle in
#               rather than all waiting for one large window.
#               mem stays at 48G -- one seed still loads the entire graph cache.
#   resilience  a walltime kill or a node failure costs ONE seed, not the whole
#               sweep. Previously a 12h job that died at 11h produced nothing.
#   visibility  partial results land as tasks finish, instead of one blind wait.
#
# SUBMIT -- N tasks = N seeds (task 1 -> seed 0, task 2 -> seed 1, ...):
#     qsub -t 1-3 scripts/run_patterns_array.sh
#
# THEN merge. combine_results.py refuses to merge parts whose cohort
# fingerprints differ, so a segmentation job landing mid-sweep cannot silently
# splice two experiments together:
#     python combine_results.py ~/Scratch/results/
#
# NOTE: SGE spools a COPY of this script at submit time, so editing it does not
# affect an already-queued job. Change something? qdel and resubmit.
# =============================================================================

#$ -N pat_seed
#$ -l h_rt=1:0:0
#$ -l mem=48G
#$ -l gpu=1
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/pat_seed.$TASK_ID.out
#$ -e /home/ucabim3/Scratch/logs/pat_seed.$TASK_ID.err

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/segmentation/cellvit_env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }

RESULTS=/home/ucabim3/Scratch/results
mkdir -p /home/ucabim3/Scratch/logs "$RESULTS"

# SGE array tasks are 1-based; seeds are 0-based.
SEED=$(( SGE_TASK_ID - 1 ))

# LEARNING-CURVE MODE. Set SIZES to a space-separated list and the array task
# indexes the LIST rather than the seed -- one cohort size per task, all on the
# same seed. Does macro-F1 rise with n (more slides would help) or sit flat at
# the collapse floor (the limit is representational, and the full 1133-slide
# cohort will not rescue it)?
#
#     qsub -t 1-5 -v SIZES="40 60 80 100 113",RESULT_TAG=curve run_patterns_array.sh
#
# Each point gets its own cohort fingerprint, so combine_results.py will refuse
# to pool them -- correct, they are different experiments. Read them separately.
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
# Patience MUST scale with EPOCHS. Training is full-batch -- one optimiser step
# per epoch -- so with patience 20 a run ends around step 40 regardless of the
# cap. Raising --epochs without raising this tests nothing. ~10% of the cap.
PATIENCE="${PATIENCE:-$(( EPOCHS / 10 > 20 ? EPOCHS / 10 : 20 ))}"
FOLDS="${FOLDS:-5}"
TASK="${TASK:-pattern4}"
RPB="${RPB:-16}"
RESULT_TAG="${RESULT_TAG:-}"
# Arms. Empty = train_patterns.py's DEFAULT_ARMS (pw-knn hg-knn). Passing this
# is how you get hg-radius or an aggregation variant into a run -- adding an arm
# to graphs.ARMS is NOT enough, since the default list is separate.
#     ARMS="pw-knn hg-knn hg-radius"        both constructions
#     ARMS="pw-knn hg-knn hg-knn@sum"       deepsets vs sum on the SAME graphs
# arm@agg selects the aggregation layer; the cache is keyed by construction only.
ARMS="${ARMS:-}"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || echo "WARNING: no GPU visible -- this will not finish in the walltime"
echo "array task $SGE_TASK_ID -> SEED $SEED | task=$TASK folds=$FOLDS" \
     "epochs=$EPOCHS patience=$PATIENCE"

# RESULT_TAG keeps a different configuration in its own directory. Merging a
# 1000-epoch run with a 150-epoch one would pool two different experiments;
# combine_results.py refuses on the epochs mismatch, but separate directories
# make the intent obvious rather than relying on the guard.
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
echo "when all tasks finish:  python combine_results.py $RESULTS"
