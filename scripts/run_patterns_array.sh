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
# A different target needs no re-precompute -- the graph cache is label-free:
#     qsub -t 1-5 -v TASK=auto,LABELS=~/Scratch/er_labels.csv,LABEL_COL=label,\
#     RESULT_TAG=er scripts/run_patterns_array.sh
#
# LEARNING CURVE: set SIZES and the task index selects a cohort size instead of
# a seed. Each point gets its own cohort fingerprint, so combine_results.py
# refuses to pool them -- read them separately.
#
#     qsub -t 1-5 -v SIZES="40 60 80 100 113",RESULT_TAG=curve scripts/run_patterns_array.sh

#$ -N pat_seed
#$ -l h_rt=1:0:0
# 16G not 48G: 48 was set when regions-per-batch was 16 and training was
# full-batch. A 48G single-slot request is harder for the scheduler to
# place than the GPU, and jobs sat unqueued for hours because of it.
#$ -l mem=16G
# No GPU. Capacity-matched models are ~10^4 parameters over a few thousand nodes
# per region, which a CPU handles in minutes -- but a gpu=1 request queued for
# hours to days behind real GPU work. The wait dwarfed the runtime.
# Add -l gpu=1 on the command line if a sweep ever outgrows this.
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
# $JOB_ID as well as $TASK_ID: without it, task 1 of every pat_seed job
# overwrites task 1 of the last one, and you cannot tell which sweep a log
# belongs to after the fact
#$ -o /home/ucabim3/Scratch/logs/pat_seed.$JOB_ID.$TASK_ID.out
#$ -e /home/ucabim3/Scratch/logs/pat_seed.$JOB_ID.$TASK_ID.err

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
# Any categorical slide-level column: TASK=auto reads LABEL_COL from LABELS.
# Build the CSV with make_labels.py, which joins a patient-level clinical
# table onto the segmented slides and reports the class balance first.
LABELS="${LABELS:-/home/ucabim3/Scratch/til_indices.csv}"
LABEL_COL="${LABEL_COL:-}"
MIN_CLASS="${MIN_CLASS:-}"
RPB="${RPB:-16}"
RESULT_TAG="${RESULT_TAG:-}"
BLEND="${BLEND:-}"                # set to 1 for the family-blending ablation
STAR_LAYERS="${STAR_LAYERS:-}"
PW_LAYERS="${PW_LAYERS:-}"          # pairwise depth; 4 reach-matches a hypergraph arm     # @star depth; 4 matches a 2-layer hypergraph arm
# capacity knobs. HIDDEN sizes the encoder; REGION_DIM/ATT_DIM size the
# pool+head, which HIDDEN does not touch at all
HIDDEN="${HIDDEN:-}"
REGION_DIM="${REGION_DIM:-}"
ATT_DIM="${ATT_DIM:-}"
# abundance skip: a DIFFERENT experiment (does structure add given
# composition), not a fix to the default one. PATH_DROP zeroes one path at
# random each step -- SYMMETRIC, or the graph-zeroed ablation is measuring
# distribution shift rather than information loss.
AB_SKIP="${AB_SKIP:-}"
PATH_DROP="${PATH_DROP:-}"
# empty = train_patterns.py's DEFAULT_ARMS. arm@agg picks the aggregation layer;
# the cache is keyed by construction only.
#     ARMS="pw-knn hg-knn hg-radius"        ARMS="pw-knn hg-knn hg-knn@sum"
ARMS="${ARMS:-}"

echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "CPU only"
echo "task $SGE_TASK_ID -> SEED $SEED | task=$TASK${LABEL_COL:+ col=$LABEL_COL} folds=$FOLDS" \
     "epochs=$EPOCHS patience=$PATIENCE arms=${ARMS:-<default>}"
echo "labels: $LABELS"

# RESULT_TAG keeps a different configuration in its own directory
OUT_DIR="$RESULTS${RESULT_TAG:+/$RESULT_TAG}"
mkdir -p "$OUT_DIR"
# TASK=auto is the same string for every custom label, so fold LABEL_COL
# into the filename or two different targets overwrite each other
NAME_TAG="$TASK${LABEL_COL:+_$LABEL_COL}"
RESULT_NAME="${NAME_TAG}_seed${SEED}"
[ -n "$SUBSAMPLE" ] && RESULT_NAME="${NAME_TAG}_n${SIZE_ARR[$(( SGE_TASK_ID - 1 ))]}_seed${SEED}"

python -u train_patterns.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --labels "$LABELS" \
    --task "$TASK" \
    --regions-per-batch "$RPB" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --folds "$FOLDS" \
    --seed "$SEED" \
    ${ARMS:+--arms $ARMS} \
    ${LABEL_COL:+--label-col "$LABEL_COL"} \
    ${MIN_CLASS:+--min-class $MIN_CLASS} \
    ${BLEND:+--blend-families} \
    ${STAR_LAYERS:+--star-layers $STAR_LAYERS} \
    ${PW_LAYERS:+--pw-layers $PW_LAYERS} \
    ${HIDDEN:+--hidden $HIDDEN} \
    ${REGION_DIM:+--region-dim $REGION_DIM} \
    ${ATT_DIM:+--att-dim $ATT_DIM} \
    ${AB_SKIP:+--abundance-skip} \
    ${AB_DROP:+--abundance-dropout $AB_DROP} \
    $SUBSAMPLE \
    --save-results "$OUT_DIR/${RESULT_NAME}.json"

echo "=== done: $(date) ==="
echo "when all tasks finish:  python combine_results.py $OUT_DIR"
