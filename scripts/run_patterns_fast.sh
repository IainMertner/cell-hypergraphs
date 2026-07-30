#!/bin/bash -l
# =============================================================================
# SMOKE TEST for stage 2. NOT A RESULT -- do not report numbers from this.
#
# Purpose: prove the pipeline runs end to end on real data and measure the true
# per-run cost, without waiting for a GPU. Runs on CPU, so it schedules in
# minutes rather than the 24h+ a gpu=1 request has waited here.
#
# WHY ITS NUMBERS ARE NOT USABLE
#   --seeds 1     5 runs/arm puts the corrected t-test on df=4. You will get a
#                 p-value; it will not mean much.
#   --epochs 30   patience is 20, so early stopping can barely engage. Runs are
#                 cut off possibly mid-convergence, and pw-knn / hg-knn may
#                 converge at different rates -- so a null here could be an
#                 artefact of truncation rather than a finding about topology.
#
# WHAT IT IS GOOD FOR
#   - confirming train_patterns.py completes on the real cache (the last full
#     CPU attempt ran 30 CPU hours and only ever printed the abundance line)
#   - the macroF1 column, showing whether models collapse to majority under the
#     real class imbalance
#   - seconds-per-run from the progress log, to size the proper GPU run from a
#     measurement instead of an estimate
#
# SUBMIT:
#     qsub scripts/run_patterns_fast.sh
#
# Then, for numbers you can actually use:
#     qsub scripts/run_patterns.sh          # 50 runs/arm, GPU, ~2-4h
# =============================================================================

#$ -N pat_fast
#$ -l h_rt=2:0:0
#$ -l mem=32G
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/patterns_fast.out
#$ -e /home/ucabim3/Scratch/logs/patterns_fast.err

# No gpu=1 on purpose: this exists to start immediately.

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/segmentation/cellvit_env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }
mkdir -p /home/ucabim3/Scratch/logs
echo "=== $(date) on $(hostname) ==="

SEEDS="${SEEDS:-1}"
EPOCHS="${EPOCHS:-30}"
FOLDS="${FOLDS:-5}"
TASK="${TASK:-pattern4}"
# rpb=4 measured ~1.3x faster than 16 on CPU (cache locality). Cannot change a
# result -- region boundaries are never split.
RPB="${RPB:-4}"

echo "SMOKE TEST -- NOT A RESULT"
echo "  task=$TASK seeds=$SEEDS folds=$FOLDS epochs=$EPOCHS rpb=$RPB"
echo "  -> $((SEEDS * FOLDS)) runs/arm (proper run uses 50)"

python -u train_patterns.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --labels /home/ucabim3/Scratch/til_indices.csv \
    --task "$TASK" \
    --regions-per-batch "$RPB" \
    --epochs "$EPOCHS" \
    --folds "$FOLDS" \
    --seeds "$SEEDS"

echo
echo "REMINDER: smoke test only. Report nothing from this file."
echo "Read seconds-per-run above, then: qsub scripts/run_patterns.sh"
echo "=== done: $(date) ==="
