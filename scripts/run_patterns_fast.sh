#!/bin/bash -l
# =============================================================================
# SMOKE TEST for stage 2. NOT A RESULT -- do not report numbers from this.
#
# Purpose: prove the pipeline runs end to end on real data and measure the true
# per-run cost, without waiting for a GPU. Runs on CPU, so it schedules in
# minutes rather than the 24h+ a gpu=1 request has waited here.
#
# WHY ITS NUMBERS ARE NOT USABLE
#   --seeds 1     3 runs/arm puts any significance test on df=2. Meaningless.
#   --folds 3     training folds of ~38 slides instead of ~68 (one fold is held
#                 for validation, so training gets k-2 of k).
#   --epochs 10   patience is 20, so early stopping NEVER engages -- every run is
#                 truncated mid-convergence. pw-knn and hg-knn may converge at
#                 different rates, so a difference here says nothing about
#                 topology, only about which arm learns faster in 10 epochs.
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
#$ -l mem=8G
#$ -pe smp 4
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/patterns_fast.out
#$ -e /home/ucabim3/Scratch/logs/patterns_fast.err

# No gpu=1 on purpose: this exists to start immediately.
#
# DO request cores here, unlike run_patterns.sh. The single-slot request there
# exists because "N free cores ON THE SAME GPU HOST" is what kept a gpu=1 job
# unplaced for 24h. A CPU job has no such constraint -- any node will do, and a
# 4-slot CPU job here placed in minutes. Dropping to one slot cost 4x the
# compute for no scheduling benefit. mem is PER SLOT on Myriad, so 8G x 4 = 32G.

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/segmentation/cellvit_env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }
mkdir -p /home/ucabim3/Scratch/logs
echo "=== $(date) on $(hostname) ==="

# Match torch's thread count to the slots SGE actually granted. Without this
# torch either uses one thread (wasting the allocation) or oversubscribes the
# node (bad citizenship, and the scheduler may throttle it).
export OMP_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
echo "slots granted: ${NSLOTS:-1} (torch threads set to match)"

# A smoke test only has to prove the thing runs and reveal the per-run cost, so
# it is deliberately minimal: 3 folds not 5, 10 epochs not 30. Training folds are
# thinner than the proper run uses, which is fine precisely because these numbers
# are not reportable.
SEEDS="${SEEDS:-1}"
EPOCHS="${EPOCHS:-10}"
FOLDS="${FOLDS:-3}"
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
