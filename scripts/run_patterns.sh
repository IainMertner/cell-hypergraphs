#!/bin/bash -l
# =============================================================================
# STAGE 2: train the MIL pattern classifier off the precomputed graph cache.
#
# THIS IS THE PROPER RUN -- the configuration whose numbers you can defend.
# For a quick does-it-work check use run_patterns_fast.sh instead; do not report
# anything from that one.
#
# WHY THESE SETTINGS
#   --seeds 3         3 seeds x 5 folds = 15 runs/arm. Each seed is a fresh
#                     patient->fold assignment AND init, so the spread reflects
#                     which patients land in test -- the dominant variance at
#                     n~113 -- not just initialisation noise. The corrected
#                     t-test then runs on df=14, which is thin but usable; at 5
#                     runs it is df=4 and near-meaningless.
#
#                     10 seeds (50 runs) would be better but does NOT fit. The
#                     smoke test measured ~670s/run (pw) and ~845s/run (hg) at
#                     10 epochs on 4 CPU cores -- about 1.3s per slide
#                     forward+backward, 13x slower than a synthetic benchmark
#                     suggested. Scaled to 5 folds, ~50 epochs and 50 runs that
#                     is ~20h+ even on GPU. Raise seeds only after a completed
#                     run tells you the real per-run cost at these settings.
#   --epochs 150      patience is 20, so early stopping decides when to stop.
#                     A LOW epoch cap is not a neutral saving: pw-knn and hg-knn
#                     may converge at different rates, so truncating can
#                     penalise one arm and manufacture a null. Set the cap high
#                     enough that early stopping, not the cap, ends every run.
#   all regions       No region subsampling. Regions are ranked by CELL count,
#                     not TIL content, and the label is a whole-slide
#                     arrangement property -- dropping regions can turn a
#                     multifocal slide into a focal-looking one. That would
#                     damage the exact signal being measured.
#
# COST, from the smoke test on REAL slides (4 CPU cores, 10 epochs, 3 folds):
#     pw-knn  ~670 s/run      hg-knn  ~845 s/run
# i.e. ~1.3s per slide forward+backward. A synthetic benchmark suggested 98ms;
# the real slides are ~13x heavier, so trust the measured figure. GPU was 5.7x
# (pw) and 10.7x (hg) faster on that synthetic data -- treat as indicative.
#
# 15 runs/arm at these settings should land inside the 12h wall on GPU. It will
# NOT finish on CPU, which is why gpu=1 is required rather than optional. Watch
# the ETA in the log after run 1 and qdel early if it is going to overrun.
#
# REQUEST SHAPE is tuned for placement:
#   mem=48G     on ONE slot -- no -pe. Asking for N cores means N FREE CORES ON
#               THE SAME GPU HOST, a far harder constraint than the same memory
#               on one slot. An otherwise-identical job with -pe smp 4 sat
#               unplaced for 24h while a CPU job asking DOUBLE the memory
#               started in minutes.
#   h_rt=12:0:0 headroom over the estimate, and deliberately NOT trimmed to help
#               placement. A shorter request places sooner but a run that gets
#               killed at the wall produces nothing -- which has already happened
#               twice here. Comparable jobs elsewhere on this cluster ask 5h/32G
#               and place in minutes; this workload is heavier than those, so the
#               wait is the cost of a run that can actually finish. If it still
#               overruns, cut SEEDS rather than raise this further.
#   mem=48G     likewise sized for the job, not for the queue.
#   no -ac      `-ac allow=X` NARROWS eligibility to one node class. Free GPUs
#               were seen on both E and L nodes, and jobs here have run on both
#               with no allow= flag, so restricting would shrink the pool.
#
# SUBMIT:
#     qsub scripts/run_patterns.sh
#
# The progress log prints seconds-per-run and an ETA from the first run onward,
# so you can tell early whether it will fit inside the walltime.
#
# NOTE: SGE spools a COPY of this script at submit time, so editing it does not
# affect an already-queued job. Change something? qdel and resubmit.
# =============================================================================

#$ -N patterns
#$ -l h_rt=12:0:0
#$ -l mem=48G
#$ -l gpu=1
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/patterns.out
#$ -e /home/ucabim3/Scratch/logs/patterns.err

# Source from the repo, not a copy in $HOME, and abort if it is not there.
ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/segmentation/cellvit_env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }
mkdir -p /home/ucabim3/Scratch/logs
echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || echo "WARNING: no GPU visible -- this will not finish in the walltime"

# Overridable, but the defaults ARE the defensible configuration. Lower them
# only for a deliberate pilot, and then use run_patterns_fast.sh instead so the
# distinction stays visible in the logs.
SEEDS="${SEEDS:-3}"
EPOCHS="${EPOCHS:-150}"
FOLDS="${FOLDS:-5}"
TASK="${TASK:-pattern4}"
# Region grouping is a pure memory/speed knob -- region boundaries are never
# split, so it cannot change a result. 16 suits GPU (fewer, larger kernels).
RPB="${RPB:-16}"

echo "PROPER RUN | task=$TASK seeds=$SEEDS folds=$FOLDS epochs=$EPOCHS rpb=$RPB"
echo "  -> $((SEEDS * FOLDS)) runs/arm"

python -u train_patterns.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --labels /home/ucabim3/Scratch/til_indices.csv \
    --task "$TASK" \
    --regions-per-batch "$RPB" \
    --epochs "$EPOCHS" \
    --folds "$FOLDS" \
    --seeds "$SEEDS"

echo "=== done: $(date) ==="
