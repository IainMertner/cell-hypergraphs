#!/bin/bash -l
# =============================================================================
# CellViT++ inference. ONE SLIDE PER ARRAY TASK.
#
# This used to batch slides into chunks, on the theory that fewer, larger tasks
# meant fewer queue waits. That was wrong: SGE schedules array tasks
# INDEPENDENTLY, so 160 tasks queue once, not 160 times -- while the longer
# walltime a chunk needs makes every one of them harder to place. An 8h/20-slide
# chunk sat queued ~14h with 1 of 15 tasks placed; a 3h GPU job for the same
# work fared little better. Short jobs backfill into gaps long ones cannot reach.
#
# One slide (~15 min) against a 1.5h wall is a small, easy-to-place request:
#   - tasks start as GPUs free, so you see steady progress, not all-or-nothing
#   - a failure or overrun costs ONE slide, not a whole chunk
#   - no walltime is wasted by a chunk that finishes early
#
# Set CHUNK_SIZE above 1 to restore batching, but raise h_rt with it.
#
# Safe to resubmit: the skip-if-done check means completed slides cost seconds.
# Better still, point SLIDE_LIST at only the unsegmented slides, so no task
# spends a queue wait just to print "already done":
#
#     find ~/Scratch/tcga_brca_slides -name '*.svs' | sort | while read w; do
#       id=$(basename "$w" .svs | cut -d. -f1)
#       [ -f ~/Scratch/cellvit_out/$id/cells_cache.npz ] || echo "$w"
#     done > ~/Scratch/slide_list.txt
#
# SUBMIT -- one task per slide:
#     N=$(wc -l < ~/Scratch/slide_list.txt)
#     qsub -t 1-$N segmentation/cellvit_chunked.sh
#
# Add -tc to cap how many run at once if you want to leave GPUs for other jobs:
#     qsub -t 1-$N -tc 8 segmentation/cellvit_chunked.sh
# =============================================================================

#$ -N cellvit
#$ -l h_rt=1:30:0
#$ -l mem=32G
#$ -l gpu=1
#$ -wd /home/ucabim3/Scratch
#$ -o /home/ucabim3/Scratch/logs/chunk.$TASK_ID.out
#$ -e /home/ucabim3/Scratch/logs/chunk.$TASK_ID.err

set -uo pipefail          # NOT -e: one bad slide must not kill the whole chunk

SLIDE_LIST=/home/ucabim3/Scratch/slide_list.txt
OUTROOT=/home/ucabim3/Scratch/cellvit_out
# One slide per task. h_rt=1:30:0 gives ~6x the ~15min average, enough for the
# largest slides. If you raise this, raise h_rt to match or tasks die mid-slide.
CHUNK_SIZE="${CHUNK_SIZE:-1}"

# Everything comes from the repo, not from copies in $HOME. SGE runs a SPOOLED
# copy of this file (/var/opt/sge/.../job_scripts/<jobid>), so $0 cannot be used
# to locate siblings -- the path has to be absolute.
REPO=/home/ucabim3/Scratch/cell-hypergraphs
ENV_SH="$REPO/segmentation/cellvit_env.sh"
CACHE_PY="$REPO/segmentation/cache_cells.py"

# FAIL FAST. Without these checks a missing env is not fatal: the script carries
# on, prints "cellvit-inference: command not found" once per slide, exits 0, and
# throws away a multi-hour GPU allocation having segmented nothing. Jobs 36789
# and 38663 both did exactly that after ~14h queue waits.
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
command -v cellvit-inference >/dev/null 2>&1 || {
    echo "FATAL: cellvit-inference not on PATH after sourcing $ENV_SH" >&2
    echo "       (conda env missing or broken?)" >&2; exit 1; }
[ -f "$CACHE_PY" ] || { echo "FATAL: missing $CACHE_PY" >&2; exit 1; }
[ -s "$SLIDE_LIST" ] || { echo "FATAL: $SLIDE_LIST missing or empty" >&2; exit 1; }

TOTAL=$(wc -l < "$SLIDE_LIST")
START=$(( (SGE_TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( START + CHUNK_SIZE - 1 ))
[ "$END" -gt "$TOTAL" ] && END=$TOTAL

echo "=== chunk $SGE_TASK_ID | slides $START-$END of $TOTAL | $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

DONE=0; SKIPPED=0; FAILED=0

for i in $(seq "$START" "$END"); do
    WSI=$(sed -n "${i}p" "$SLIDE_LIST")
    [ -z "$WSI" ] && continue
    SLIDE_ID=$(basename "$WSI" .svs | cut -d. -f1)
    OUTDIR="$OUTROOT/$SLIDE_ID"

    # completion is marked by the CACHE file: the output dir is created before
    # processing starts, so its existence proves nothing
    if [ -f "$OUTDIR/cells_cache.npz" ]; then
        echo "[$i] $SLIDE_ID -- already done, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "----------------------------------------------------------------"
    echo "[$i] $SLIDE_ID  started $(date +%H:%M:%S)"
    mkdir -p "$OUTDIR"

    if cellvit-inference \
            --model SAM \
            --nuclei_taxonomy pannuke \
            --enforce_amp \
            --batch_size 8 \
            --geojson \
            --outdir "$OUTDIR" \
            process_wsi \
            --wsi_path "$WSI"
    then
        python "$CACHE_PY" "$OUTDIR" \
            && DONE=$((DONE + 1)) \
            || { echo "[$i] WARN: caching failed"; FAILED=$((FAILED + 1)); }
    else
        echo "[$i] ERROR: cellvit failed on $SLIDE_ID -- continuing"
        FAILED=$((FAILED + 1))
    fi
    echo "[$i] $SLIDE_ID  finished $(date +%H:%M:%S)"
done

echo
echo "=== chunk $SGE_TASK_ID complete: $(date) ==="
echo "processed $DONE | skipped $SKIPPED | failed $FAILED"
echo "total slides cached so far: $(ls "$OUTROOT"/*/cells_cache.npz 2>/dev/null | wc -l)"
