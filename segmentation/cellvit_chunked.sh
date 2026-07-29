#!/bin/bash -l
# =============================================================================
# CellViT++ inference, CHUNKED.
#
# Instead of one array task per slide (98 separate queue waits), each task
# processes a BLOCK of slides sequentially. You queue ~5 times instead of ~98.
#
# Trade-off: a longer walltime request is much harder to SCHEDULE. A GPU job
# asking 8h needs a big contiguous free window and cannot backfill into the gaps
# that short jobs slot into -- an 8h/20-slide chunk sat queued ~14h on Myriad and
# still had only 1 of 15 tasks placed. 8 slides x ~15 min ~= 2h of compute, so
# the 3h walltime below leaves room for a few large slides while staying easy to
# place. Prefer MORE, SHORTER tasks: they start independently, so you see partial
# progress instead of all-or-nothing.
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
# SUBMIT -- N = ceil(lines in slide_list.txt / CHUNK_SIZE):
#     wc -l ~/Scratch/slide_list.txt        # e.g. 160
#     qsub -t 1-20 cellvit_chunked.sh       # 160 / 8 = 20 tasks
# =============================================================================

#$ -N cellvit_chunk
#$ -l h_rt=2:0:0
#$ -l mem=32G
#$ -l gpu=1
#$ -wd /home/ucabim3/Scratch
#$ -o /home/ucabim3/Scratch/logs/chunk.$TASK_ID.out
#$ -e /home/ucabim3/Scratch/logs/chunk.$TASK_ID.err

set -uo pipefail          # NOT -e: one bad slide must not kill the whole chunk

SLIDE_LIST=/home/ucabim3/Scratch/slide_list.txt
OUTROOT=/home/ucabim3/Scratch/cellvit_out
CHUNK_SIZE=4            # 4 x ~15min ~= 1h against a 2h wall, so a slow slide
                        # cannot blow the chunk. Short tasks also place far more
                        # easily, and the skip-if-done check makes an overrun
                        # cheap: resubmit and only the unfinished slides rerun.

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
