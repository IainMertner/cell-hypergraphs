#!/bin/bash -l
# =============================================================================
# CellViT++ inference, CHUNKED.
#
# Instead of one array task per slide (98 separate queue waits), each task
# processes a BLOCK of slides sequentially. You queue ~5 times instead of ~98.
#
# Trade-off: a longer walltime request is harder to schedule than a short one,
# so don't go too big. 20 slides x ~15 min average ~= 5 hours of compute; the
# 8 hour walltime below leaves room for a few large slides.
#
# Safe to resubmit: the skip-if-done check means completed slides cost seconds.
#
# SUBMIT (99 slides / 20 per chunk = 5 tasks):
#     qsub -t 1-5 cellvit_chunked.sh
# =============================================================================

#$ -N cellvit_chunk
#$ -l h_rt=8:0:0
#$ -l mem=8G
#$ -l gpu=1
#$ -pe smp 4
#$ -wd /home/ucabim3/Scratch
#$ -o /home/ucabim3/Scratch/logs/chunk.$TASK_ID.out
#$ -e /home/ucabim3/Scratch/logs/chunk.$TASK_ID.err

set -uo pipefail          # NOT -e: one bad slide must not kill the whole chunk

SLIDE_LIST=/home/ucabim3/Scratch/slide_list.txt
OUTROOT=/home/ucabim3/Scratch/cellvit_out
CHUNK_SIZE=20

source /home/ucabim3/cellvit_env.sh

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
        python /home/ucabim3/cache_cells.py "$OUTDIR" \
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
