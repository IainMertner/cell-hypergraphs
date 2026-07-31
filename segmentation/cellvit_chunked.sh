#!/bin/bash -l
# CellViT++ inference, one slide per array task.
#
# SGE schedules array tasks independently, so 160 tasks queue once, not 160
# times -- batching them only made the walltime longer and placement harder.
#
# Point SLIDE_LIST at unsegmented slides only:
#     find ~/Scratch/tcga_brca_slides -name '*.svs' | sort | while read w; do
#       id=$(basename "$w" .svs | cut -d. -f1)
#       [ -f ~/Scratch/cellvit_out/$id/cells_cache.npz ] || echo "$w"
#     done > ~/Scratch/slide_list.txt
#
#     N=$(wc -l < ~/Scratch/slide_list.txt)
#     qsub -t 1-$N segmentation/cellvit_chunked.sh
#
# Safe to resubmit: completed slides are skipped in seconds.

#$ -N cellvit
#$ -l h_rt=1:30:0
#$ -l mem=32G
#$ -l gpu=1
#$ -wd /home/ucabim3/Scratch
#$ -o /home/ucabim3/Scratch/logs/chunk.$TASK_ID.out
#$ -e /home/ucabim3/Scratch/logs/chunk.$TASK_ID.err

set -uo pipefail          # NOT -e: one bad slide must not kill the chunk

SLIDE_LIST=/home/ucabim3/Scratch/slide_list.txt
OUTROOT=/home/ucabim3/Scratch/cellvit_out
CHUNK_SIZE="${CHUNK_SIZE:-1}"   # raise h_rt to match if you raise this

# SGE runs a SPOOLED copy of this file, so $0 cannot locate siblings -- absolute
# paths required.
REPO=/home/ucabim3/Scratch/cell-hypergraphs
ENV_SH="$REPO/env.sh"
CACHE_PY="$REPO/segmentation/cache_cells.py"

# Fail fast: without these a missing env is not fatal, and the job burns its GPU
# allocation printing "command not found" once per slide before exiting 0.
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

    # keyed on the CACHE file: the output dir is created before processing starts
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
