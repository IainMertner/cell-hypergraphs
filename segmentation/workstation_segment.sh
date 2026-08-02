#!/bin/bash
# Segment slides on a CS GPU workstation: download -> CellViT -> .npz -> keep.
# No scheduler, no gdc-client, no $USER (the CS session cannot resolve its own
# uid, so anything relying on whoami/getpwuid fails).
#
#     source /scratch0/imertner/env.sh
#     tmux new -s seg
#     bash cell-hypergraphs/segmentation/workstation_segment.sh /scratch0/imertner/batch.tsv
#
# batch.tsv is "uuid filename" per line, from the GDC manifest:
#     awk 'NR>1 {print $1, $2}' gdc_manifest_brca_dx.txt | tail -60 > batch.tsv
#
# Env overrides:
#   WORK  scratch workspace   default: the directory holding batch.tsv
#   KEEP  where .npz files go default: $HOME/cellvit_out
#   N     stop after N slides default: 0 (all)
#
# /scratch0 IS WIPED WHEN THE RESERVATION ENDS, so each .npz is copied to KEEP
# the moment it is written. A crash then costs the slide in flight, not the run.
# Slides and cells.json are deleted after each slide -- 60 slides at 1-2GB would
# fill scratch, and both are reproducible.
#
# Safe to re-run: slides whose .npz is already in KEEP are skipped.

set -uo pipefail          # NOT -e: one bad slide must not kill the run

BATCH="${1:?usage: workstation_segment.sh <batch.tsv>}"
[ -f "$BATCH" ] || { echo "FATAL: no such batch file: $BATCH" >&2; exit 1; }
BATCH=$(readlink -f "$BATCH")

WORK="${WORK:-$(dirname "$BATCH")}"
KEEP="${KEEP:-$HOME/cellvit_out}"
N="${N:-0}"
CACHE_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cache_cells.py"

command -v cellvit-inference >/dev/null 2>&1 || {
    echo "FATAL: cellvit-inference not on PATH -- source env.sh first" >&2
    exit 1; }
[ -f "$CACHE_PY" ] || { echo "FATAL: missing $CACHE_PY" >&2; exit 1; }
mkdir -p "$WORK/slides" "$WORK/out" "$KEEP"

echo "batch $BATCH ($(wc -l < "$BATCH") slides)"
echo "work  $WORK"
echo "keep  $KEEP  (already have $(ls "$KEEP"/*/cells_cache.npz 2>/dev/null | wc -l))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

DONE=0; SKIP=0; FAIL=0
while read -r UUID FNAME; do
    [ -z "${UUID:-}" ] && continue
    [ "$N" -gt 0 ] && [ "$DONE" -ge "$N" ] && { echo "reached N=$N"; break; }

    # TCGA-A2-A0CQ-01Z-...svs -> TCGA-A2-A0CQ-01Z, matching what Myriad stores
    ID="${FNAME%%.*}"
    if [ -f "$KEEP/$ID/cells_cache.npz" ]; then
        SKIP=$((SKIP + 1)); continue
    fi

    SVS="$WORK/slides/$FNAME"
    OUTDIR="$WORK/out/$ID"
    echo "=============================================================="
    echo "[$((DONE + FAIL + 1))] $ID  $(date +%H:%M:%S)"

    if [ ! -f "$SVS" ]; then
        curl -fsSL -o "$SVS" "https://api.gdc.cancer.gov/data/$UUID" || {
            echo "  download failed"; FAIL=$((FAIL + 1)); rm -f "$SVS"; continue; }
    fi
    echo "  $(du -h "$SVS" | cut -f1) downloaded, segmenting ..."

    mkdir -p "$OUTDIR"
    if cellvit-inference --model SAM --nuclei_taxonomy pannuke --enforce_amp \
            --batch_size 8 --geojson --outdir "$OUTDIR" \
            process_wsi --wsi_path "$SVS" >"$OUTDIR/cellvit.log" 2>&1 \
       && python "$CACHE_PY" "$OUTDIR" >>"$OUTDIR/cellvit.log" 2>&1
    then
        mkdir -p "$KEEP/$ID"
        cp "$OUTDIR/cells_cache.npz" "$KEEP/$ID/"
        DONE=$((DONE + 1))
        echo "  OK -> $KEEP/$ID/cells_cache.npz ($(du -h "$KEEP/$ID/cells_cache.npz" | cut -f1))"
    else
        FAIL=$((FAIL + 1))
        echo "  FAILED -- last lines of $OUTDIR/cellvit.log:"
        tail -5 "$OUTDIR/cellvit.log" 2>/dev/null | sed 's/^/    /'
        cp "$OUTDIR/cellvit.log" "$KEEP/$ID.failed.log" 2>/dev/null
    fi

    rm -rf "$OUTDIR" "$SVS"
    echo "  done $DONE | failed $FAIL | scratch free $(df -h "$WORK" | tail -1 | awk '{print $4}')"
done < "$BATCH"

echo
echo "=== segmented $DONE | skipped $SKIP | failed $FAIL ==="
echo "$(ls "$KEEP"/*/cells_cache.npz 2>/dev/null | wc -l) caches in $KEEP"
echo "COPY $KEEP TO MYRIAD BEFORE THE RESERVATION ENDS"
