#!/bin/bash
# Segment slides on a CS GPU workstation: download -> CellViT -> .npz -> keep.
# No scheduler, no gdc-client, no $USER (the CS session cannot resolve its own
# uid, so anything relying on whoami/getpwuid fails).
#
#     source /scratch0/imertner/env.sh
#     tmux new -s seg
#     bash cell-hypergraphs/segmentation/workstation_segment.sh /scratch0/imertner/batch.tsv
#
# batch.tsv is "uuid filename size" per line, from the GDC manifest:
#     awk 'NR>1 {print $1, $2, $4}' gdc_manifest_brca_dx.txt | tail -60 > batch.tsv
#
# The size column is not optional. curl can return 0 having written a TRUNCATED
# file -- observed: 252MB of a 565MB slide, no error -- and CellViT then dies
# with "Unsupported or missing image file" 20 minutes later. Every download is
# checked against the manifest size and retried.
#
# Env overrides:
#   WORK  scratch workspace   default: the directory holding batch.tsv
#   KEEP  where .npz files go default: $HOME/cellvit_out
#   N     stop after N slides default: 0 (all)
#   RAY_WORKER   Ray pool size; needed on machines with few cores
#   BATCH_SIZE   patch batch; lower on cards with <16GB (default 8)
#   MAX_MP       skip slides above this many megapixels (default 0 = no limit)
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
# CellViT sizes its Ray pool as int(available_cpus / ray_remote_cpus) and
# reserves 2 cores, so on a 6-core machine it floors to zero and dies with
# ZeroDivisionError. RAY_WORKER overrides it. Keep it <= cpu_count-2:
# overwrite_ray_worker recomputes ray_remote_cpus = (cpu_count-2)/ray_worker,
# and a larger value floors to zero again.
RAY_WORKER="${RAY_WORKER:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# CellViT accumulates detected cells for the whole slide in RAM, so peak
# memory scales with pixel count. On a machine that cannot hold the largest
# slides, MAX_MP skips them BEFORE the 30-90 minutes it takes to fail, and
# records them for a machine with more memory.
MAX_MP="${MAX_MP:-0}"
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

fetch() {   # uuid dest expected_bytes -- resume-and-verify, 3 attempts
    local uuid="$1" dest="$2" want="$3" got
    for attempt in 1 2 3; do
        curl -fsSL --retry 3 --retry-delay 5 -C - \
             -o "$dest" "https://api.gdc.cancer.gov/data/$uuid"
        got=$(stat -c%s "$dest" 2>/dev/null || echo 0)
        [ -n "$want" ] && [ "$want" -gt 0 ] || return 0     # no size to check
        [ "$got" -eq "$want" ] && return 0
        echo "  attempt $attempt: got $got of $want bytes, retrying"
        sleep 5
    done
    return 1
}


# CellViT runs on Ray, and a slide killed by the OOM reaper leaves its workers
# and object store behind holding GBs. The NEXT slide then starts with too
# little memory and fails too, so one oversized slide takes the rest of the
# batch with it. Reap them between slides -- this is what makes the run safe to
# leave unattended.
reap_ray() {
    ray stop --force >/dev/null 2>&1
    pkill -u "$(id -u)" -f "ray::"  >/dev/null 2>&1
    pkill -u "$(id -u)" -f raylet   >/dev/null 2>&1
    pkill -u "$(id -u)" -f plasma_store >/dev/null 2>&1
    sleep 3
}

DONE=0; SKIP=0; FAIL=0
while read -r UUID FNAME SIZE; do
    [ -z "${UUID:-}" ] && continue
    [ "$N" -gt 0 ] && [ "$DONE" -ge "$N" ] && { echo "reached N=$N"; break; }

    # TCGA-A2-A0CQ-01Z-...svs -> TCGA-A2-A0CQ-01Z, matching what Myriad stores
    ID="${FNAME%%.*}"
    if [ -f "$KEEP/$ID/cells_cache.npz" ]; then
        SKIP=$((SKIP + 1)); continue
    fi
    # already known to lack MPP -- do not re-download it on every restart
    if [ -f "$KEEP/no_mpp.txt" ] && grep -qxF "$ID" "$KEEP/no_mpp.txt"; then
        SKIP=$((SKIP + 1)); continue
    fi

    SVS="$WORK/slides/$FNAME"
    OUTDIR="$WORK/out/$ID"
    echo "=============================================================="
    echo "[$((DONE + FAIL + 1))] $ID  $(date +%H:%M:%S)"

    # curl is silent, so without this there is nothing between the header and
    # the "downloaded" line for several minutes, which reads as a hang
    echo "  downloading $(( ${SIZE:-0} / 1000000 ))MB ..."
    if ! fetch "$UUID" "$SVS" "${SIZE:-0}"; then
        echo "  download failed after 3 attempts -- skipping"
        FAIL=$((FAIL + 1)); rm -f "$SVS"; continue
    fi
    # Reject slides with no usable MPP BEFORE spending GPU time. A truncated
    # Aperio header carries neither MPP nor magnification, CellViT then fails
    # anyway, and guessing 40x would silently rescale every micron-denominated
    # parameter for that slide.
    META=$(python "$(dirname "$CACHE_PY")/check_mpp.py" "$SVS" 2>/dev/null)
    if [ -z "$META" ]; then
        echo "  SKIPPED: no usable MPP in slide metadata"
        echo "$ID" >> "$KEEP/no_mpp.txt"
        SKIP=$((SKIP + 1)); rm -f "$SVS"; continue
    fi
    # Runtime tracks PIXEL COUNT, not file size -- J2K compression varies by an
    # order of magnitude, so a 51MB file held 696MP while a 565MB one held less.
    # Logged per slide because it is the only thing that predicts how long a
    # slide will take.
    MPP=${META% *}; MP=${META#* }
    if [ "$MAX_MP" -gt 0 ] && [ "${MP%%.*}" -gt "$MAX_MP" ]; then
        echo "  SKIPPED: ${MP}MP exceeds MAX_MP=$MAX_MP -- needs more memory than this host"
        echo "$ID" >> "$KEEP/too_big.txt"
        SKIP=$((SKIP + 1)); rm -f "$SVS"; continue
    fi
    echo "  $(du -h "$SVS" | cut -f1) downloaded (size verified), mpp $MPP, ${MP}MP, segmenting ..."

    # No --geojson: cache_cells.py reads cells.json, and building the geojson
    # costs GBs of RAM on a large slide for a file nothing downstream reads.
    mkdir -p "$OUTDIR"
    if cellvit-inference ${RAY_WORKER:+--ray_worker "$RAY_WORKER"} --model SAM --nuclei_taxonomy pannuke --enforce_amp \
            --batch_size "$BATCH_SIZE" --outdir "$OUTDIR" \
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
    reap_ray
    echo "  memory free: $(free -g 2>/dev/null | awk '/^Mem:/{print $7}')GB"
    echo "  done $DONE | failed $FAIL | scratch free $(df -h "$WORK" | tail -1 | awk '{print $4}')"
done < "$BATCH"

echo
echo "=== segmented $DONE | skipped $SKIP | failed $FAIL ==="
echo "$(ls "$KEEP"/*/cells_cache.npz 2>/dev/null | wc -l) caches in $KEEP"
[ -s "$KEEP/too_big.txt" ] && echo "$(sort -u "$KEEP/too_big.txt" | wc -l) slide(s) skipped as too large -- listed in $KEEP/too_big.txt"
[ -s "$KEEP/no_mpp.txt" ] && echo "$(wc -l < "$KEEP/no_mpp.txt") slide(s) skipped for missing MPP -- listed in $KEEP/no_mpp.txt"
echo "COPY $KEEP TO MYRIAD BEFORE THE RESERVATION ENDS"
