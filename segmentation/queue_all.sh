#!/bin/bash
# Queue the whole unattended pipeline: segment every slide that is not already
# segmented, then build graphs once that finishes, then optionally train.
#
# Run this ON THE LOGIN NODE -- it only builds a file list and calls qsub, no
# compute of its own. Everything it submits runs on compute nodes.
#
#     bash segmentation/queue_all.sh
#     TRAIN=1 bash segmentation/queue_all.sh          # also queue stage 2
#     ARMS="pw-knn hg-knn" bash segmentation/queue_all.sh
#
# Safe to re-run: slides already carrying cells_cache.npz are skipped, and
# precompute tops up existing slides rather than rebuilding them.

set -euo pipefail

SLIDES=/home/ucabim3/Scratch/tcga_brca_slides
OUTROOT=/home/ucabim3/Scratch/cellvit_out
LIST=/home/ucabim3/Scratch/slide_list.txt
REPO=/home/ucabim3/Scratch/cell-hypergraphs

ARMS="${ARMS:-pw-radius hg-radius hg-radius+semantic}"
TRAIN_ARMS="${TRAIN_ARMS:-pw-radius@gin+deg hg-radius@deepsets2 hg-radius+semantic@deepsets2 hg-radius+semantic@star}"
PRECOMPUTE_HRT="${PRECOMPUTE_HRT:-12:0:0}"   # scales with cohort size
TRAIN="${TRAIN:-}"

mkdir -p /home/ucabim3/Scratch/logs

# Unsegmented only, keyed on cells_cache.npz -- cellvit_chunked.sh writes that
# LAST, so a slide killed mid-inference is correctly treated as still pending
# rather than skipped because its output directory exists.
find "$SLIDES" -name '*.svs' | sort | while read -r w; do
    id=$(basename "$w" .svs | cut -d. -f1)
    [ -f "$OUTROOT/$id/cells_cache.npz" ] || echo "$w"
done > "$LIST"

TOTAL=$(find "$SLIDES" -name '*.svs' | wc -l)
N=$(wc -l < "$LIST")
DONE=$(( TOTAL - N ))
echo "=== $(date) ==="
echo "$TOTAL slides on disk | $DONE already segmented | $N to do"
echo "precompute arms: $ARMS"

HOLD=""
if [ "$N" -gt 0 ]; then
    SEG=$(qsub -terse -t 1-"$N" "$REPO/segmentation/cellvit_chunked.sh" \
          | cut -d. -f1)
    echo "  segmentation array $SEG  (tasks 1-$N)"
    # -hold_jid on an array waits for EVERY task, so precompute cannot start
    # against a half-segmented cohort and bake a partial slide set into the
    # cache fingerprint
    HOLD="-hold_jid $SEG"
else
    echo "  nothing to segment"
fi

PRE=$(qsub -terse $HOLD -l h_rt="$PRECOMPUTE_HRT" -v ARMS="$ARMS" \
      "$REPO/scripts/precompute.sh" | cut -d. -f1)
echo "  precompute $PRE${HOLD:+  (held on $SEG)}"

if [ -n "$TRAIN" ]; then
    # RESULT_TAG keeps this in its own directory. The cohort will have grown, so
    # its fingerprint differs from any earlier run and combine_results.py will
    # (correctly) refuse to pool the two -- they are different experiments.
    TAG="full$(date +%m%d)"
    TR=$(qsub -terse -hold_jid "$PRE" -t 1-5 -l h_rt=6:0:0 \
         -v ARMS="$TRAIN_ARMS",RPB=8,RESULT_TAG="$TAG" \
         "$REPO/scripts/run_patterns_array.sh" | cut -d. -f1)
    echo "  training array $TR  (held on $PRE, results -> results/$TAG)"
fi

echo
echo "watch:    qstat"
echo "logs:     ~/Scratch/logs/"
echo "when done: python combine_results.py ~/Scratch/results${TRAIN:+/$TAG}"
