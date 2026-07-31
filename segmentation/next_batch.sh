#!/bin/bash
# Emit a GDC manifest for the next N slides not already on disk, derived from the
# full cohort manifest. Batch files stay disposable.
#
#     bash next_batch.sh 200
#     bash next_batch.sh 200 /home/ucabim3/manifest_batch.txt

set -euo pipefail

N="${1:-200}"
OUT="${2:-/home/ucabim3/manifest_batch.txt}"
FULL="${FULL_MANIFEST:-/home/ucabim3/gdc_manifest_brca_dx.txt}"
DEST="${SLIDE_DIR:-/home/ucabim3/Scratch/tcga_brca_slides}"

[ -f "$FULL" ] || { echo "ERROR: full manifest not found: $FULL" >&2; exit 1; }

# Downloaded = UUID directories that CONTAIN a .svs. gdc-client creates the
# directory when the download starts, so checking the directory would skip
# interrupted transfers forever.
done_list=$(mktemp)
trap 'rm -f "$done_list"' EXIT
if [ -d "$DEST" ]; then
    find "$DEST" -name '*.svs' -printf '%h\n' 2>/dev/null \
        | xargs -r -n1 basename | sort -u > "$done_list"
fi

total=$(($(wc -l < "$FULL") - 1))
have=$(wc -l < "$done_list")

# awk rather than `head -n` to cap: head exits early, SIGPIPEs grep, and
# pipefail turns that into a failure.
{
    head -1 "$FULL"
    tail -n +2 "$FULL" | grep -vFf "$done_list" | awk -v n="$N" 'NR <= n'
} > "$OUT"

got=$(($(wc -l < "$OUT") - 1))
echo "full cohort   : $total slides"
echo "already have  : $have"
echo "remaining     : $((total - have))"
echo "this batch    : $got  -> $OUT"
if [ "$got" -lt "$N" ]; then
    echo "(fewer than the $N requested -- that is the rest of the cohort)"
fi
