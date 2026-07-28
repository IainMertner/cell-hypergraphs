#!/bin/bash
# =============================================================================
# Emit a GDC manifest for the next N slides you do NOT already have.
#
# The full cohort is manifests/gdc_manifest_brca_dx.txt. Rather than hand-editing
# sample manifests (which is how manifest_100 / manifest_extra / an empty
# manifest_next200 came to exist), derive each batch from the full manifest minus
# whatever is already on disk. The full manifest stays the single source of
# truth and batch files become disposable.
#
# Run on the login node (instant):
#     bash next_batch.sh 200
#     bash next_batch.sh 200 /home/ucabim3/manifest_batch.txt
# then point download_slides.sh at the output.
# =============================================================================

set -euo pipefail

N="${1:-200}"
OUT="${2:-/home/ucabim3/manifest_batch.txt}"
FULL="${FULL_MANIFEST:-/home/ucabim3/gdc_manifest_brca_dx.txt}"
DEST="${SLIDE_DIR:-/home/ucabim3/Scratch/tcga_brca_slides}"

[ -f "$FULL" ] || { echo "ERROR: full manifest not found: $FULL" >&2; exit 1; }

# Already-downloaded = UUID directories that actually CONTAIN a .svs. Checking
# for the file rather than the directory matters: gdc-client creates the
# directory when a download starts, so an interrupted transfer would otherwise
# look complete and the slide would be skipped forever.
done_list=$(mktemp)
trap 'rm -f "$done_list"' EXIT
if [ -d "$DEST" ]; then
    find "$DEST" -name '*.svs' -printf '%h\n' 2>/dev/null \
        | xargs -r -n1 basename | sort -u > "$done_list"
fi

total=$(($(wc -l < "$FULL") - 1))
have=$(wc -l < "$done_list")

# header + the first N records whose UUID is not already present.
# awk rather than `head -n` to cap the count: head exits as soon as it has N
# lines, which SIGPIPEs grep upstream, and `set -o pipefail` turns that into a
# job failure. awk drains the whole stream instead -- irrelevant at ~1k lines.
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
