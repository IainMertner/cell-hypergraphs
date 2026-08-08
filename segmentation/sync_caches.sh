#!/bin/bash
# Push segmentation caches to Myriad, verify them, and optionally free the local
# copies. Run it whenever, not just at the end of a reservation -- CS home is
# 10GB and 300 caches will fill it.
#
#     bash sync_caches.sh              push and verify
#     bash sync_caches.sh --prune      ...then delete the verified local copies
#
# Needs env.sh sourced, which puts rclone on PATH and sets the SFTP variables.
# It prompts once for your UCL password.
#
# --prune deletes ONLY after `rclone check` reports no differences, so a failed
# or partial transfer cannot cost you caches. no_mpp.txt is always kept: it is
# the record of slides deliberately skipped, and losing it means re-downloading
# them on every future run to rediscover they have no usable MPP.

set -uo pipefail

SRC="${SRC:-$HOME/cellvit_out}"
DST="${DST:-:sftp:Scratch/cellvit_out}"
PRUNE=""
[ "${1:-}" = "--prune" ] && PRUNE=1

command -v rclone >/dev/null 2>&1 || {
    echo "FATAL: rclone not on PATH -- source env.sh first" >&2; exit 1; }
[ -d "$SRC" ] || { echo "FATAL: no such directory: $SRC" >&2; exit 1; }

have=$(find "$SRC" -name cells_cache.npz | wc -l)
echo "local  $SRC  ($have caches, $(du -sh "$SRC" | cut -f1))"
[ "$have" -eq 0 ] && { echo "nothing to push"; exit 0; }

echo
echo "=== push ==="
rclone copy "$SRC" "$DST" -P || { echo "FATAL: copy failed" >&2; exit 1; }

echo
echo "=== verify ==="
# --one-way: Myriad holds caches from other machines that were never here, and
# those are not a discrepancy.
if ! rclone check "$SRC" "$DST" --one-way; then
    echo "FATAL: verify failed -- NOT deleting anything" >&2
    exit 1
fi

if [ -z "$PRUNE" ]; then
    echo
    echo "verified. re-run with --prune to free the local copies."
    exit 0
fi

echo
echo "=== prune ==="
before=$(du -sh "$SRC" | cut -f1)
find "$SRC" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
echo "freed $before, kept $(ls "$SRC" 2>/dev/null | tr '\n' ' ')"
echo
echo "NOTE: workstation_segment.sh skips slides whose .npz is in this directory."
echo "It is now empty, so rebuild the batch from Myriad's done-list before"
echo "restarting, or it will segment everything again."
