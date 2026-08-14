#!/bin/bash
# Pull slides one at a time from a shared queue and segment them. Run the same
# command on every lab machine; they coordinate through the shared filesystem
# and no machine can be left idle while another still has work.
#
#     bash queue_worker.sh /shared/path/queue
#
# The queue directory holds:
#     shared_pcs.tsv  the batch, "uuid filename size" per line (make_batch.py)
#     claims/<ID>/    created atomically by whichever machine took that slide
#     failed/<ID>     written when a slide could not be segmented
#
# A slide is claimed with mkdir, which is atomic on a shared filesystem and
# fails if the directory exists. That is the whole locking scheme -- no daemon,
# no database, and nothing to start in a particular order. Machines may join or
# leave at any time.
#
# Env:
#   KEEP     where .npz output goes    default: <queue>/../cellvit_out
#   WORK     scratch for downloads     default: /tmp/segqueue.$USER
#            Put this on LOCAL disk if there is any: every machine downloads a
#            1-2GB slide at a time and they are deleted immediately after, so
#            the shared volume gains nothing and pays the network cost.
#   STALE    seconds before a claim whose owner stopped reporting is retaken
#            default: 7200. A machine that is logged out mid-slide leaves its
#            claim behind; without this the slide is never done by anyone.
#   BATCH    path to the batch file  default: <queue>/shared_pcs.tsv
#   MAX_MP / BATCH_SIZE / RAY_WORKER  passed through to workstation_segment.sh
#
# Safe to run twice on the same machine, and safe to re-run after a crash.

set -uo pipefail          # NOT -e: a failed slide must not kill the worker

QUEUE="${1:?usage: queue_worker.sh <queue-dir>}"
QUEUE=$(cd "$QUEUE" && pwd) || { echo "FATAL: no such queue dir: $1" >&2; exit 1; }
BATCH_FILE="${BATCH:-$QUEUE/shared_pcs.tsv}"
TODO="$BATCH_FILE"
CLAIMS="$QUEUE/claims"
FAILED="$QUEUE/failed"
[ -f "$TODO" ] || { echo "FATAL: missing $TODO" >&2; exit 1; }

ME=$(basename "$HOME")
KEEP="${KEEP:-$(dirname "$QUEUE")/cellvit_out}"
WORK="${WORK:-/tmp/segqueue.$ME}"
STALE="${STALE:-7200}"
SEGMENT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/workstation_segment.sh"
[ -f "$SEGMENT" ] || { echo "FATAL: missing $SEGMENT" >&2; exit 1; }

mkdir -p "$CLAIMS" "$FAILED" "$KEEP" "$WORK" || {
    echo "FATAL: cannot create queue dirs under $QUEUE / $KEEP / $WORK" >&2; exit 1; }

HOST=$(hostname -s 2>/dev/null || echo unknown)
TAG="$HOST.$$"
echo "worker $TAG | queue $QUEUE | keep $KEEP | work $WORK"
echo "$(wc -l < "$TODO") slides in $(basename "$TODO")"

# A claim is only meaningful while its owner is alive. The owner touches
# heartbeat every 60s from a background process, so a claim with an old
# heartbeat and no cached output belongs to a machine that went away.
heartbeat() {
    while :; do
        touch "$1/heartbeat" 2>/dev/null || return
        sleep 60
    done
}

claim_is_stale() {
    local d="$1" hb
    hb=$(stat -c %Y "$d/heartbeat" 2>/dev/null) || hb=$(stat -c %Y "$d" 2>/dev/null) || return 1
    [ $(( $(date +%s) - hb )) -gt "$STALE" ]
}

# Take the first slide nobody holds. Returns its line on stdout, or 1 if the
# queue is exhausted.
take_next() {
    local uuid fname size id
    while read -r uuid fname size; do
        [ -n "${uuid:-}" ] || continue
        id="${fname%%.*}"
        [ -f "$KEEP/$id/cells_cache.npz" ] && continue     # done by someone
        [ -e "$FAILED/$id" ] && continue                   # already tried, failed
        if mkdir "$CLAIMS/$id" 2>/dev/null; then
            printf '%s\n' "$HOST $$ $(date +%s)" > "$CLAIMS/$id/owner"
            printf '%s %s %s\n' "$uuid" "$fname" "$size"
            return 0
        fi
        # held by someone else -- unless they are gone
        if claim_is_stale "$CLAIMS/$id"; then
            echo "  reclaiming stale $id (owner: $(cat "$CLAIMS/$id/owner" 2>/dev/null))" >&2
            rm -rf "$CLAIMS/$id" 2>/dev/null
            if mkdir "$CLAIMS/$id" 2>/dev/null; then
                printf '%s\n' "$HOST $$ $(date +%s)" > "$CLAIMS/$id/owner"
                printf '%s %s %s\n' "$uuid" "$fname" "$size"
                return 0
            fi
        fi
    done < "$TODO"
    return 1
}

n_done=0; n_failed=0
while :; do
    LINE=$(take_next) || { echo; echo "queue exhausted -- nothing left to claim"; break; }
    set -- $LINE
    ID="${2%%.*}"
    echo
    echo "=== $(date +%H:%M:%S) $TAG -> $ID ==="

    heartbeat "$CLAIMS/$ID" &
    HB=$!

    printf '%s\n' "$LINE" > "$WORK/one.tsv"
    KEEP="$KEEP" WORK="$WORK" bash "$SEGMENT" "$WORK/one.tsv"

    kill "$HB" 2>/dev/null; wait "$HB" 2>/dev/null

    if [ -f "$KEEP/$ID/cells_cache.npz" ]; then
        n_done=$((n_done + 1))
        rm -rf "$CLAIMS/$ID"                    # done is recorded by the cache
        echo "  OK ($n_done done, $n_failed failed on this machine)"
    elif grep -qxF "$ID" "$KEEP/no_mpp.txt" 2>/dev/null \
      || grep -qxF "$ID" "$KEEP/too_big.txt" 2>/dev/null; then
        # deliberately skipped, not a failure: leave the claim so no other
        # machine spends 20 minutes rediscovering the same thing
        echo "  skipped (no mpp / too large) -- recorded, not retried"
    else
        n_failed=$((n_failed + 1))
        printf '%s %s\n' "$(date -Is)" "$TAG" > "$FAILED/$ID"
        rm -rf "$CLAIMS/$ID"
        echo "  FAILED ($n_done done, $n_failed failed on this machine)"
    fi
done

echo
echo "worker $TAG finished: $n_done segmented, $n_failed failed"
echo "queue state: $(ls "$CLAIMS" 2>/dev/null | wc -l) in flight, \
$(ls "$FAILED" 2>/dev/null | wc -l) failed, \
$(find "$KEEP" -name cells_cache.npz 2>/dev/null | wc -l) cached"
