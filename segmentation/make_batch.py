"""Emit a segmentation batch straight from the GDC API, skipping what is done.

Replaces the downloaded gdc_manifest_brca_dx.txt, which is a portal export that
nothing in this repo can regenerate -- so when it goes missing (twice now) the
batch cannot be rebuilt. The API is the authority anyway.

    python make_batch.py --out batch.tsv
    python make_batch.py --out batch.tsv --done ~/Scratch/cellvit_out ~/cellvit_out

Output is "uuid filename size" per line, which is what workstation_segment.sh
reads. --done takes any number of cellvit_out trees: the workstation and Myriad
each hold a different subset, and a slide finished on either is one this batch
must not repeat.

Ordering is deterministic (by slide id) so two machines given the same batch
work through it in the same order. Use --shuffle with a seed when two machines
are running concurrently and should not collide on the same slides.
"""

import argparse
import json
import os
import random
import urllib.request

API = "https://api.gdc.cancer.gov/files"

# Diagnostic (DX) slides only. TCGA also holds frozen-section (TS/BS) slides,
# which are a different tissue preparation and not comparable.
QUERY = {
    "filters": {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id",
                                     "value": ["TCGA-BRCA"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
            {"op": "in", "content": {"field": "experimental_strategy",
                                     "value": ["Diagnostic Slide"]}},
        ],
    },
    "fields": "file_id,file_name,file_size",
    "format": "JSON",
    "size": "20000",
}


def slide_id(file_name):
    """TCGA-A2-A0CQ-01Z-00-DX1.<uuid>.svs -> TCGA-A2-A0CQ-01Z-00-DX1.

    Must match workstation_segment.sh's ID="${FNAME%%.*}" exactly, or the skip
    check there and the filter here disagree and slides get done twice.
    """
    return file_name.split(".")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--done", nargs="*", default=[],
                    help="cellvit_out trees whose slides are already segmented")
    ap.add_argument("--limit", type=int, default=0, help="first N only (0 = all)")
    ap.add_argument("--shuffle", type=int, default=None,
                    help="seed; use different seeds on machines running at once")
    args = ap.parse_args()

    req = urllib.request.Request(
        API, data=json.dumps(QUERY).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        hits = json.load(fh)["data"]["hits"]
    if not hits:
        raise SystemExit("GDC returned no files -- check the query or the network")

    done = set()
    for d in args.done:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            print(f"  note: {d} is not a directory, ignoring")
            continue
        # a directory alone is not evidence: cellvit creates it before it
        # succeeds, so key on the artefact
        found = {e for e in os.listdir(d)
                 if os.path.exists(os.path.join(d, e, "cells_cache.npz"))}
        print(f"  {len(found):>4} already segmented in {d}")
        done |= found

    rows = [(h["file_id"], h["file_name"], h["file_size"]) for h in hits]
    rows.sort(key=lambda r: slide_id(r[1]))
    todo = [r for r in rows if slide_id(r[1]) not in done]
    if args.shuffle is not None:
        random.Random(args.shuffle).shuffle(todo)
    if args.limit:
        todo = todo[:args.limit]

    with open(args.out, "w") as fh:
        for uuid, name, size in todo:
            fh.write(f"{uuid} {name} {size}\n")

    gb = sum(r[2] for r in todo) / 1e9
    print(f"\n{len(rows)} diagnostic slides in TCGA-BRCA")
    print(f"{len(done)} already segmented")
    print(f"wrote {len(todo)} to {args.out} ({gb:.0f}GB to download)")


if __name__ == "__main__":
    main()
