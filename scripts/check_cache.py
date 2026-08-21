#!/usr/bin/env python
"""Find graph-cache files whose contents are impossible.

Corruption in transit survives unpickling: a .pt loads fine and then holds
indices like 1812012350, which surface much later as a CUDA device-side assert
naming a line that has nothing to do with the cause. This checks the invariant
directly -- every index must address a node that exists -- and names the slides
that fail, so only those need re-copying.

    python scripts/check_cache.py ~/Scratch/gc_final
    python scripts/check_cache.py $SHARED/gc_final --out /tmp/bad.txt

Writes bad slide ids one per line, which rclone takes directly:

    rclone copy :sftp:Scratch/gc_final $SHARED/gc_final \
        --files-from-raw /tmp/bad.txt
"""

import argparse
import glob
import os
import sys

import torch


def bad_reasons(d):
    """Everything wrong with one slide's cache entry. Empty list = sound."""
    out = []
    for arm, bags in (d.get("bags") or {}).items():
        for r, g in enumerate(bags):
            x, struct = g[0], g[1]
            n = x.shape[0]
            if n == 0:
                out.append(f"{arm}[{r}]: no nodes")
                continue
            if not torch.isfinite(x).all():
                out.append(f"{arm}[{r}]: non-finite features")
            if struct.numel() == 0:
                continue
            if struct.min() < 0:
                out.append(f"{arm}[{r}]: negative index {int(struct.min())}")
            # pairwise: both rows are node ids. hypergraph: row 0 is node ids,
            # row 1 indexes hyperedges and is not bounded by the node count.
            rows = struct if arm.startswith("pw-") else struct[:1]
            hi = int(rows.max())
            if hi >= n:
                out.append(f"{arm}[{r}]: index {hi} >= {n} nodes")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--out", default=None, help="write bad slide ids here")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.cache, "*.pt"))
                   if not f.endswith("_params.pt"))
    if not files:
        sys.exit(f"no .pt files in {args.cache}")
    print(f"checking {len(files)} slides in {args.cache}")

    bad = []
    for i, f in enumerate(files, 1):
        sid = os.path.basename(f)[:-3]
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            why = bad_reasons(d)
        except Exception as e:                     # unpickling failure counts
            why = [f"unreadable: {type(e).__name__}: {e}"]
        if why:
            bad.append(sid)
            print(f"  BAD {sid}: {why[0]}"
                  + (f" (+{len(why)-1} more)" if len(why) > 1 else ""))
        if i % 100 == 0:
            print(f"  {i}/{len(files)} | {len(bad)} bad so far", flush=True)

    print(f"\n{len(bad)} of {len(files)} slides are corrupt")
    if args.out and bad:
        with open(args.out, "w") as fh:
            for sid in bad:
                fh.write(f"{sid}.pt\n")
        print(f"wrote {args.out} -- pass it to rclone --files-from-raw")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
