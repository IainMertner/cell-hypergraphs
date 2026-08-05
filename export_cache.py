"""Copy a graph cache keeping only the constructions a sweep actually uses.

precompute_graphs.py writes one .pt per slide holding every construction, so a
cache with four of them is ~3x the size of one with the two you are training on.
That costs nothing on the cluster, where the cache sits next to the job, and
costs a lot when it has to cross a domestic connection.

    python export_cache.py --in graph_cache --out graph_cache_radius \\
        --arms pw-radius hg-radius

Constructions, not arms: the cache is keyed by construction ('hg-radius'), while
a model arm names an aggregation too ('hg-radius@deepsets2'). The @ half is a
model choice and is not cached, so pass the bare names.

Idempotent -- slides already written to --out are skipped, so an interrupted run
can be restarted rather than repeated.
"""

import argparse
import glob
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="constructions to keep, e.g. pw-radius hg-radius")
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("--in and --out must differ; this does not edit in place")

    params_path = os.path.join(args.src, "_params.pt")
    if not os.path.exists(params_path):
        raise SystemExit(f"no _params.pt in {args.src}")
    params = torch.load(params_path, weights_only=False)

    keep = list(args.arms)
    missing = [a for a in keep if a not in params["arms"]]
    if missing:
        raise SystemExit(f"{missing} not in the source cache "
                         f"(has {params['arms']}); nothing to export")

    os.makedirs(args.dst, exist_ok=True)
    files = [f for f in sorted(glob.glob(os.path.join(args.src, "*.pt")))
             if not f.endswith("_params.pt")]
    print(f"{len(files)} slides | keeping {keep} of {params['arms']}")

    t0, done, skipped, src_b, dst_b = time.time(), 0, 0, 0, 0
    for i, f in enumerate(files, 1):
        out_f = os.path.join(args.dst, os.path.basename(f))
        if os.path.exists(out_f):
            skipped += 1
            continue
        d = torch.load(f, weights_only=False)
        lack = [a for a in keep if a not in d["bags"]]
        if lack:
            print(f"  SKIP {os.path.basename(f)}: lacks {lack}")
            continue
        d["bags"] = {k: v for k, v in d["bags"].items() if k in keep}
        # write beside the target then rename, so an interrupted run never
        # leaves a truncated .pt that the next pass would skip as done
        tmp = out_f + ".part"
        torch.save(d, tmp)
        os.replace(tmp, out_f)
        done += 1
        src_b += os.path.getsize(f)
        dst_b += os.path.getsize(out_f)
        if i % 25 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  {i}/{len(files)} | {el:.0f}s ({el / max(done, 1):.1f}s/slide) "
                  f"| {src_b / 1e9:.1f}GB -> {dst_b / 1e9:.1f}GB", flush=True)

    # the manifest must advertise only what the exported slides contain, or
    # train_patterns.py accepts an arm it will then fail to find per slide
    params["arms"] = keep
    torch.save(params, os.path.join(args.dst, "_params.pt"))

    ratio = (dst_b / src_b) if src_b else 0
    print(f"\nwrote {done} slides ({skipped} already present) to {args.dst}")
    if src_b:
        print(f"{src_b / 1e9:.1f}GB -> {dst_b / 1e9:.1f}GB ({ratio:.0%} of source)")


if __name__ == "__main__":
    main()
