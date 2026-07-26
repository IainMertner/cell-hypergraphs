"""
precompute_graphs.py
--------------------
STAGE 1 of the split pipeline: build every arm's region graphs for every slide,
once, and save them to disk. Training (stage 2, train_patterns.py) then loads
these instead of rebuilding -- turning every future run from hours into minutes.

The graphs are deterministic functions of the segmentation output and the
construction parameters (k, radii, thresholds). None of that changes between
experiments, so this is computed once and reused across tasks, seeds, arms, and
capacity regimes.

CACHE INVALIDATION: if you change a construction parameter -- k, hg_radius_um,
window_um, min_cells, min_infl, tile_px -- the affected caches are STALE. The
parameters used are written into each cache file and checked on load; a mismatch
warns you. Delete the cache dir and rerun to rebuild.

Output: one .pt file per slide in <out>/, each containing every arm's list of
region graphs, the slide abundance vector, and the parameters used.

Usage:
    python precompute_graphs.py --cache-root cellvit_out --out graph_cache
    python precompute_graphs.py --arms pw-knn hg-knn hg-radius   # subset
"""

import argparse
import glob
import os
import time
import numpy as np
import torch

from graphs import build, load_cache, regions as find_regions, N_TYPES, STAGE1, PARAMS


def slide_id(path):
    return os.path.basename(os.path.dirname(path))


def build_one_slide(cache_path, arms, tile_px, min_cells, min_infl):
    """Build every arm's region graphs for one slide. Returns a dict or None."""
    centroids, types, mpp, morph = load_cache(cache_path)
    regs = find_regions(centroids, tile_px, min_cells)     # no cap; validity filter below

    bags = {a: [] for a in arms}
    kept = 0
    for mask, _, _ in regs:
        c, t, m = centroids[mask], types[mask], morph[mask]
        if int((t == 2).sum()) < min_infl:                 # 2 = Inflammatory
            continue
        kept += 1
        for a in arms:
            d = build(a, c, t, mpp, morph=m)
            struct = d.edge_index if a.startswith("pw-") else d.hyperedge_index
            # store only what training needs: features + topology
            bags[a].append((d.x.clone(), struct.clone()))
    if kept == 0:
        return None

    frac = np.array([(types == k).mean() for k in range(1, N_TYPES + 1)])
    abund = torch.tensor(np.concatenate([frac, [np.log1p(len(types))]]),
                         dtype=torch.float)
    return dict(bags=bags, abundance=abund, n_regions=kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="cellvit_out")
    ap.add_argument("--out", default="graph_cache")
    ap.add_argument("--arms", nargs="*", default=STAGE1)
    ap.add_argument("--tile-px", type=int, default=4000)
    ap.add_argument("--min-cells", type=int, default=2000)
    ap.add_argument("--min-infl", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # parameters that define cache validity -- stored and checked on load
    params = dict(arms=list(args.arms), tile_px=args.tile_px,
                  min_cells=args.min_cells, min_infl=args.min_infl,
                  **{k: PARAMS[k] for k in
                     ("k", "radius_um", "hg_radius_um", "window_um", "min_size")})
    torch.save(params, os.path.join(args.out, "_params.pt"))
    print(f"arms: {args.arms}")
    print(f"params: {params}\n")

    caches = sorted(glob.glob(os.path.join(args.cache_root, "*", "cells_cache.npz")))
    print(f"{len(caches)} segmented slides found\n")

    done, skipped, empty = 0, 0, 0
    t0 = time.time()
    for i, path in enumerate(caches):
        sid = slide_id(path)
        out_path = os.path.join(args.out, f"{sid}.pt")
        if os.path.exists(out_path):
            skipped += 1
            continue
        t = time.time()
        result = build_one_slide(path, args.arms, args.tile_px,
                                 args.min_cells, args.min_infl)
        if result is None:
            empty += 1
            print(f"[{i+1}/{len(caches)}] {sid}: no valid regions, skipped")
            continue
        result["params"] = params
        torch.save(result, out_path)
        done += 1
        print(f"[{i+1}/{len(caches)}] {sid}: {result['n_regions']} regions, "
              f"{time.time()-t:.1f}s", flush=True)

    print(f"\nbuilt {done} | already-cached {skipped} | no-valid-regions {empty}")
    print(f"total {time.time()-t0:.0f}s -> {args.out}/")


if __name__ == "__main__":
    main()