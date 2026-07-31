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

CACHE INVALIDATION: if you change a construction parameter -- k, radius_um,
min_cells, min_infl, tile_px, or the feature encoding -- caches are STALE. The
parameters used are written into each cache file and checked against the current
ones on every rerun; a mismatch ABORTS rather than silently mixing graphs built
under different settings into one cache. Delete the cache dir (or pass a fresh
--out) to rebuild.

ADDING ARMS: rerunning with an arm the cache lacks tops up existing slides with
just that arm rather than skipping them, so you never have to rebuild the whole
cache to add one construction. _params.pt is written at the END of a run and
records only the arms present in EVERY cached slide, so an interrupted run
leaves the manifest conservative rather than overstating what is on disk.

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

from graphs import (build, load_cache, regions as find_regions, N_TYPES,
                    DEFAULT_ARMS, PARAMS, FEATURE_VERSION)


# parameters that change the GRAPHS themselves. Two caches agreeing on these are
# interchangeable; disagreeing on any one of them means the graphs are not
# comparable and must not share a cache dir. `arms` is deliberately NOT here --
# arm coverage varies per slide and is reconciled separately.
# feature_version is here for the same reason: the cache stores finished feature
# tensors, so an encoding change is just as invalidating as a geometry change,
# and unlike geometry it cannot be spotted by eye from the stored parameters.
GEOMETRY_KEYS = ("tile_px", "min_cells", "min_infl", "feature_version",
                 "k", "radius_um", "hg_radius_um")


def slide_id(path):
    return os.path.basename(os.path.dirname(path))


def geometry_mismatch(old, new):
    """(changed, added) -- keys where a cached param set disagrees, and keys the
    cache predates entirely.

    CHANGED is fatal: the cached graphs were built under a different setting and
    are not comparable with new ones.

    ADDED is not. A key absent from the cached params means the cache was built
    before that parameter existed -- so nothing in it used any value for it, and
    topping up is consistent. This is what lets a new construction (hg-radius,
    which introduced hg_radius_um) be added to an existing cache without forcing
    a full rebuild of every slide. Reported, not silent.
    """
    changed, added = {}, {}
    for k in GEOMETRY_KEYS:
        if k not in old:
            added[k] = new[k]
        elif old[k] != new[k]:
            changed[k] = (old[k], new[k])
    return changed, added


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
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS,
                    help="default is the single pw-knn vs hg-knn comparison; "
                         "pass more (e.g. hg-radius) to top up an existing cache")
    ap.add_argument("--tile-px", type=int, default=4000)
    ap.add_argument("--min-cells", type=int, default=2000)
    ap.add_argument("--min-infl", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # parameters that define cache validity -- stored and checked on every rerun
    params = dict(arms=list(args.arms), tile_px=args.tile_px,
                  min_cells=args.min_cells, min_infl=args.min_infl,
                  feature_version=FEATURE_VERSION,
                  **{k: PARAMS[k] for k in ("k", "radius_um", "hg_radius_um")})
    params_path = os.path.join(args.out, "_params.pt")

    # Check the manifest BEFORE building anything: a geometry change means the
    # cached graphs and the ones we are about to build are not comparable, and
    # topping up would silently mix them. Abort with the offending keys named.
    if os.path.exists(params_path):
        changed, added = geometry_mismatch(torch.load(params_path), params)
        if changed:
            lines = "\n".join(f"    {k}: cached {o!r} -> requested {n!r}"
                              for k, (o, n) in sorted(changed.items()))
            raise SystemExit(
                f"cache {args.out}/ was built with different parameters:\n{lines}\n"
                "  the cached graphs are stale. Delete the cache dir and rerun, "
                "or pass a fresh --out.")
        if added:
            print("note: cache predates " + ", ".join(sorted(added))
                  + " -- nothing in it used those, so topping up is consistent. "
                  + "Now recording "
                  + ", ".join(f"{k}={v!r}" for k, v in sorted(added.items())))

    print(f"arms: {args.arms}")
    print(f"params: {params}\n")

    caches = sorted(glob.glob(os.path.join(args.cache_root, "*", "cells_cache.npz")))
    print(f"{len(caches)} segmented slides found\n")

    done, topped, skipped, empty = 0, 0, 0, 0
    arms_everywhere = None          # intersection of arms present across slides
    t0 = time.time()
    for i, path in enumerate(caches):
        sid = slide_id(path)
        out_path = os.path.join(args.out, f"{sid}.pt")
        existing = torch.load(out_path) if os.path.exists(out_path) else None

        if existing is not None:
            # per-file check too: the manifest can be absent or lag behind if an
            # earlier run was interrupted, but every slide file carries its own
            changed, _ = geometry_mismatch(existing.get("params", {}), params)
            if changed:
                lines = "\n".join(f"    {k}: cached {o!r} -> requested {n!r}"
                                  for k, (o, n) in sorted(changed.items()))
                raise SystemExit(
                    f"{out_path} was built with different parameters:\n{lines}\n"
                    "  delete the cache dir and rerun, or pass a fresh --out.")
            missing = [a for a in args.arms if a not in existing["bags"]]
            if not missing:
                skipped += 1
                arms_everywhere = (set(existing["bags"]) if arms_everywhere is None
                                   else arms_everywhere & set(existing["bags"]))
                continue

        t = time.time()
        # build only what is absent; regions are a deterministic function of the
        # geometry params (checked identical above), so topped-up arms line up
        # index-for-index with the bags already stored
        build_arms = missing if existing is not None else list(args.arms)
        result = build_one_slide(path, build_arms, args.tile_px,
                                 args.min_cells, args.min_infl)
        if result is None:
            empty += 1
            print(f"[{i+1}/{len(caches)}] {sid}: no valid regions, skipped")
            continue

        if existing is not None:
            existing["bags"].update(result["bags"])
            result = existing
            topped += 1
            note = f"+{len(build_arms)} arm(s) {build_arms}"
        else:
            done += 1
            note = f"{result['n_regions']} regions"
        result["params"] = dict(params, arms=sorted(result["bags"]))
        torch.save(result, out_path)
        arms_everywhere = (set(result["bags"]) if arms_everywhere is None
                           else arms_everywhere & set(result["bags"]))
        print(f"[{i+1}/{len(caches)}] {sid}: {note}, "
              f"{time.time()-t:.1f}s", flush=True)

    # Written LAST, and only for arms every cached slide actually has -- an
    # interrupted run then leaves a manifest that understates the cache rather
    # than one that promises arms training would KeyError on.
    if arms_everywhere:
        torch.save(dict(params, arms=sorted(arms_everywhere)), params_path)

    print(f"\nbuilt {done} | topped-up {topped} | already-complete {skipped} "
          f"| no-valid-regions {empty}")
    if arms_everywhere:
        print(f"arms available for every cached slide: {sorted(arms_everywhere)}")
    else:
        print("no slides cached -- _params.pt not written")
    print(f"total {time.time()-t0:.0f}s -> {args.out}/")


if __name__ == "__main__":
    main()