"""STAGE 1: build every arm's region graphs for every slide, once.

The graphs are deterministic in the segmentation output and the construction
parameters, so training (stage 2) loads these instead of rebuilding.

Changing a construction parameter makes the cache stale. Each file records the
parameters it was built under and a mismatch aborts, rather than mixing settings
into one cache -- delete the dir or pass a fresh --out.

Rerunning with a new arm tops up existing slides with just that arm. _params.pt
is written last and lists only arms present in EVERY slide, so an interrupted
run understates the cache rather than promising arms that are not there.

Output: one .pt per slide holding each arm's region graphs, the slide abundance
vector, and the parameters used.

    qsub scripts/precompute.sh
"""

import argparse
import glob
import os
import time
import numpy as np
import torch

from graphs import (build, load_cache, regions as find_regions, N_TYPES,
                    microns_to_px,
                    DEFAULT_ARMS, PARAMS, FEATURE_VERSION)


# Parameters that change the graphs themselves; disagreeing on any one means two
# caches must not be mixed. `arms` is excluded -- coverage varies per slide and
# is reconciled separately. feature_version counts because the cache stores
# finished feature tensors, so an encoding change invalidates just as hard.
GEOMETRY_KEYS = ("tile_um", "min_cells", "feature_version",
                 "k", "radius_um", "hg_radius_um", "max_size",
                 "window_um", "semantic_min_size", "semantic_stride")


def slide_id(path):
    return os.path.basename(os.path.dirname(path))


def geometry_mismatch(old, new):
    """(changed, added) -- keys the cached params disagree on, and keys it
    predates entirely.

    Changed is fatal. Added is not: nothing in the cache used a value for a key
    that did not exist yet, so topping up stays consistent -- this is what lets
    a new construction be added without rebuilding every slide.
    """
    changed, added = {}, {}
    for k in GEOMETRY_KEYS:
        if k not in old:
            added[k] = new[k]
        elif old[k] != new[k]:
            changed[k] = (old[k], new[k])
    return changed, added


def build_one_slide(cache_path, arms, tile_um, min_cells):
    """Build every arm's region graphs for one slide. Returns a dict or None."""
    centroids, types, mpp, morph = load_cache(cache_path)
    # Regions are cut in MICRONS, like every construction parameter. Cutting in
    # pixels instead makes a region's physical size depend on the scanner: at
    # 0.25 um/px a 4000px tile is 1mm^2, at 0.50 it is 4mm^2, so the cell
    # threshold below would admit four times the tissue on the coarser slides.
    # The cohort holds both: most slides near 0.25, a minority near 0.50, and a
    # few finer still.
    regs = find_regions(centroids, microns_to_px(tile_um, mpp), min_cells)

    bags = {a: [] for a in arms}
    # Per-region inflammatory count, recorded but NOT filtered on. An immune
    # threshold here would bake one task's assumption into a cache every task
    # shares: on a target with no immune component it silently selects regions
    # for an irrelevant property, and undoing it costs a full rebuild. Storing
    # the count leaves the choice to training time, where it is per-task and
    # reversible.
    infl = []
    kept = 0
    for mask, _, _ in regs:
        c, t, m = centroids[mask], types[mask], morph[mask]
        infl.append(int((t == 2).sum()))                   # 2 = Inflammatory
        kept += 1
        for a in arms:
            d = build(a, c, t, mpp, morph=m)
            struct = d.edge_index if a.startswith("pw-") else d.hyperedge_index
            # multi-family arms carry a per-hyperedge family tag, stored as a
            # third slot. Without it the cache cannot distinguish a 6-cell
            # spatial hyperedge from a 200-cell semantic one and family-aware
            # aggregation is impossible. Single-family arms stay 2-tuples, so
            # existing caches and unpack sites are unaffected.
            fam = getattr(d, "family_id", None)
            bags[a].append((d.x.clone(), struct.clone())
                           if fam is None
                           else (d.x.clone(), struct.clone(), fam.clone()))
    if kept == 0:
        return None

    frac = np.array([(types == k).mean() for k in range(1, N_TYPES + 1)])
    abund = torch.tensor(np.concatenate([frac, [np.log1p(len(types))]]),
                         dtype=torch.float)
    return dict(bags=bags, abundance=abund, n_regions=kept,
                region_inflammatory=torch.tensor(infl, dtype=torch.long))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="cellvit_out")
    ap.add_argument("--out", default="graph_cache")
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS,
                    help="default is the single pw-knn vs hg-knn comparison; "
                         "pass more (e.g. hg-radius) to top up an existing cache")
    ap.add_argument("--tile-um", type=float, default=1000.0,
                    help="region side in microns. 1000 reproduces the old "
                         "4000px tile on a 0.25um/px slide, which is 95%% of "
                         "the cohort, while regularising the rest")
    ap.add_argument("--min-cells", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0,
                    help="build only the first N slides (0 = all). For a quick "
                         "measurement run before committing to a full precompute "
                         "-- note it writes a real cache whose _params.pt then "
                         "describes a partial slide set")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    params = dict(arms=list(args.arms), tile_um=args.tile_um,
                  min_cells=args.min_cells,
                  feature_version=FEATURE_VERSION,
                  **{k: PARAMS[k] for k in
                     ("k", "radius_um", "hg_radius_um", "max_size",
                      "window_um", "semantic_min_size", "semantic_stride")})
    params_path = os.path.join(args.out, "_params.pt")

    # check the manifest before building anything -- topping up across a
    # geometry change would silently mix incomparable graphs
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
    if args.limit:
        caches = caches[:args.limit]
        print(f"LIMIT {args.limit}: measurement run, not a full cache\n")
    print(f"{len(caches)} segmented slides found\n")

    done, topped, skipped, empty = 0, 0, 0, 0
    arms_everywhere = None          # intersection of arms present across slides
    t0 = time.time()
    for i, path in enumerate(caches):
        sid = slide_id(path)
        out_path = os.path.join(args.out, f"{sid}.pt")
        existing = torch.load(out_path) if os.path.exists(out_path) else None

        if existing is not None:
            # per-file too: the manifest can lag an interrupted run
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
        # build only what is absent; regions are deterministic in the geometry
        # params, so topped-up arms line up index-for-index with the stored bags
        build_arms = missing if existing is not None else list(args.arms)
        result = build_one_slide(path, build_arms, args.tile_um,
                                 args.min_cells)
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

    # written last, and only for arms every slide has
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