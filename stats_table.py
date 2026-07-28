"""
stats_table.py
--------------
Structural statistics per arm, straight off the precomputed graph cache.

Deterministic: no seeds, no training, no models. Reports hyperedge cardinality
and the clique-expansion ratio -- the measured version of "how badly does clique
expansion blow up for this construction". That spectrum is the mechanistic claim
the hypergraph arms rest on, so it is worth being able to re-measure cheaply and
across the whole cohort rather than from one slide.

(This table used to be reachable only through the masked-cell-type script, which
has since been removed; that route also required re-parsing raw cells.json per
slide. Reading the cache instead makes a whole-cohort measurement cheap.)

Usage:
    python stats_table.py --graph-cache graph_cache
    python stats_table.py --graph-cache graph_cache --slides 20
"""

import argparse
import glob
import os
from types import SimpleNamespace

import numpy as np
import torch

from graphs import structural_stats, print_stats_table


def as_data(arm, x, struct):
    """Wrap a cached (features, topology) pair so structural_stats can read it."""
    if arm.startswith("pw-"):
        return SimpleNamespace(edge_index=struct)
    n_he = int(struct[1].max()) + 1 if struct.numel() else 0
    return SimpleNamespace(hyperedge_index=struct, num_hyperedges=n_he)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-cache", default="graph_cache")
    ap.add_argument("--slides", type=int, default=0,
                    help="cap slides scanned (0 = all); regions are all used")
    args = ap.parse_args()

    params = torch.load(os.path.join(args.graph_cache, "_params.pt"))
    arms = list(params["arms"])
    print(f"cache {args.graph_cache} | arms {arms}")
    print(f"params {({k: v for k, v in params.items() if k != 'arms'})}\n")

    files = [f for f in sorted(glob.glob(os.path.join(args.graph_cache, "*.pt")))
             if not f.endswith("_params.pt")]
    if args.slides:
        files = files[:args.slides]

    # accumulate every region of every slide, per arm
    acc = {a: [] for a in arms}
    for f in files:
        bags = torch.load(f)["bags"]
        for a in arms:
            for x, struct in bags.get(a, []):
                acc[a].append(structural_stats(a, as_data(a, x, struct),
                                               int(x.shape[0])))
    n_reg = len(acc[arms[0]]) if arms else 0
    print(f"{len(files)} slides | {n_reg} regions per arm\n")
    if not n_reg:
        print("nothing to report -- empty cache?")
        return

    # mean across regions, plus the expansion range (the figure that matters)
    rows = []
    for a in arms:
        r = acc[a]
        rows.append(dict(
            name=a, kind=r[0]["kind"],
            units=int(np.mean([s["units"] for s in r])),
            incidences=int(np.mean([s["incidences"] for s in r])),
            mean_size=float(np.mean([s["mean_size"] for s in r])),
            median_size=float(np.mean([s["median_size"] for s in r])),
            max_size=int(max(s["max_size"] for s in r)),
            mean_degree=float(np.mean([s["mean_degree"] for s in r])),
            clique_edges=int(np.mean([s["clique_edges"] for s in r])),
            expansion=float(np.mean([s["expansion"] for s in r]))))
    print_stats_table(rows)

    print("\nexpansion ratio across regions (min - max, mean):")
    for a in arms:
        e = np.array([s["expansion"] for s in acc[a]])
        print(f"  {a:<18} {e.min():5.1f}x - {e.max():5.1f}x   (mean {e.mean():5.1f}x)")
    print("\nA hypergraph advantage is mechanistic only if it TRACKS this "
          "ordering.\nRead the spread, not just the mean -- an arm whose ratio "
          "swings widely\nacross regions is not a stable point on the spectrum.")


if __name__ == "__main__":
    main()
