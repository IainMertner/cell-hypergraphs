"""Structural statistics per arm, straight off the precomputed graph cache.

No seeds, no training, no models. Reports hyperedge cardinality and the
clique-expansion ratio -- the measured version of the mechanistic claim the
hypergraph arms rest on, over the whole cohort rather than one slide.

    python stats_table.py --graph-cache graph_cache
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

    # cached regions are (x, struct) or, for multi-family arms, (x, struct, fam)
    acc = {a: [] for a in arms}
    fam_sizes = {}                       # arm -> family -> [cardinalities]
    for f in files:
        bags = torch.load(f)["bags"]
        for a in arms:
            for g in bags.get(a, []):
                x, struct = g[0], g[1]
                acc[a].append(structural_stats(a, as_data(a, x, struct),
                                               int(x.shape[0])))
                if len(g) > 2 and struct.numel():
                    n_he = int(struct[1].max()) + 1
                    sizes = np.bincount(struct[1].numpy(), minlength=n_he)
                    fam = g[2].numpy()
                    for k in np.unique(fam):
                        fam_sizes.setdefault(a, {}).setdefault(
                            int(k), []).extend(sizes[fam == k].tolist())
    n_reg = len(acc[arms[0]]) if arms else 0
    print(f"{len(files)} slides | {n_reg} regions per arm\n")
    if not n_reg:
        print("nothing to report -- empty cache?")
        return

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

    # Multi-family arms need their families reported separately. A blended mean
    # over a ~6-cell spatial family and a ~200-cell semantic one describes
    # neither, and the gap between them is exactly what family-aware
    # aggregation has to handle.
    if fam_sizes:
        print("\nper-family cardinality (0 = spatial, 1 = semantic):")
        for a, fams in fam_sizes.items():
            for k in sorted(fams):
                s = np.array(fams[k])
                cliq = int((s * (s - 1) // 2).sum())
                print(f"  {a:<18} family {k}  n={len(s):>7,}  "
                      f"mean {s.mean():6.1f}  median {np.median(s):5.0f}  "
                      f"p99 {np.percentile(s, 99):6.0f}  max {s.max():5d}  "
                      f"| clique {cliq:>12,} edges")
    print("\nA hypergraph advantage is mechanistic only if it tracks this "
          "ordering.\nRead the spread too: an arm whose ratio swings widely "
          "across regions is\nnot a stable point on the spectrum.")


if __name__ == "__main__":
    main()
