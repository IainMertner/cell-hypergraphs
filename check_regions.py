"""
check_regions.py
----------------
Quick robustness probe: does the arm ordering (baselines < clique < deepsets)
reproduce across REGIONS, or was it a one-region fluke?

Runs the morph-only condition at k=5 over the N most-populated tiles and prints
the per-region test macro-F1 for each arm, plus how often deepsets tops the
clique control. One seed — this is a fragility check, not a significance test.
"""

import numpy as np
import torch

from build_graph import (
    load_cells, top_regions, microns_to_px,
    build_knn_graph, build_neighbourhood_hypergraph,
    build_delaunay_graph, clique_expand, N_TYPES,
)
import train_masked as T

CELLS_JSON = T.CELLS_JSON
TILE_PX = 4000
K = 5
N_REGIONS = 5
RADIUS_UM = 35.0
T.EPOCHS = 150
T.FEATURES = "morph"


def feats(morph, n):
    return torch.from_numpy(morph).float(), morph.shape[1]


def main():
    centroids, types, mpp, morph_all = load_cells(CELLS_JSON, with_morphology=True)
    radius_px = microns_to_px(RADIUS_UM, mpp)
    regions = top_regions(centroids, types, TILE_PX, N_REGIONS)
    print(f"features=morph | k={K} | {len(regions)} regions\n")

    header = f"{'region':>7} {'cells':>6} | {'pw-knn':>7} {'pw-del':>7} " \
             f"{'pw-clq':>7} {'hg-clq':>7} {'hg-ds':>7} | deepsets>clique?"
    print(header)
    print("-" * len(header))

    wins = 0
    for i, (mask, (x0, y0), count) in enumerate(regions):
        sub_c, sub_t, sub_m = centroids[mask], types[mask], morph_all[mask]
        n = len(sub_c)
        y = (torch.from_numpy(sub_t).long() - 1)
        x_full, in_dim = feats(sub_m, n)

        all_t, tr, va, te = T.make_targets(n, T.MASK_FRAC, T.SPLIT, T.SEED)
        x_in = x_full.clone()
        x_in[all_t] = 0.0

        g = build_knn_graph(sub_c, sub_t, K, radius_px)
        h = build_neighbourhood_hypergraph(sub_c, sub_t, K, radius_px)
        d = build_delaunay_graph(sub_c, sub_t, radius_px)
        pwc = clique_expand(h.hyperedge_index)

        vals = {}
        for name, model, struct in [
            ("pw-knn", T.PairwiseGNN(in_dim, T.HIDDEN, N_TYPES), g.edge_index),
            ("pw-del", T.PairwiseGNN(in_dim, T.HIDDEN, N_TYPES), d.edge_index),
            ("pw-clq", T.PairwiseGNN(in_dim, T.HIDDEN, N_TYPES), pwc),
            ("hg-clq", T.HyperGNN(in_dim, T.HIDDEN, N_TYPES), h.hyperedge_index),
            ("hg-ds", T.DeepSetsHyperGNN(in_dim, T.HIDDEN, N_TYPES), h.hyperedge_index),
        ]:
            best = T.run_arm(name, model, x_in, struct, y, tr, va, te)
            vals[name] = best[3]  # test F1

        ds_win = vals["hg-ds"] > vals["hg-clq"]
        wins += int(ds_win)
        print(f"{i:>7} {n:>6} | {vals['pw-knn']:>7.3f} {vals['pw-del']:>7.3f} "
              f"{vals['pw-clq']:>7.3f} {vals['hg-clq']:>7.3f} {vals['hg-ds']:>7.3f} | "
              f"{'yes' if ds_win else 'no'}")

    print(f"\ndeepsets > hg-clique in {wins}/{len(regions)} regions")


if __name__ == "__main__":
    main()