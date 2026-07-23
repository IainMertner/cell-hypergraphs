"""
capacity_check.py
-----------------
Resolve the open confound: does hg-deepsets win because of SUM AGGREGATION
(structure), or simply because it has MORE PARAMETERS than the other arms?

Averaged over 5 regions, the four equal-capacity arms tied at ~0.463 test F1
while deepsets alone reached ~0.528. That is exactly the pattern BOTH
hypotheses predict, because deepsets is also the only arm with extra params.

This script runs three capacity regimes so the two explanations separate:

  "default"    - as before: baselines hidden=32, deepsets hidden=32
                 (deepsets has more params -- the confounded comparison)
  "matched"    - baselines widened until their param count ~= deepsets
                 (if deepsets still wins, it is not just capacity)
  "handicap"   - deepsets SHRUNK below the baselines' param count
                 (if deepsets still wins while smaller, that is the strongest
                  possible version of the result)

Prints per-arm parameter counts so the comparison is transparent, then the
per-region and mean test macro-F1 for each regime.

Runs on CPU against the single slide's cells.json. No cluster needed.
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
EPOCHS = 150
SEED = 0


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def find_matching_hidden(target_params, in_dim, tol=0.05, lo=8, hi=4096):
    """Smallest PairwiseGNN hidden dim whose param count is within tol of target."""
    best, best_err = lo, float("inf")
    for h in range(lo, hi, 4):
        p = n_params(T.PairwiseGNN(in_dim, h, N_TYPES))
        err = abs(p - target_params) / target_params
        if err < best_err:
            best, best_err = h, err
        if p > target_params * (1 + tol):
            break
    return best


def find_deepsets_hidden(target_params, in_dim, lo=4, hi=256):
    """Largest DeepSets hidden dim whose param count stays BELOW target."""
    best = lo
    for h in range(lo, hi, 2):
        p = n_params(T.DeepSetsHyperGNN(in_dim, h, N_TYPES))
        if p <= target_params:
            best = h
        else:
            break
    return best


def run_regime(name, regions, radius_px, hid_pw, hid_hg, hid_ds, in_dim):
    """Train all five arms at the given hidden sizes across all regions."""
    print(f"\n{'='*74}")
    print(f"REGIME: {name}   (pairwise h={hid_pw}, hg-clique h={hid_hg}, deepsets h={hid_ds})")
    print(f"{'='*74}")
    print(f"  params: pairwise={n_params(T.PairwiseGNN(in_dim, hid_pw, N_TYPES)):,} | "
          f"hg-clique={n_params(T.HyperGNN(in_dim, hid_hg, N_TYPES)):,} | "
          f"deepsets={n_params(T.DeepSetsHyperGNN(in_dim, hid_ds, N_TYPES)):,}")

    arms = ["pw-knn", "pw-del", "pw-clq", "hg-clq", "hg-ds"]
    scores = {a: [] for a in arms}

    for i, (mask, (x0, y0), count) in enumerate(regions):
        sub_c, sub_t, sub_m = region_data[i]
        n = len(sub_c)
        y = torch.from_numpy(sub_t).long() - 1
        x_full = torch.from_numpy(sub_m).float()

        all_t, tr, va, te = T.make_targets(n, T.MASK_FRAC, T.SPLIT, SEED)
        x_in = x_full.clone()
        x_in[all_t] = 0.0

        g = build_knn_graph(sub_c, sub_t, K, radius_px)
        h = build_neighbourhood_hypergraph(sub_c, sub_t, K, radius_px)
        d = build_delaunay_graph(sub_c, sub_t, radius_px)
        pwc = clique_expand(h.hyperedge_index)

        specs = [
            ("pw-knn", T.PairwiseGNN(in_dim, hid_pw, N_TYPES), g.edge_index),
            ("pw-del", T.PairwiseGNN(in_dim, hid_pw, N_TYPES), d.edge_index),
            ("pw-clq", T.PairwiseGNN(in_dim, hid_pw, N_TYPES), pwc),
            ("hg-clq", T.HyperGNN(in_dim, hid_hg, N_TYPES), h.hyperedge_index),
            ("hg-ds", T.DeepSetsHyperGNN(in_dim, hid_ds, N_TYPES), h.hyperedge_index),
        ]
        row = {}
        for nm, model, struct in specs:
            best = T.run_arm(nm, model, x_in, struct, y, tr, va, te)
            row[nm] = best[3]              # test macro-F1
            scores[nm].append(best[3])
        print(f"  region {i} (n={n:,}): " +
              " ".join(f"{a}={row[a]:.3f}" for a in arms))

    print(f"  {'-'*70}")
    print("  MEAN test F1: " +
          "  ".join(f"{a}={np.mean(scores[a]):.3f}" for a in arms))
    return {a: float(np.mean(scores[a])) for a in arms}


if __name__ == "__main__":
    T.EPOCHS = EPOCHS
    centroids, types, mpp, morph_all = load_cells(CELLS_JSON, with_morphology=True)
    radius_px = microns_to_px(RADIUS_UM, mpp)
    regions = top_regions(centroids, types, TILE_PX, N_REGIONS)

    region_data = []
    for mask, (x0, y0), count in regions:
        region_data.append((centroids[mask], types[mask], morph_all[mask]))

    in_dim = morph_all.shape[1]        # morph-only features
    print(f"features=morph (dim {in_dim}) | k={K} | {len(regions)} regions | seed={SEED}")

    # --- work out the matched / handicapped hidden sizes ---
    ds_params = n_params(T.DeepSetsHyperGNN(in_dim, 32, N_TYPES))
    pw_params = n_params(T.PairwiseGNN(in_dim, 32, N_TYPES))
    print(f"\nat hidden=32: deepsets has {ds_params:,} params vs pairwise {pw_params:,} "
          f"({ds_params / pw_params:.1f}x)")

    hid_matched = find_matching_hidden(ds_params, in_dim)
    hid_handicap = find_deepsets_hidden(pw_params, in_dim)
    print(f"-> to match deepsets, pairwise needs hidden={hid_matched}")
    print(f"-> to fit under pairwise, deepsets needs hidden={hid_handicap}")

    results = {}
    results["default"] = run_regime("default (confounded)", regions, radius_px,
                                    32, 32, 32, in_dim)
    results["matched"] = run_regime("matched (baselines widened)", regions, radius_px,
                                    hid_matched, hid_matched, 32, in_dim)
    results["handicap"] = run_regime("handicap (deepsets shrunk)", regions, radius_px,
                                     32, 32, hid_handicap, in_dim)

    print(f"\n{'='*74}")
    print("SUMMARY - mean test macro-F1")
    print(f"{'='*74}")
    arms = ["pw-knn", "pw-del", "pw-clq", "hg-clq", "hg-ds"]
    print(f"{'regime':<12}" + "".join(f"{a:>9}" for a in arms) + "   ds - best_other")
    for reg, sc in results.items():
        best_other = max(sc[a] for a in arms if a != "hg-ds")
        print(f"{reg:<12}" + "".join(f"{sc[a]:>9.3f}" for a in arms) +
              f"   {sc['hg-ds'] - best_other:+.3f}")
    print("\nIf the deepsets margin survives 'matched' and 'handicap', the effect")
    print("is structural (sum aggregation), not capacity.")