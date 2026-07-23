"""
train_masked.py
---------------
Task 0: masked cell-type prediction. A HARNESS SMOKE TEST, not a result.

30% of cells have their features zeroed; the model predicts their PanNuke type
from the surrounding cells and the graph topology. Because the label (cell type)
is also what the visible neighbours carry, the signal is locally readable and
this task is not expected to discriminate representations -- it exists to prove
the pipeline runs identically across arms, and to collect the deterministic
structural statistics.

Consolidates what were three scripts (train_masked / check_regions /
capacity_check): region count, seed count and capacity regime are now flags.

Usage
-----
    python train_masked.py                      # defaults: 5 regions, 3 seeds, matched
    python train_masked.py --regions 5 --seeds 10
    python train_masked.py --capacity default   # unmatched (confounded; for comparison)
    python train_masked.py --features morph     # type | morph | both
    python train_masked.py --stats-only         # structural stats, no training
"""

import argparse
import numpy as np
import torch

from graphs import (build, load_cells, regions as find_regions, region_mask,
                    structural_stats, print_stats_table, STAGE1, N_TYPES)
from models import (model_for, struct_of, matched_hidden, n_params,
                    make_targets, train_eval, PairwiseGNN, DeepSetsHyperGNN)

CELLS_JSON = r"\\wsl$\Ubuntu\home\iain\cellvit\test_out\TCGA-E2-A14P\cells.json"


def feature_matrix(types, morph, mode):
    """Which node features to use. Returns (morph_or_None, label) for graphs.build."""
    if mode == "type":
        return None
    return morph                      # graphs.build concatenates one-hot + morph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=CELLS_JSON)
    ap.add_argument("--regions", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--tile-px", type=int, default=4000)
    ap.add_argument("--min-cells", type=int, default=2000)
    ap.add_argument("--mask-frac", type=float, default=0.30)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--features", choices=["type", "morph", "both"], default="morph")
    ap.add_argument("--capacity", choices=["matched", "default"], default="matched")
    ap.add_argument("--arms", nargs="*", default=STAGE1)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    print(f"loading {args.cells}")
    centroids, types, mpp, morph_all = load_cells(args.cells, with_morphology=True)
    print(f"{len(centroids):,} cells | mpp={mpp}")

    regs = find_regions(centroids, args.tile_px, args.min_cells, top_n=args.regions)
    print(f"{len(regs)} regions of {args.tile_px}px (>= {args.min_cells} cells)")
    print(f"features={args.features} | capacity={args.capacity} | "
          f"seeds={args.seeds} | arms={len(args.arms)}\n")

    # ---- build every arm on every region once, up front ----
    built = []                                   # [{arm: Data}, ...] per region
    for mask, (x0, y0), n in regs:
        c, t = centroids[mask], types[mask]
        m = morph_all[mask] if args.features != "type" else None
        if args.features == "morph":
            # morphology ONLY: drop the one-hot type columns after building
            per_arm = {a: build(a, c, t, mpp, morph=m) for a in args.arms}
            for d in per_arm.values():
                d.x = d.x[:, N_TYPES:]
        else:
            per_arm = {a: build(a, c, t, mpp, morph=m) for a in args.arms}
        built.append(per_arm)

    # ---- structural statistics (deterministic; pre-reg 8.2) ----
    print("=== structural statistics (region 0) ===")
    n0 = int(regs[0][2])
    print_stats_table([structural_stats(a, built[0][a], n0) for a in args.arms])

    in_dim = built[0][args.arms[0]].x.shape[1]
    print(f"\nnode feature dim: {in_dim}")

    # ---- capacity matching ----
    hidden = {a: args.hidden for a in args.arms}
    if args.capacity == "matched":
        target = n_params(DeepSetsHyperGNN(in_dim, args.hidden, N_TYPES))
        pw_h = matched_hidden(PairwiseGNN, target, in_dim, N_TYPES)
        for a in args.arms:
            if a.startswith("pw-"):
                hidden[a] = pw_h
        print(f"capacity target {target:,} params -> pairwise hidden={pw_h}")
    print("params: " + " | ".join(
        f"{a}={n_params(model_for(a, in_dim, hidden[a], N_TYPES)):,}"
        for a in args.arms))

    if args.stats_only:
        return

    # ---- train: arms x regions x seeds ----
    print(f"\n=== masked cell-type prediction ===")
    scores = {a: [] for a in args.arms}          # test macro-F1, one per (region, seed)
    for ri, (per_arm, (mask, _, n)) in enumerate(zip(built, regs)):
        t = types[mask]
        y = torch.from_numpy(t).long() - 1
        all_t, tr, va, te = make_targets(n, args.mask_frac, (0.6, 0.2, 0.2), 0)

        row = {}
        for arm in args.arms:
            d = per_arm[arm]
            x = d.x.clone()
            x[all_t] = 0.0                       # hide the target cells
            per_seed = []
            for seed in range(args.seeds):
                model = model_for(arm, in_dim, hidden[arm], N_TYPES)
                res = train_eval(model, x, struct_of(arm, d), y, tr, va, te,
                                 N_TYPES, epochs=args.epochs, seed=seed)
                per_seed.append(res[3])          # test macro-F1
                scores[arm].append(res[3])
            row[arm] = (float(np.mean(per_seed)), float(np.std(per_seed)))
        print(f"region {ri} (n={n:,}): " +
              "  ".join(f"{a}={row[a][0]:.3f}+-{row[a][1]:.3f}" for a in args.arms))

    # ---- summary ----
    print(f"\n=== mean test macro-F1 over {len(regs)} regions x {args.seeds} seeds ===")
    for a in args.arms:
        v = np.array(scores[a])
        print(f"  {a:<18} {v.mean():.3f} +- {v.std():.3f}   (n={len(v)})")
    best_pw = max((np.mean(scores[a]) for a in args.arms if a.startswith("pw-")),
                  default=float("nan"))
    print(f"\n  best pairwise baseline: {best_pw:.3f}")
    for a in args.arms:
        if a.startswith("hg-"):
            print(f"  {a:<18} margin over best pairwise: "
                  f"{np.mean(scores[a]) - best_pw:+.3f}")
    print("\nNOTE: this task is expected to produce a null. Differences here are")
    print("not evidence about higher-order structure -- see pre-registration section 9.")


if __name__ == "__main__":
    main()