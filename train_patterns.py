"""
train_patterns.py
-----------------
Task 1 (real): predict the Saltz TIL PatternLabel of a slide from its cells.

Labels are SLIDE-LEVEL (one per slide), so this is multiple-instance learning:
each slide is a bag of region graphs, aggregated to a slide prediction. See
mil.py for the models.

The four labels decompose along two axes:
    briskness (abundance): Brisk {Diffuse, Band-like} vs Non-Brisk {Focal, Multifocal}
    arrangement (spatial): how the lymphocytes are organised
The abundance axis is exactly what a counting model captures, so the AbundanceOnly
control matters: if it does as well as the graph arms, the task reduced to counting
and no spatial claim survives. --task arrangement collapses to the spatial axis
only, which is the part higher-order structure should actually help with.

Slide-level means ~n_slides labels (small), so treat results as a debugging /
scaffolding run until the region-level maps arrive. MIL, class encoding, the
abundance control, splitting by slide, and Spearman/accuracy plumbing are all
reused by the later tasks, so this is not throwaway work.

Usage:
    python train_patterns.py --labels til_indices.csv --cache-root cellvit_out/
    python train_patterns.py --task arrangement          # binary spatial axis
    python train_patterns.py --arms pw-knn hg-knn         # subset of arms
    python train_patterns.py --seeds 10
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from graphs import build, load_cache, regions as find_regions, region_mask, N_TYPES, STAGE1
from models import set_seed, matched_hidden, n_params, PairwiseGNN, DeepSetsHyperGNN
from mil import MILClassifier, AbundanceOnly, PairwiseRegionEncoder, HyperRegionEncoder

# 4-way and the collapsed spatial-only ("arrangement") mapping
CLASSES4 = ["Brisk Diffuse", "Brisk Band-like", "Non-Brisk Focal", "Non-Brisk Multifocal"]
# arrangement axis: is the infiltrate localised (focal/multifocal) or spread
# (diffuse/band-like)?  This is the distinction abundance cannot explain.
ARRANGEMENT = {"Brisk Diffuse": "spread", "Brisk Band-like": "spread",
               "Non-Brisk Focal": "localised", "Non-Brisk Multifocal": "localised"}


def slide_id_from_cache(path):
    """cellvit_out/<SLIDE_ID>/cells_cache.npz -> SLIDE_ID"""
    return os.path.basename(os.path.dirname(path))


def load_labels(csv_path, task):
    df = pd.read_csv(csv_path)
    df = df[["SlideID", "PatternLabels"]].dropna()
    if task == "arrangement":
        df = df[df.PatternLabels.isin(ARRANGEMENT)]
        df["y"] = df.PatternLabels.map(ARRANGEMENT)
        classes = ["localised", "spread"]
    else:
        df = df[df.PatternLabels.isin(CLASSES4)]
        df["y"] = df.PatternLabels
        classes = CLASSES4
    idx = {c: i for i, c in enumerate(classes)}
    return dict(zip(df.SlideID, df.y.map(idx))), classes


def build_slide_bag(cache_path, arms, tile_px, min_cells, min_infl):
    """Load one slide's cache, tile it, build every arm's graph per region.

    Region selection is VALIDITY-based, not capped: keep every region with
    enough total cells AND enough inflammatory cells to have a meaningful immune
    arrangement. No densest-N cap (that biases toward tumour-dense regions, which
    is exactly where focal TILs are NOT). Sparse / immune-poor regions are
    dropped because the arrangement label is undefined there, not to save compute
    -- they are cheap anyway.

    Returns dict: arm -> list of (x, struct) region graphs, plus the slide-level
    abundance feature vector for the control.
    """
    centroids, types, mpp, morph = load_cache(cache_path)
    regs = find_regions(centroids, tile_px, min_cells)          # no top_n cap
    if not regs:
        return None, None

    bags = {a: [] for a in arms}
    kept = 0
    for mask, _, _ in regs:
        c, t, m = centroids[mask], types[mask], morph[mask]
        if int((t == 2).sum()) < min_infl:                     # 2 = Inflammatory
            continue                                            # arrangement undefined
        kept += 1
        for a in arms:
            d = build(a, c, t, mpp, morph=m)
            struct = d.edge_index if a.startswith("pw-") else d.hyperedge_index
            bags[a].append((d.x, struct))

    if kept == 0:                                              # no valid region
        return None, None

    # slide-level abundance feature: overall type fractions + log total count
    frac = np.array([(types == k).mean() for k in range(1, N_TYPES + 1)])
    abund = torch.tensor(np.concatenate([frac, [np.log1p(len(types))]]),
                         dtype=torch.float)
    return bags, abund


def train_eval_mil(model, bags, labels_t, tr, va, te, epochs, lr, seed,
                   abundance=None):
    """Train one arm (or the abundance control) and return best-val test accuracy."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val, best_test = -1.0, 0.0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = 0.0
        for i in tr:
            out = (model(abundance[i]) if abundance is not None
                   else model(bags[i])[0])
            loss = loss + F.cross_entropy(out.unsqueeze(0), labels_t[i:i + 1])
        (loss / len(tr)).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            def acc(idx):
                preds = []
                for i in idx:
                    out = (model(abundance[i]) if abundance is not None
                           else model(bags[i])[0])
                    preds.append(int(out.argmax()))
                return float(np.mean([p == int(labels_t[i]) for p, i in zip(preds, idx)]))
            va_acc, te_acc = acc(va), acc(te)
            if va_acc > best_val:
                best_val, best_test = va_acc, te_acc
    return best_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cache-root", default="cellvit_out")
    ap.add_argument("--task", choices=["pattern4", "arrangement"], default="pattern4")
    ap.add_argument("--arms", nargs="*", default=[a for a in STAGE1 if a != "hg-knn+semantic"])
    ap.add_argument("--tile-px", type=int, default=4000)
    ap.add_argument("--min-cells", type=int, default=2000)
    ap.add_argument("--min-infl", type=int, default=50,
                    help="min inflammatory cells for a region to have a defined arrangement")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--split", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    args = ap.parse_args()

    labels, classes = load_labels(args.labels, args.task)
    print(f"task={args.task} | {len(classes)} classes: {classes}")

    # match processed slides to labels
    caches = glob.glob(os.path.join(args.cache_root, "*", "cells_cache.npz"))
    matched = [(p, slide_id_from_cache(p)) for p in caches
               if slide_id_from_cache(p) in labels]
    print(f"{len(caches)} slides segmented | {len(matched)} have a label")
    if len(matched) < 10:
        print("too few labelled slides to train meaningfully yet -- come back when"
              " more of the cohort has segmented.")
        return

    # build every slide's bag once
    print("building slide bags...")
    slide_bags, slide_abund, y = [], [], []
    for path, sid in matched:
        bags, abund = build_slide_bag(path, args.arms, args.tile_px,
                                      args.min_cells, args.min_infl)
        if bags is None:
            continue
        slide_bags.append(bags)
        slide_abund.append(abund)
        y.append(labels[sid])
    n = len(y)
    y = torch.tensor(y).long()
    abund = torch.stack(slide_abund)
    print(f"{n} usable slides | class counts "
          f"{ {c: int((y == i).sum()) for i, c in enumerate(classes)} }\n")

    in_dim = slide_bags[0][args.arms[0]][0][0].shape[1]
    n_classes = len(classes)

    # capacity: match pairwise region-encoder params to the hypergraph one
    target = n_params(HyperRegionEncoder(in_dim, args.hidden))
    pw_h = matched_hidden(lambda i, h, o: PairwiseRegionEncoder(i, h),
                          target, in_dim, 0)
    hidden = {a: (pw_h if a.startswith("pw-") else args.hidden) for a in args.arms}
    print(f"capacity: hyper encoder {target:,} params -> pairwise hidden={pw_h}")

    # split by slide
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_tr, n_va = int(n * args.split[0]), int(n * args.split[1])
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    print(f"split: train {len(tr)} / val {len(va)} / test {len(te)}\n")

    # abundance-only control
    print("=== arms (test accuracy, mean +- sd over seeds) ===")
    ab_scores = [train_eval_mil(AbundanceOnly(abund.shape[1], 32, n_classes),
                                None, y, tr, va, te, args.epochs, 0.01, s,
                                abundance=abund) for s in range(args.seeds)]
    print(f"  {'abundance-only':<18} {np.mean(ab_scores):.3f} +- {np.std(ab_scores):.3f}")

    for arm in args.arms:
        bags_for_arm = [sb[arm] for sb in slide_bags]
        scores = []
        for s in range(args.seeds):
            set_seed(s)
            model = MILClassifier(arm, in_dim, hidden[arm], n_classes)
            scores.append(train_eval_mil(model, bags_for_arm, y, tr, va, te,
                                         args.epochs, 0.01, s))
        print(f"  {arm:<18} {np.mean(scores):.3f} +- {np.std(scores):.3f}")

    print(f"\nmajority-class baseline: "
          f"{float((y == y.bincount().argmax()).float().mean()):.3f}")
    print("\nNOTE: slide-level task, small n -- scaffolding for the MIL pipeline.")
    print("Report an arm as beating the control only if it clears abundance-only.")


if __name__ == "__main__":
    main()