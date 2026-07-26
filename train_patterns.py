"""
train_patterns.py
-----------------
Task 1 (real): predict the Saltz TIL PatternLabel of a slide from its cells.

STAGE 2 of the split pipeline. Loads region graphs precomputed by
precompute_graphs.py rather than rebuilding them -- so this runs in minutes and
can be re-run freely across tasks, seeds, arms, and capacity regimes.

Labels are SLIDE-LEVEL (one per slide), so this is multiple-instance learning:
each slide is a bag of region graphs, aggregated to a slide prediction (see
mil.py). The four labels decompose into a briskness (abundance) axis and an
arrangement (spatial) axis; --task arrangement isolates the spatial axis, which
is the part higher-order structure should help with and abundance cannot explain.
The AbundanceOnly control must be cleared before any spatial claim is made.

Slide-level means small n -- treat as scaffolding / a first real test until the
region-level maps arrive.

Usage:
    python precompute_graphs.py --cache-root cellvit_out --out graph_cache   # once
    python train_patterns.py --graph-cache graph_cache --labels til_indices.csv
    python train_patterns.py --graph-cache graph_cache --labels til_indices.csv \
        --task arrangement --arms pw-knn hg-knn --seeds 10
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from graphs import N_TYPES, STAGE1
from models import set_seed, matched_hidden, n_params
from mil import MILClassifier, AbundanceOnly, PairwiseRegionEncoder, HyperRegionEncoder

# 4-way and the collapsed spatial-only ("arrangement") mapping
CLASSES4 = ["Brisk Diffuse", "Brisk Band-like", "Non-Brisk Focal", "Non-Brisk Multifocal"]
ARRANGEMENT = {"Brisk Diffuse": "spread", "Brisk Band-like": "spread",
               "Non-Brisk Focal": "localised", "Non-Brisk Multifocal": "localised"}


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
    ap.add_argument("--graph-cache", default="graph_cache",
                    help="dir of precomputed per-slide graphs (precompute_graphs.py)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--task", choices=["pattern4", "arrangement"], default="pattern4")
    ap.add_argument("--arms", nargs="*", default=[a for a in STAGE1 if a != "hg-knn+semantic"])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--split", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    args = ap.parse_args()

    labels, classes = load_labels(args.labels, args.task)
    n_classes = len(classes)
    print(f"task={args.task} | {n_classes} classes: {classes}")

    cache_params = torch.load(os.path.join(args.graph_cache, "_params.pt"))
    print(f"graph cache: k={cache_params['k']}, "
          f"hg_radius_um={cache_params['hg_radius_um']}, "
          f"min_infl={cache_params['min_infl']}")
    for a in args.arms:
        if a not in cache_params["arms"]:
            raise ValueError(f"arm {a!r} not in graph cache "
                             f"(cached: {cache_params['arms']}); rerun precompute")

    # load every cached slide that has a label
    files = [f for f in sorted(glob.glob(os.path.join(args.graph_cache, "*.pt")))
             if not f.endswith("_params.pt")]
    slide_bags, slide_abund, y = [], [], []
    for f in files:
        sid = os.path.basename(f)[:-3]
        if sid not in labels:
            continue
        d = torch.load(f)
        slide_bags.append(d["bags"])
        slide_abund.append(d["abundance"])
        y.append(labels[sid])
    n = len(y)
    print(f"{len(files)} cached slides | {n} have a label")
    if n < 10:
        print("too few labelled slides to train meaningfully.")
        return

    y = torch.tensor(y).long()
    abund = torch.stack(slide_abund)
    print(f"class counts { {c: int((y == i).sum()) for i, c in enumerate(classes)} }\n")

    in_dim = slide_bags[0][args.arms[0]][0][0].shape[1]

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