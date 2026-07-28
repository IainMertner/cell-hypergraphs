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
                   abundance=None, device="cpu", patience=20, class_weight=None):
    """Train one arm on train fold `tr`, early-stop on val `va`, score test `te`.

    class_weight: per-class loss weights (inverse frequency) so the model is not
    rewarded for collapsing to the majority class -- essential with imbalanced
    labels, where plain cross-entropy sits at the majority predictor.
    Early stopping uses the VAL fold only, never the test fold, so test stays
    untouched until the final read.
    """
    set_seed(seed)
    model = model.to(device)
    labels_t = labels_t.to(device)
    if abundance is not None:
        abundance = abundance.to(device)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    def run(i):
        if abundance is not None:
            return model(abundance[i])
        bag = [(x.to(device), s.to(device)) for x, s in bags[i]]
        return model(bag)[0]

    best_val, best_test, since = -1.0, 0.0, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = 0.0
        for i in tr:
            loss = loss + F.cross_entropy(run(i).unsqueeze(0), labels_t[i:i + 1],
                                          weight=class_weight)
        (loss / len(tr)).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            def acc(idx):
                return float(np.mean([int(run(i).argmax()) == int(labels_t[i])
                                      for i in idx]))
            va_acc = acc(va)
            if va_acc > best_val:
                best_val, best_test, since = va_acc, acc(te), 0
            else:
                since += 1
                if since >= patience:
                    break
    return best_test


def _make_folds(y, patient, k, seed=0):
    """Patient-grouped, stratified k-fold. Returns list of (train, val, test) idx.

    Grouped: all of a patient's slides go to the same fold (no same-patient
    leakage across train/test). Stratified: folds balanced by class as far as the
    grouping allows -- patients are assigned to folds greedily in class order so
    each fold gets a similar class mix. A val set is carved from each training
    fold for early stopping, so the test fold is never seen during training.
    """
    rng = np.random.default_rng(seed)
    # one representative label per patient (its most common slide label)
    pats = {}
    for i, p in enumerate(patient):
        pats.setdefault(p, []).append(i)
    plist = list(pats)
    plabel = {p: np.bincount([y[i] for i in pats[p]]).argmax() for p in plist}

    # assign patients to folds, balancing classes: for each class, round-robin
    # its patients across folds
    fold_of = {}
    for c in sorted(set(plabel.values())):
        cp = [p for p in plist if plabel[p] == c]
        rng.shuffle(cp)
        for j, p in enumerate(cp):
            fold_of[p] = j % k

    folds = []
    for f in range(k):
        te = [i for p in plist if fold_of[p] == f for i in pats[p]]
        trainval = [i for p in plist if fold_of[p] != f for i in pats[p]]
        # carve a val set: hold out one other fold's patients for validation
        val_fold = (f + 1) % k
        va = [i for p in plist if fold_of[p] == val_fold for i in pats[p]]
        tr = [i for i in trainval if i not in set(va)]
        folds.append((np.array(tr), np.array(va), np.array(te)))
    return folds


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
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--regions-per-batch", type=int, default=16,
                    help="regions encoded per GPU pass; lower if OOM on big slides")
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

    files = [f for f in sorted(glob.glob(os.path.join(args.graph_cache, "*.pt")))
             if not f.endswith("_params.pt")]
    slide_bags, slide_abund, y, patient = [], [], [], []
    for f in files:
        sid = os.path.basename(f)[:-3]
        if sid not in labels:
            continue
        d = torch.load(f)
        slide_bags.append(d["bags"])
        slide_abund.append(d["abundance"])
        y.append(labels[sid])
        # TCGA patient = first 3 hyphen fields (TCGA-XX-YYYY); keep a patient's
        # slides in the SAME fold so same-patient leakage cannot inflate scores
        patient.append("-".join(sid.split("-")[:3]))
    n = len(y)
    print(f"{len(files)} cached slides | {n} have a label "
          f"| {len(set(patient))} unique patients")
    if n < 20:
        print("too few labelled slides for cross-validation.")
        return

    y = torch.tensor(y).long()
    abund = torch.stack(slide_abund)
    counts = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    print(f"class counts {counts}")
    # inverse-frequency class weights for the loss
    freq = torch.tensor([max(counts[c], 1) for c in classes], dtype=torch.float)
    class_weight = (freq.sum() / (len(classes) * freq))
    print(f"class weights {[round(w, 2) for w in class_weight.tolist()]}\n")

    in_dim = slide_bags[0][args.arms[0]][0][0].shape[1]
    target = n_params(HyperRegionEncoder(in_dim, args.hidden))
    pw_h = matched_hidden(lambda i, h, o: PairwiseRegionEncoder(i, h),
                          target, in_dim, 0)
    hidden = {a: (pw_h if a.startswith("pw-") else args.hidden) for a in args.arms}
    print(f"capacity: hyper encoder {target:,} params -> pairwise hidden={pw_h}")

    folds = _make_folds(y.numpy(), patient, args.folds, seed=0)
    print(f"{args.folds}-fold patient-grouped stratified CV; "
          f"fold test sizes {[len(te) for _, _, te in folds]}\n")

    # evaluate one arm across all folds x seeds -> flat list of test accuracies
    def eval_arm(arm):
        scores = []
        bags = None if arm == "abundance-only" else [sb[arm] for sb in slide_bags]
        for tr, va, te in folds:
            for s in range(args.seeds):
                if arm == "abundance-only":
                    m = AbundanceOnly(abund.shape[1], 32, n_classes)
                    acc = train_eval_mil(m, None, y, tr, va, te, args.epochs,
                                         0.01, s, abundance=abund,
                                         device=args.device, class_weight=class_weight)
                else:
                    set_seed(s)
                    m = MILClassifier(arm, in_dim, hidden[arm], n_classes,
                                      regions_per_batch=args.regions_per_batch)
                    acc = train_eval_mil(m, bags, y, tr, va, te, args.epochs,
                                         0.01, s, device=args.device,
                                         class_weight=class_weight)
                scores.append(acc)
        return np.array(scores)

    maj = float((y == y.bincount().argmax()).float().mean())
    print(f"=== CV test accuracy (mean +- sd over {args.folds} folds x {args.seeds} seeds) ===")
    print(f"  {'majority baseline':<18} {maj:.3f}")
    ab = eval_arm("abundance-only")
    print(f"  {'abundance-only':<18} {ab.mean():.3f} +- {ab.std():.3f}")
    for arm in args.arms:
        sc = eval_arm(arm)
        # how often this arm beats abundance-only, fold-and-seed wise
        beat = float((sc > ab).mean())
        flag = "  *beats abundance" if sc.mean() > ab.mean() else ""
        print(f"  {arm:<18} {sc.mean():.3f} +- {sc.std():.3f}"
              f"  (>{'abund'} {beat:.0%} of runs){flag}")

    print("\nabundance-only is the bar: an arm shows spatial signal only if it")
    print("clears it. Majority baseline is the floor. Small n -- read the")
    print("beat-rate alongside the mean, not the mean alone.")


if __name__ == "__main__":
    main()
    print("Report an arm as beating the control only if it clears abundance-only.")


if __name__ == "__main__":
    main()