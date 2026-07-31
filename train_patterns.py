"""
train_patterns.py
-----------------
Task 1 (real): predict the Saltz TIL PatternLabel of a slide from its cells.

STAGE 2 of the split pipeline. Loads region graphs precomputed by
precompute_graphs.py rather than rebuilding them -- so this runs in minutes and
can be re-run freely across tasks, seeds, arms, and capacity regimes.

Labels are SLIDE-LEVEL (one per slide), so this is multiple-instance learning:
each slide is a bag of region graphs, aggregated to a slide prediction (see
models.py). The four labels decompose into a briskness (abundance) axis and an
arrangement (spatial) axis; --task arrangement isolates the spatial axis, which
is the part higher-order structure should help with. See the ARRANGEMENT mapping
below -- collapsing those four labels naively gives you the abundance axis back.
The AbundanceOnly control must be cleared before any spatial claim is made.

Slide-level labels mean small n. Read the corrected p and the win rate together,
not the mean difference alone.

Usage:
    python precompute_graphs.py --cache-root cellvit_out --out graph_cache   # once
    python train_patterns.py --graph-cache graph_cache --labels til_indices.csv
    python train_patterns.py --graph-cache graph_cache --labels til_indices.csv \
        --task arrangement --arms pw-knn hg-knn --seeds 10
"""

import argparse
import glob
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

from graphs import N_TYPES, DEFAULT_ARMS
from models import (set_seed, matched_hidden, n_params, macro_f1,
                    MILClassifier, AbundanceOnly, pack_bag)

# 4-way and the collapsed spatial-only ("arrangement") mapping
CLASSES4 = ["Brisk Diffuse", "Brisk Band-like", "Non-Brisk Focal", "Non-Brisk Multifocal"]

# The four Saltz labels pair a briskness (abundance) term with an arrangement
# (spatial) term -- but NOT independently. Brisk only ever appears with
# Diffuse/Band-like, Non-Brisk only with Focal/Multifocal, so across this label
# set the two axes are perfectly collinear. Any split grouping
# {Diffuse, Band-like} against {Focal, Multifocal} is therefore just
# Brisk vs Non-Brisk: an ABUNDANCE task wearing a spatial name. An earlier
# version of this mapping did exactly that, which made --task arrangement the
# worst available test of spatial structure rather than the best.
#
# Crossing briskness -- each target class holding one Brisk and one Non-Brisk
# label -- removes that PERFECT collinearity. It does NOT make the target
# independent of abundance. How much briskness still predicts arrangement
# depends on the cohort's joint distribution, and can be large: on 113 TCGA-BRCA
# slides a briskness-only rule scores 0.717 against a 0.558 majority baseline,
# because Brisk skews Diffuse and Non-Brisk skews Focal. load_labels() prints
# that number every run -- read it before calling this a spatial task.
#
# The genuinely abundance-free test is a WITHIN-briskness contrast (Non-Brisk
# Focal vs Non-Brisk Multifocal, say), which holds briskness constant by
# construction at the cost of sample size.
#
# The axis below is "one contiguous locus" vs "several / dispersed":
#     Band-like  -- a single band at the invasive margin  -> focused
#     Focal      -- a single focus                        -> focused
#     Diffuse    -- spread throughout                     -> dispersed
#     Multifocal -- several separate foci                 -> dispersed
#
# That reading is a domain judgement about Saltz's taxonomy, not something the
# data settles. Check it against your label source before quoting a result --
# load_labels() prints the briskness balance so you can see the confound is gone.
ARRANGEMENT = {"Brisk Band-like": "focused", "Non-Brisk Focal": "focused",
               "Brisk Diffuse": "dispersed", "Non-Brisk Multifocal": "dispersed"}


def load_labels(csv_path, task, label_col="PatternLabels", min_class=5):
    """CSV -> ({slide id: class index}, ordered class names).

    task="auto" takes `label_col` at face value: the classes are its sorted
    unique values. Because the graph cache is LABEL-FREE -- precompute_graphs.py
    never sees a label -- pointing stage 2 at a different CSV is the whole cost
    of running another slide-level categorical task. No re-precompute.

    That covers discrete targets (subtype, grade, stage, receptor status).
    Continuous targets need a regression path instead: the loss, class weights,
    stratified folds and accuracy/macro-F1 all assume discrete classes.

    pattern4/arrangement are the TIL-specific mappings and always read
    PatternLabels, ignoring label_col.
    """
    df = pd.read_csv(csv_path)
    if "SlideID" not in df.columns:
        raise ValueError(f"{csv_path} has no SlideID column "
                         f"(found: {list(df.columns)})")

    if task == "auto":
        if label_col not in df.columns:
            raise ValueError(f"{csv_path} has no {label_col!r} column "
                             f"(found: {list(df.columns)}); set --label-col")
        df = df[["SlideID", label_col]].dropna()
        df["y"] = df[label_col].astype(str)        # str so numeric codes name cleanly
        # Drop classes too small to survive fold splitting. With ~140 slides a
        # class of 2 cannot appear in train, val AND test, so it contributes
        # nothing but a depressed macro-F1. Dropping is loud, not silent --
        # it changes the task, and you should know it happened.
        counts = df.y.value_counts()
        rare = sorted(counts[counts < min_class].index)
        if rare:
            print(f"dropping {len(rare)} class(es) with < {min_class} slides: "
                  + ", ".join(f"{c} (n={counts[c]})" for c in rare))
            df = df[~df.y.isin(rare)]
        classes = sorted(df.y.unique())
        if len(classes) < 2:
            raise ValueError(
                f"{label_col!r} leaves {len(classes)} class(es) after the "
                f"min-class={min_class} filter; nothing to classify")
        print(f"label column {label_col!r} -> {len(classes)} classes\n")
    elif task == "arrangement":
        df = df[["SlideID", "PatternLabels"]].dropna()
        df = df[df.PatternLabels.isin(ARRANGEMENT)]
        df["y"] = df.PatternLabels.map(ARRANGEMENT)
        classes = ["focused", "dispersed"]
        # Quantify how much of this "spatial" target abundance alone explains.
        # A mixed cross-tab is necessary but NOT sufficient: the best
        # briskness-only rule is what actually says how confounded the task is,
        # and it can sit far above the majority baseline even when both rows are
        # mixed. Anything the graph arms achieve below this line is not spatial.
        brisk = df.PatternLabels.str.startswith("Brisk")
        print("briskness x arrangement:")
        for c in classes:
            m = df.y == c
            print(f"  {c:<10} Brisk {int((m & brisk).sum()):>3} | "
                  f"Non-Brisk {int((m & ~brisk).sum()):>3}")
        n_tot = len(df)
        if n_tot:
            # best rule using briskness alone: per stratum, predict its majority
            best = sum(max(int(((df.y == c) & b).sum()) for c in classes)
                       for b in (brisk, ~brisk))
            maj = max(int((df.y == c).sum()) for c in classes)
            print(f"  majority baseline    {maj / n_tot:.3f}")
            print(f"  briskness-only rule  {best / n_tot:.3f}   "
                  "<- IN-SAMPLE oracle, not cross-validated")
            print("    (fitted and scored on all slides, so it is an upper "
                  "bound on what briskness\n     could give -- do NOT compare "
                  "it directly to the cross-validated arm scores below.\n"
                  "     The CV-fair abundance comparator is the abundance-only "
                  "arm.)")
            if best / n_tot - maj / n_tot > 0.05:
                print("  WARNING: briskness explains much of this target. It is "
                      "only partly a spatial task;\n           an arm must clear "
                      "the briskness-only rule, not just the majority baseline.")
        print()
    else:
        df = df[["SlideID", "PatternLabels"]].dropna()
        df = df[df.PatternLabels.isin(CLASSES4)]
        df["y"] = df.PatternLabels
        classes = CLASSES4
    idx = {c: i for i, c in enumerate(classes)}
    return dict(zip(df.SlideID, df.y.map(idx))), classes


def train_eval_mil(model, bags, labels_t, tr, va, te, n_classes, epochs, lr, seed,
                   abundance=None, device="cpu", patience=20, class_weight=None,
                   select_on="macro_f1"):
    """Train on `tr`, early-stop on `va`, score `te`. Returns (accuracy, macro-F1).

    class_weight: per-class loss weights (inverse frequency) so the model is not
    rewarded for collapsing to the majority class -- essential with imbalanced
    labels, where plain cross-entropy sits at the majority predictor.
    Early stopping uses the VAL fold only, never the test fold, so test stays
    untouched until the final read.

    select_on: WHICH validation metric picks the snapshot, and it matters more
    than it looks. With a 44% majority class, a collapsed model scores 0.44 val
    ACCURACY on its first update. Genuine learning then temporarily LOWERS
    accuracy while the minority classes get sorted out -- so selecting on
    accuracy snapshots the collapsed model immediately, patience expires, and
    the run ends having learned nothing. Observed directly: best@2, stop@22
    against a 200-epoch cap.

    macro_f1 does not reward collapse -- a one-class predictor scores the floor
    (2p/(p+1)/n_classes, ~0.153 here) -- so improvement on it tracks real
    learning. It is also consistent with the inverse-frequency class weights
    already used in the loss; selecting on accuracy while training on a balanced
    objective optimises one thing and stops on another.

    MEMORY: each slide's loss is backwarded immediately and the gradients
    accumulate in .grad, rather than summing every slide's loss into one graph
    and backwarding once at the end. Identical arithmetic -- the gradient of a
    sum is the sum of the gradients -- but peak memory is one slide's autograd
    graph instead of the whole training fold's. Summing first silently defeated
    MILClassifier's regions_per_batch cap, which bounds a single slide only.
    """
    if select_on not in ("acc", "macro_f1"):
        raise ValueError(f"select_on must be 'acc' or 'macro_f1', got {select_on!r}")
    set_seed(seed)
    model = model.to(device)
    labels_t = labels_t.to(device)
    if abundance is not None:
        abundance = abundance.to(device)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Normaliser for the weighted mean, and it has to be the fold's TOTAL WEIGHT.
    # F.cross_entropy(reduction='mean') divides by the sum of weights, not by N,
    # so on a single sample it computes w_c*l/w_c = l -- the class weight cancels
    # exactly. Passing `weight` inside a per-sample loop is a silent no-op: the
    # imbalance correction never reaches the optimiser. reduction='sum' keeps the
    # w_c factor, and dividing by sum(w_y) over the fold reproduces the true
    # batched weighted loss (verified to 0.0e+00 on gradients).
    if class_weight is not None:
        idx = torch.as_tensor(np.asarray(tr), device=labels_t.device).long()
        denom = float(class_weight[labels_t[idx]].sum())
    else:
        denom = float(len(tr))

    def run(i):
        if abundance is not None:
            return model(abundance[i])
        # bags[i] is already PACKED (see pack_bag in main) -- move the packed
        # tensors to the device rather than re-packing per call. On CPU .to() is
        # a no-op; on GPU this transfers a few large tensors instead of many
        # small ones.
        bag = [tuple(t.to(device) if torch.is_tensor(t) else t for t in g)
               for g in bags[i]]
        return model(bag)[0]

    y_np = labels_t.detach().cpu().numpy()

    def preds(idx):
        return np.array([int(run(i).argmax()) for i in idx])

    # Snapshot the best-val WEIGHTS and score the test fold once at the end,
    # rather than re-scoring test on every epoch that improves val. Identical
    # result -- test is still measured at exactly the best-val point -- but it
    # drops ~20% of the forward passes, since early epochs improve constantly.
    # Safe because these models have no running buffers (no BatchNorm), so
    # state_dict fully captures them.
    best_val, best_state, since = -1.0, None, 0
    stopped_at, best_epoch = epochs, 0
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        for i in tr:
            loss = F.cross_entropy(run(i).unsqueeze(0), labels_t[i:i + 1],
                                   weight=class_weight, reduction="sum") / denom
            loss.backward()          # accumulates into .grad; graph freed here
        opt.step()
        model.eval()
        with torch.no_grad():
            vp = preds(va)
            score = (float((vp == y_np[va]).mean()) if select_on == "acc"
                     else macro_f1(vp, y_np[va], n_classes))
            if score > best_val:
                best_val, since, best_epoch = score, 0, ep
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                since += 1
                if since >= patience:
                    stopped_at = ep
                    break

    if best_state is None:                      # never improved on -1.0
        return 0.0, 0.0, stopped_at, best_epoch
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        # macro-F1 alongside accuracy: with imbalanced labels a majority
        # predictor scores respectable accuracy but near-floor macro-F1, so the
        # pair makes collapse visible where accuracy alone cannot.
        tp = preds(te)
        return (float((tp == y_np[te]).mean()),
                macro_f1(tp, y_np[te], n_classes), stopped_at, best_epoch)


def subsample_cohort(y, patient, target_n, seed=0):
    """Indices of a patient-grouped, class-stratified subsample of ~target_n.

    For learning curves: train at 40, 60, 80, ... slides and see whether
    macro-F1 trends upward with n. A rising curve says more data would help and
    the full cohort is worth waiting for; a curve flat at the collapse floor
    says the limit is representational, not sample size, and no amount of extra
    slides will fix it. That distinction is not settleable by argument.

    Subsamples the same way folds are built -- whole patients, balanced across
    classes -- so the smaller cohort is a scaled-down version of the real one
    rather than a differently-biased one. Class proportions are preserved as
    closely as whole-patient granularity allows.
    """
    rng = np.random.default_rng(seed)
    pats = {}
    for i, p in enumerate(patient):
        pats.setdefault(p, []).append(i)
    plist = list(pats)
    plabel = {p: np.bincount([y[i] for i in pats[p]]).argmax() for p in plist}

    frac = target_n / len(y)
    keep = []
    for c in sorted(set(plabel.values())):
        cp = [p for p in plist if plabel[p] == c]
        rng.shuffle(cp)
        # at least one patient per class, or the class vanishes and the task
        # silently changes shape
        take = max(1, int(round(len(cp) * frac)))
        keep.extend(cp[:take])
    idx = sorted(i for p in keep for i in pats[p])
    return idx


def corrected_t_test(diffs, n_test, n_train):
    """Nadeau & Bengio corrected resampled t-test on paired per-run differences.

    Repeated k-fold runs share training data, so they are NOT independent and a
    plain t-test (or a plain sd) badly understates the variance -- it will call
    differences significant that a rerun would not reproduce. The correction
    inflates the variance by (1/n + n_test/n_train) to account for the overlap.

    Returns (mean difference, t, p). p is two-sided; treat it as indicative at
    this sample size, not as a decisive threshold.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    var = float(d.var(ddof=1))
    if var == 0.0:
        return mean, float("nan"), (1.0 if mean == 0.0 else 0.0)
    t = mean / np.sqrt(var * (1.0 / n + n_test / max(n_train, 1)))
    return mean, float(t), float(2 * stats.t.sf(abs(t), df=n - 1))


def _make_folds(y, patient, k, seed=0):
    """Patient-grouped, stratified k-fold. Returns list of (train, val, test) idx.

    Grouped: all of a patient's slides go to the same fold (no same-patient
    leakage across train/test). Stratified: folds balanced by class as far as the
    grouping allows -- patients are assigned to folds greedily in class order so
    each fold gets a similar class mix. A val set is carved from each training
    fold for early stopping, so the test fold is never seen during training.
    """
    # One fold is test and another is val, so training gets only k-2 of k folds:
    # k=5 -> 60% train, k=3 -> 33%, k=2 -> nothing at all (and a div-by-zero on
    # len(tr) downstream). Refuse the degenerate cases rather than train on air.
    if k < 3:
        raise ValueError(f"--folds must be >= 3 (got {k}); one fold is held out "
                         "for validation, so k=2 leaves an empty training set")
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
    ap.add_argument("--task", choices=["pattern4", "arrangement", "auto"],
                    default="pattern4",
                    help="pattern4/arrangement are the TIL mappings; auto reads "
                         "--label-col as-is for any categorical slide-level task")
    ap.add_argument("--label-col", default="PatternLabels",
                    help="CSV column holding the label (--task auto only)")
    ap.add_argument("--min-class", type=int, default=5,
                    help="--task auto: drop classes with fewer slides than this")
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS,
                    help="default is the single pw-knn vs hg-knn comparison; "
                         "abundance-only always runs as the control")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--subsample", type=int, default=None,
                    help="train on a patient-grouped, class-stratified subsample "
                         "of ~N slides. For learning curves: does macro-F1 rise "
                         "with n, or is it flat at the collapse floor?")
    ap.add_argument("--subsample-seed", type=int, default=0,
                    help="which subsample to draw; vary to average over draws")
    ap.add_argument("--select-on", choices=["acc", "macro_f1"],
                    default="macro_f1",
                    help="validation metric for early stopping. acc rewards "
                         "majority collapse when classes are imbalanced; "
                         "macro_f1 does not (see train_eval_mil docstring)")
    ap.add_argument("--patience", type=int, default=20,
                    help="epochs without val improvement before stopping. MUST "
                         "scale with --epochs: training is full-batch, so an "
                         "epoch is ONE optimiser step, and patience 20 will end "
                         "a run around step 40 no matter how high the cap is")
    ap.add_argument("--seeds", type=int, default=5,
                    help="each seed = a fresh patient->fold assignment AND init")
    ap.add_argument("--seed", type=int, default=None,
                    help="run ONLY this seed (0-based). For splitting a sweep "
                         "across array tasks; merge with combine_results.py")
    ap.add_argument("--save-results", default=None,
                    help="write per-run scores + cohort fingerprint to this "
                         "JSON path, for combine_results.py")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--regions-per-batch", type=int, default=16,
                    help="regions encoded per GPU pass; lower if OOM on big slides")
    args = ap.parse_args()

    labels, classes = load_labels(args.labels, args.task,
                                  args.label_col, args.min_class)
    n_classes = len(classes)
    print(f"task={args.task} | {n_classes} classes: {classes}")

    cache_params = torch.load(os.path.join(args.graph_cache, "_params.pt"))
    print(f"graph cache: k={cache_params['k']}, "
          f"radius_um={cache_params['radius_um']}, "
          f"min_infl={cache_params['min_infl']}")
    for a in args.arms:
        if a not in cache_params["arms"]:
            raise ValueError(f"arm {a!r} not in graph cache "
                             f"(cached: {cache_params['arms']}); rerun precompute")

    files = [f for f in sorted(glob.glob(os.path.join(args.graph_cache, "*.pt")))
             if not f.endswith("_params.pt")]
    slide_bags, slide_abund, y, patient, slide_ids = [], [], [], [], []
    for f in files:
        sid = os.path.basename(f)[:-3]
        if sid not in labels:
            continue
        slide_ids.append(sid)
        d = torch.load(f)
        # the manifest describes the cache as a whole; each slide also carries
        # the params it was built under. Disagreement means the cache was
        # assembled across a parameter change and the bags are not comparable.
        stale = {k: (d.get("params", {}).get(k, "<missing>"), v)
                 for k, v in cache_params.items()
                 if k != "arms" and d.get("params", {}).get(k, "<missing>") != v}
        if stale:
            raise ValueError(
                f"{f} disagrees with {args.graph_cache}/_params.pt on "
                f"{sorted(stale)}; the cache mixes parameter settings. "
                "Rebuild it with precompute_graphs.py against a fresh --out.")
        missing = [a for a in args.arms if a not in d["bags"]]
        if missing:
            raise ValueError(f"{f} lacks arm(s) {missing}; rerun "
                             f"precompute_graphs.py --arms {' '.join(missing)}")
        slide_bags.append(d["bags"])
        slide_abund.append(d["abundance"])
        y.append(labels[sid])
        # TCGA patient = first 3 hyphen fields (TCGA-XX-YYYY); keep a patient's
        # slides in the SAME fold so same-patient leakage cannot inflate scores
        patient.append("-".join(sid.split("-")[:3]))
    n = len(y)
    # Cohort fingerprint: a hash of the exact slide set this run used.
    #
    # While the cohort is growing, every run is on different data. Two logs with
    # the SAME fingerprint are comparable; different fingerprints are separate
    # experiments, not successive refinements of one -- the folds, class balance
    # and majority baseline all move with n. Interim runs are a pilot: use them
    # to settle task/arms/hyperparameters, then freeze those and run once on the
    # final cohort. Choosing when to stop BECAUSE a result looked good is
    # optional stopping, and no p-value here corrects for it.
    # Subsample BEFORE the fingerprint, so a learning-curve point gets its own
    # cohort identity and combine_results.py cannot pool a 60-slide run with a
    # 113-slide one. They are different experiments by construction.
    if args.subsample and args.subsample < len(y):
        idx = subsample_cohort(np.array(y), patient, args.subsample,
                               seed=args.subsample_seed)
        slide_bags = [slide_bags[i] for i in idx]
        slide_abund = [slide_abund[i] for i in idx]
        slide_ids = [slide_ids[i] for i in idx]
        patient = [patient[i] for i in idx]
        y = [y[i] for i in idx]
        n = len(y)
        print(f"SUBSAMPLED to {n} slides ({len(set(patient))} patients), "
              f"target {args.subsample}, subsample-seed {args.subsample_seed} "
              f"-- learning-curve point, NOT the full cohort")

    fingerprint = hashlib.sha1("\n".join(sorted(slide_ids)).encode()).hexdigest()[:10]
    print(f"cohort {fingerprint} | {len(files)} cached slides | {n} have a label "
          f"| {len(set(patient))} unique patients")
    # Patient grouping parses TCGA-XX-YYYY out of the slide id. On any other
    # naming scheme that parse is meaningless and every slide becomes its own
    # "patient" -- which silently removes the leakage protection rather than
    # failing, so say so loudly.
    if not all(s.startswith("TCGA-") for s in slide_ids):
        print("  WARNING: some slide ids are not TCGA-formatted. Patient "
              "grouping parses TCGA-XX-YYYY and will not group them, so "
              "same-patient slides could span train/test.")
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
    # Match on the whole model. Since models.REGION_DIM pins the encoder output
    # width, the pool and head are identical across arms and this is equivalent
    # to matching encoders -- but stated on the total, which is the quantity
    # that actually has to be equal. hg-knn/pw-knn stand in for their families:
    # every hg-* shares HyperRegionEncoder and every pw-* PairwiseRegionEncoder,
    # so parameter count depends only on the prefix.
    def _mil(arm_prefix):
        return lambda i, h, o: MILClassifier(arm_prefix, i, h, o)
    target = n_params(_mil("hg-knn")(in_dim, args.hidden, n_classes))
    pw_h = matched_hidden(_mil("pw-knn"), target, in_dim, n_classes)
    hidden = {a: (pw_h if a.startswith("pw-") else args.hidden) for a in args.arms}
    pw_total = n_params(_mil("pw-knn")(in_dim, pw_h, n_classes))
    print(f"capacity: hypergraph model {target:,} params (hidden={args.hidden}) "
          f"-> pairwise hidden={pw_h} = {pw_total:,} params "
          f"({pw_total / target:.2f}x target)")

    # Report every arm's total, and assert the shared MIL stage really is shared.
    # REGION_DIM makes this true by construction; asserting it means a later
    # change to att_dim, the readout, or n_classes cannot silently reintroduce
    # the mismatch that encoder-only matching used to create.
    built_models = {a: MILClassifier(a, in_dim, hidden[a], n_classes)
                    for a in args.arms}
    shared = {a: n_params(m) - n_params(m.encoder) for a, m in built_models.items()}
    if len(set(shared.values())) != 1:
        raise RuntimeError(
            f"MIL pool+head differ across arms: {shared}. Region encoders must "
            "emit a fixed width (models.REGION_DIM) or capacity matching is unsound.")
    totals = {a: n_params(m) for a, m in built_models.items()}
    print("  per-arm totals: " + " | ".join(f"{a}={totals[a]:,}"
                                            for a in args.arms))
    print(f"  pool+head {next(iter(shared.values())):,} params, identical across "
          f"arms | total spread {max(totals.values()) / min(totals.values()):.3f}x")
    print("  (abundance-only is a different model class by design -- it is the "
          "triviality control, not a capacity-matched arm)")

    # Resample the folds per seed. With one fixed fold assignment, `--seeds` only
    # varied initialisation and the reported sd understated the true variance --
    # the dominant source of variance at this n is WHICH patients land in test,
    # not the init. This is repeated k-fold CV: seed s gives its own patient->fold
    # assignment, identical across arms, so arm-vs-arm stays paired.
    #
    # --seed runs ONE seed only, so a sweep can be split across array tasks.
    # Splitting sacrifices nothing: _make_folds is a deterministic function of
    # (cohort, k, seed), and the cohort comes from sorted(glob(...)), so seed 1
    # computed in a separate job is bit-identical to seed 1 computed inline.
    # Arms stay paired run-for-run, which is what the corrected test needs.
    # combine_results.py merges the parts (and refuses to merge across cohorts).
    seed_list = [args.seed] if args.seed is not None else list(range(args.seeds))
    fold_sets = {s: _make_folds(y.numpy(), patient, args.folds, seed=s)
                 for s in seed_list}
    runs = [(s, tr, va, te) for s in seed_list for tr, va, te in fold_sets[s]]
    if args.seed is not None:
        print(f"SEED {args.seed} ONLY -- this is one part of a split sweep; "
              f"merge with combine_results.py")
    print(f"{args.folds}-fold patient-grouped stratified CV, resampled per seed; "
          f"seeds {seed_list} -> {len(runs)} runs/arm")
    print(f"fold test sizes (seed {seed_list[0]}) "
          f"{[len(te) for _, _, te in fold_sets[seed_list[0]]]}\n")

    # evaluate one arm across all seed x fold runs -> flat list of test accuracies.
    # `runs` is a fixed ordered list, so scores line up index-for-index between
    # arms and the beat-rate below is a genuine paired comparison.
    def eval_arm(arm):
        scores = []
        # Pack every slide ONCE for this arm, then reuse across all runs and all
        # epochs. Previously forward() re-packed on every pass -- 15 runs x ~40
        # epochs x ~113 slides of identical torch.cat work.
        if arm == "abundance-only":
            bags = None
        else:
            t_pack = time.time()
            bags = [pack_bag(sb[arm], arm.startswith("pw-"),
                             args.regions_per_batch) for sb in slide_bags]
            print(f"  [{arm}] packed {len(bags)} slides in "
                  f"{time.time() - t_pack:.1f}s (once, reused by every run)",
                  flush=True)
        # Per-run progress, flushed. Without this an arm is silent for its whole
        # 15-50 runs, so a walltime kill loses every result AND you cannot tell a
        # slow job from a hung one. The rate also lets you size the next run.
        t_arm = time.time()
        print(f"  [{arm}] {len(runs)} runs ...", flush=True)
        for j, (s, tr, va, te) in enumerate(runs, 1):
            # seed BEFORE constructing the model, for every arm including the
            # control -- otherwise the control's init depends on whatever RNG
            # state the previous arm happened to leave behind.
            set_seed(s)
            if arm == "abundance-only":
                m = AbundanceOnly(abund.shape[1], 32, n_classes)
                r = train_eval_mil(m, None, y, tr, va, te, n_classes, args.epochs,
                                   0.01, s, abundance=abund,
                                   device=args.device, class_weight=class_weight,
                                   patience=args.patience,
                                   select_on=args.select_on)
            else:
                m = MILClassifier(arm, in_dim, hidden[arm], n_classes,
                                  regions_per_batch=args.regions_per_batch)
                r = train_eval_mil(m, bags, y, tr, va, te, n_classes, args.epochs,
                                   0.01, s, device=args.device,
                                   class_weight=class_weight,
                                   patience=args.patience,
                                   select_on=args.select_on)
            scores.append(r[:2])
            el = time.time() - t_arm
            # stop@ / best@ reveal whether --epochs actually bound. If stop@ is
            # far below the cap, the run ended on PATIENCE and raising the cap
            # alone changes nothing -- patience has to rise with it.
            cap = "CAP" if r[2] >= args.epochs else "patience"
            print(f"    [{arm}] run {j}/{len(runs)} (seed {s}) "
                  f"acc={r[0]:.3f} f1={r[1]:.3f} | "
                  f"stop@{r[2]} best@{r[3]} ({cap}) | "
                  f"{el / j:.0f}s/run | eta {(len(runs) - j) * el / j / 60:.0f}min",
                  flush=True)
        a = np.array(scores)                      # (n_runs, 2) = acc, macro-F1
        return a[:, 0], a[:, 1]

    maj = float((y == y.bincount().argmax()).float().mean())
    n_te = float(np.mean([len(te) for _, _, _, te in runs]))
    n_tr = float(np.mean([len(tr) for _, tr, _, _ in runs]))

    # macro-F1 a majority predictor would score: one class gets F1 =
    # 2p/(p+1) for p = its prevalence, every other class gets 0. Printing this
    # floor makes collapse readable directly off the table -- accuracy alone
    # cannot distinguish a model that learned nothing from one that did.
    floor_f1 = (2 * maj / (maj + 1)) / n_classes
    print(f"=== test scores over seeds {seed_list} x {args.folds} folds "
          f"({len(runs)} runs/arm) ===")
    print(f"  {'majority baseline':<18} acc {maj:.3f} | macroF1 {floor_f1:.3f} "
          f"<- the collapse floor")
    ab, ab_f1 = eval_arm("abundance-only")
    print(f"  {'abundance-only':<18} acc {ab.mean():.3f} +- {ab.std():.3f}"
          f" | macroF1 {ab_f1.mean():.3f}")
    print(f"  {'':<18} {'':<4}(the bar every arm below must clear)\n")

    per_arm = {"abundance-only": {"acc": ab.tolist(), "f1": ab_f1.tolist()}}
    for arm in args.arms:
        sc, f1 = eval_arm(arm)
        per_arm[arm] = {"acc": sc.tolist(), "f1": f1.tolist()}
        beat = float((sc > ab).mean())
        mean_d, _t, p = corrected_t_test(sc - ab, n_te, n_tr)
        sig = "n/a" if np.isnan(p) else f"p={p:.3f}"
        print(f"  {arm:<18} acc {sc.mean():.3f} +- {sc.std():.3f}"
              f" | macroF1 {f1.mean():.3f}")
        # Express the difference in TEST SLIDES as well as accuracy. With ~23
        # slides per fold, accuracy moves in steps of 1/23 = 0.043, so a "+0.044"
        # is one slide. Stating that inline stops a sub-slide difference reading
        # as an effect, however small the p-value looks.
        slides = mean_d * n_te
        print(f"  {'':<18} vs abundance {mean_d:+.3f} ({slides:+.1f} of "
              f"{n_te:.0f} test slides) ({sig}, corrected) "
              f"| wins {beat:.0%} of paired runs")
        if abs(slides) < 1.5:
            print(f"  {'':<18} ^ under 1.5 slides per fold -- at or below the "
                  f"resolution of this test set, regardless of p")

    if args.save_results:
        # Everything combine_results.py needs to merge parts SAFELY. The cohort
        # fingerprint is the guard: if segmentation lands between array tasks the
        # slide set changes, folds change, and merging would silently splice two
        # different experiments together.
        os.makedirs(os.path.dirname(os.path.abspath(args.save_results)) or ".",
                    exist_ok=True)
        with open(args.save_results, "w") as fh:
            json.dump({
                "cohort": fingerprint,
                "n_slides": n,
                # recorded so a learning curve can be plotted straight from the
                # JSONs, and so a subsampled run is never mistaken for a full one
                "subsample": args.subsample,
                "subsample_seed": args.subsample_seed,
                "n_patients": len(set(patient)),
                "task": args.task,
                "classes": classes,
                "class_counts": counts,
                "majority_baseline": maj,
                "seeds": seed_list,
                "folds": args.folds,
                "epochs": args.epochs,
                "arms": list(args.arms),
                "n_test_mean": n_te,
                "n_train_mean": n_tr,
                # run order is (seed, fold) as generated, identical across arms,
                # so scores stay paired index-for-index after merging
                "runs": [{"seed": int(s), "n_test": int(len(te))}
                         for s, _, _, te in runs],
                "scores": per_arm,
            }, fh, indent=2)
        print(f"\nwrote {args.save_results} (cohort {fingerprint})")

if __name__ == "__main__":
    main()