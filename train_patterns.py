"""STAGE 2: predict a slide's Saltz TIL PatternLabel from its cached region graphs.

Labels are slide-level, so this is multiple-instance learning: each slide is a
bag of region graphs aggregated to one prediction (see models.py). The
abundance-only control must be cleared before any spatial claim holds.

Small n -- read the corrected p and the win rate together, not the mean alone.

    qsub scripts/run_patterns.sh
    qsub -t 1-10 scripts/run_patterns_array.sh   # split by seed
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
                    MILClassifier, AbundanceOnly, pack_bag, parse_arm,
                    pack_mode)

# 4-way and the collapsed spatial-only ("arrangement") mapping
CLASSES4 = ["Brisk Diffuse", "Brisk Band-like", "Non-Brisk Focal", "Non-Brisk Multifocal"]

# Briskness and arrangement are perfectly collinear across these four labels:
# Brisk only appears with Diffuse/Band-like, Non-Brisk only with
# Focal/Multifocal. So grouping {Diffuse, Band-like} vs {Focal, Multifocal} is
# just Brisk vs Non-Brisk -- an abundance task with a spatial name.
#
# Crossing briskness removes that collinearity but NOT the association: on 113
# slides a briskness-only rule still scored 0.717 vs a 0.558 majority.
# load_labels() prints that figure every run.
#
# The axis is "one contiguous locus" vs "several / dispersed". That is a domain
# judgement about Saltz's taxonomy -- check it against your label source.
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
        # A mixed cross-tab is necessary but not sufficient -- the best
        # briskness-only rule is what says how confounded the task is.
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
            print("    (an upper bound on briskness alone; the CV-fair "
                  "comparator is the abundance-only arm)")
            if best / n_tot - maj / n_tot > 0.05:
                print("  WARNING: briskness explains much of this target -- an "
                      "arm must clear the\n           briskness-only rule, not "
                      "just the majority baseline.")
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
                   select_on="macro_f1", batch_size=8):
    """Train on `tr`, early-stop on `va`, score `te`. Returns (accuracy, macro-F1).

    batch_size: slides per optimiser step. This was FULL BATCH -- gradients
    accumulated over every training slide and one opt.step() per epoch -- which
    made an epoch a single optimiser step. With patience 20, every run then
    stopped near step 30 whatever the epoch cap was, so models trained for about
    ten gradient steps and the sweep compared early-training trajectories rather
    than converged models. Observed: best@2..13, stop@22..33, against a cap of
    150. Minibatching gives ~n/batch steps per epoch for the same number of
    forward/backward passes, so it costs no wall-clock. batch_size<=0 restores
    the old full-batch behaviour.

    class_weight: per-class inverse-frequency loss weights, so the model is not
    rewarded for collapsing to the majority class.

    select_on: selecting on val ACCURACY snapshots a collapsed model on its first
    update (0.44 with a 44% majority) and patience expires before real learning
    recovers -- observed as best@2, stop@22 against a 200-epoch cap. macro_f1
    scores collapse at the floor instead, and matches the weighted loss.

    Each slide's loss is backwarded immediately so gradients accumulate in .grad;
    summing into one graph first is the same arithmetic but defeats
    MILClassifier's regions_per_batch cap, which bounds one slide only.
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

    # Must be the fold's TOTAL WEIGHT. cross_entropy(reduction='mean') divides by
    # the sum of weights, so on a single sample w_c*l/w_c = l and the class weight
    # cancels -- passing `weight` inside a per-sample loop is a silent no-op.
    # reduction='sum' over sum(w_y) reproduces the true batched weighted loss.
    def _denom(chunk):
        """Total loss weight of a minibatch -- the correct weighted-mean divisor.

        Must be the CHUNK's weight, not the fold's. Dividing a minibatch by the
        whole fold's weight would shrink every gradient by batch/n and silently
        scale the learning rate down by the same factor.
        """
        if class_weight is None:
            return float(len(chunk))
        idx = torch.as_tensor(np.asarray(chunk), device=labels_t.device).long()
        return float(class_weight[labels_t[idx]].sum())

    def run(i, ab_i=None, bag_i=None):
        """Score slide i. ab_i/bag_i override WHICH slide each input comes from,
        which is how the permutation ablations are done."""
        if bags is None:                       # the abundance-only control
            return model(abundance[i if ab_i is None else ab_i])
        # bags[i] is already packed (pack_bag in main) -- move a few large
        # tensors rather than re-packing per call
        bag = [tuple(t.to(device) if torch.is_tensor(t) else t for t in g)
               for g in bags[i if bag_i is None else bag_i]]
        a = None if abundance is None else abundance[i if ab_i is None else ab_i]
        return model(bag, a)[0]

    y_np = labels_t.detach().cpu().numpy()

    def preds(idx, src=None, what=None):
        """src: parallel array of slides to take `what` ('ab'|'bag') from."""
        if src is None:
            return np.array([int(run(i).argmax()) for i in idx])
        kw = "ab_i" if what == "ab" else "bag_i"
        return np.array([int(run(i, **{kw: j}).argmax())
                         for i, j in zip(idx, src)])

    # Snapshot best-val weights and score test once at the end -- same result as
    # re-scoring test on every improvement, ~20% fewer forward passes. Safe
    # because no model here has running buffers.
    best_val, best_state, since = -1.0, None, 0
    stopped_at, best_epoch = epochs, 0
    rng = np.random.default_rng(seed)          # own stream, so shuffling cannot
    tr = np.asarray(tr)                        # perturb model init reproducibly
    bs = len(tr) if batch_size <= 0 else min(batch_size, len(tr))
    for ep in range(1, epochs + 1):
        model.train()
        order = tr[rng.permutation(len(tr))]
        for start in range(0, len(order), bs):
            chunk = order[start:start + bs]
            d = _denom(chunk)
            opt.zero_grad()
            for i in chunk:
                loss = F.cross_entropy(run(i).unsqueeze(0), labels_t[i:i + 1],
                                       weight=class_weight, reduction="sum") / d
                loss.backward()      # accumulates into .grad; graph freed here
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
        return 0.0, 0.0, stopped_at, best_epoch, {}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        # macro-F1 alongside accuracy makes majority collapse visible
        tp = preds(te)
        acc, f1 = float((tp == y_np[te]).mean()), macro_f1(tp, y_np[te], n_classes)
        # PERMUTATION ablations, not zeroing.
        #
        # Zeroing a path measures which branch won the optimisation, not what
        # its input carries: both feed one head, so whichever gets more training
        # pressure captures it and the other atrophies. Measured on synthetic
        # data where abundance IS the signal, the same two numbers went
        # 0.907/0.178 at path_dropout=0.1 and 0.102/0.979 at 0.5 -- swinging
        # from one extreme to the other on a parameter with no principled value.
        #
        # Permuting feeds a VALID input from the wrong slide. Same marginal
        # distribution, association destroyed, so nothing is out-of-distribution
        # and nothing depends on training dynamics. The drop from `full` is how
        # much the trained model relied on that input.
        abl = {}
        if bags is not None:
            pr = np.asarray(te)[rng.permutation(len(te))]
            if getattr(model, "abundance_dim", 0):
                pa = preds(te, src=pr, what="ab")
                abl["f1_abundance_permuted"] = macro_f1(pa, y_np[te], n_classes)
            pg = preds(te, src=pr, what="bag")
            abl["f1_graph_permuted"] = macro_f1(pg, y_np[te], n_classes)
        return acc, f1, stopped_at, best_epoch, abl


def subsample_cohort(y, patient, target_n, seed=0):
    """Indices of a patient-grouped, class-stratified subsample of ~target_n.

    For learning curves: a curve rising with n says more data would help; one
    flat at the collapse floor says the limit is representational.

    Drawn the way folds are built -- whole patients, class-balanced -- so the
    small cohort is a scaled-down version of the real one, not a biased one.
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

    Repeated k-fold runs share training data, so a plain t-test understates the
    variance and calls differences significant that a rerun would not reproduce.
    The correction inflates it by (1/n + n_test/n_train).

    Returns (mean difference, t, two-sided p). At this n, p is indicative.
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

    Grouped so a patient's slides never straddle train and test; stratified by
    round-robining each class's patients across folds. Val is carved from another
    fold, so test is unseen during training.
    """
    # test + val take one fold each, so training gets k-2 of k: k=3 -> 33%,
    # k=2 -> nothing (and a div-by-zero on len(tr) downstream)
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
                    help="patient-grouped, class-stratified subsample of ~N "
                         "slides, for learning curves")
    ap.add_argument("--subsample-seed", type=int, default=0,
                    help="which subsample to draw; vary to average over draws")
    ap.add_argument("--abundance-skip", action="store_true",
                    help="concatenate the per-slide cell-type fractions to the "
                         "pooled slide vector, so the head starts from "
                         "at-least-abundance and topology can only add. A "
                         "DIFFERENT experiment from the default, not a fix to "
                         "it: it asks whether structure adds given composition")
    ap.add_argument("--path-dropout", type=float, default=0.0,
                    help="per training step, zero one path at random with "
                         "this probability each. OFF by default -- the "
                         "permutation ablations need no such training, and the "
                         "dropout drives a winner-take-all through the shared "
                         "head that swings the ablations wildly. Max 0.5")
    ap.add_argument("--region-dim", type=int, default=64,
                    help="width every encoder emits. With --att-dim this sets "
                         "the pool+head cost, which is FIXED regardless of "
                         "--hidden and is ~80%% of the model once hidden is 8. "
                         "This is the real capacity knob at small n")
    ap.add_argument("--att-dim", type=int, default=64,
                    help="attention hidden width; see --region-dim")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="slides per optimiser step. Was full-batch, which made "
                         "an epoch ONE step and left every run stopping on "
                         "patience after ~10 useful steps. 0 restores that")
    ap.add_argument("--select-on", choices=["acc", "macro_f1"],
                    default="macro_f1",
                    help="early-stopping metric; acc rewards majority collapse "
                         "when classes are imbalanced, macro_f1 does not")
    ap.add_argument("--patience", type=int, default=20,
                    help="epochs without val improvement before stopping. Must "
                         "scale with --epochs: an epoch is one optimiser step, "
                         "so patience 20 ends a run near step 40 at any cap")
    ap.add_argument("--seeds", type=int, default=5,
                    help="each seed = a fresh patient->fold assignment AND init")
    ap.add_argument("--seed", type=int, default=None,
                    help="run ONLY this seed (0-based). For splitting a sweep "
                         "across array tasks; merge with combine_results.py")
    ap.add_argument("--save-results", default=None,
                    help="write per-run scores + cohort fingerprint to this "
                         "JSON path, for combine_results.py")
    ap.add_argument("--star-layers", type=int, default=4,
                    help="depth of @star arms. 4 reach-matches a 2-layer "
                         "hypergraph arm (one hypergraph layer = two star hops) "
                         "and is a DELIBERATE choice favouring the baseline; 2 "
                         "matches layer count instead. Sweep and report the "
                         "depth at which star catches up")
    ap.add_argument("--blend-families", action="store_true",
                    help="ABLATION: pool both hyperedge families into one "
                         "vector instead of keeping them separable. Shows what "
                         "the +semantic arm does if a 200-cell semantic "
                         "hyperedge and a 5-cell spatial one are treated alike")
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
        if parse_arm(a)[0] not in cache_params["arms"]:
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
        # each slide carries the params it was built under; disagreement with the
        # manifest means the cache spans a parameter change
        stale = {k: (d.get("params", {}).get(k, "<missing>"), v)
                 for k, v in cache_params.items()
                 if k != "arms" and d.get("params", {}).get(k, "<missing>") != v}
        if stale:
            raise ValueError(
                f"{f} disagrees with {args.graph_cache}/_params.pt on "
                f"{sorted(stale)}; the cache mixes parameter settings. "
                "Rebuild it with precompute_graphs.py against a fresh --out.")
        # the cache is keyed by construction only -- aggregation is a model choice
        missing = [a for a in args.arms if parse_arm(a)[0] not in d["bags"]]
        if missing:
            raise ValueError(f"{f} lacks arm(s) {missing}; rerun "
                             f"precompute_graphs.py --arms {' '.join(missing)}")
        slide_bags.append(d["bags"])
        slide_abund.append(d["abundance"])
        y.append(labels[sid])
        # TCGA-XX-YYYY; a patient's slides must share a fold (no leakage)
        patient.append("-".join(sid.split("-")[:3]))
    n = len(y)
    # Fingerprint = hash of the exact slide set. While the cohort grows, only
    # runs sharing a fingerprint are comparable. Subsample first, so each
    # learning-curve point gets its own identity and cannot be pooled.
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
    # on non-TCGA ids the patient parse is meaningless and every slide becomes
    # its own "patient", silently removing the leakage protection
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

    in_dim = slide_bags[0][parse_arm(args.arms[0])[0]][0][0].shape[1]
    # Match on the whole model, per arm rather than per family: GCNConv is one
    # transform, SumHyperConv one (on 2*in_dim), DeepSetsHyperConv three, so
    # "all hg-* cost the same" would leave hg-knn@sum badly under-capacity.
    def _mil(arm_name):
        return lambda i, h, o: MILClassifier(arm_name, i, h, o,
                                            blend_families=args.blend_families,
                                            star_layers=args.star_layers,
                                            region_dim=args.region_dim, att_dim=args.att_dim,
                                            abundance_dim=(abund.shape[1] if args.abundance_skip else 0),
                                            path_dropout=args.path_dropout)
    # reference: the Deep Sets hypergraph at the requested hidden dim
    target = n_params(_mil("hg-knn")(in_dim, args.hidden, n_classes))
    hidden = {}
    for a in args.arms:
        hidden[a] = (args.hidden if a == "hg-knn"
                     else matched_hidden(_mil(a), target, in_dim, n_classes))
    print(f"capacity target {target:,} params (hg-knn @ hidden={args.hidden})")
    for a in args.arms:
        tot = n_params(_mil(a)(in_dim, hidden[a], n_classes))
        print(f"  {a:<16} hidden={hidden[a]:<4} {tot:>7,} params "
              f"({tot / target:.2f}x target)")

    # REGION_DIM makes the pool+head identical by construction; assert it so a
    # later change to att_dim or the readout cannot silently undo that
    built_models = {a: MILClassifier(a, in_dim, hidden[a], n_classes,
                                     blend_families=args.blend_families,
                                     star_layers=args.star_layers,
                                     region_dim=args.region_dim, att_dim=args.att_dim,
                                     abundance_dim=(abund.shape[1] if args.abundance_skip else 0),
                                     path_dropout=args.path_dropout)
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
    if any(parse_arm(a)[1] == "star" for a in args.arms):
        note = ("reach-matched to a 2-layer hypergraph arm; favours the baseline"
                if args.star_layers >= 4 else
                "layer-matched, so the hypergraph arm holds ~2x the reach")
        print(f"  star arms at depth {args.star_layers} -- {note}. One "
              f"hypergraph layer is two\n  star hops, so this is an active "
              f"choice, not a default. Sweep --star-layers to\n  report the "
              f"depth at which star catches up.")

    # Repeated k-fold: each seed resamples the patient->fold assignment, since at
    # this n which patients land in test dominates the variance. The assignment is
    # shared across arms, so comparisons stay paired.
    #
    # --seed runs one seed, so a sweep can be split across array tasks -- folds are
    # deterministic in (cohort, k, seed), so a split run is bit-identical.
    seed_list = [args.seed] if args.seed is not None else list(range(args.seeds))
    fold_sets = {s: _make_folds(y.numpy(), patient, args.folds, seed=s)
                 for s in seed_list}
    runs = [(s, tr, va, te) for s in seed_list for tr, va, te in fold_sets[s]]
    if args.seed is not None:
        print(f"SEED {args.seed} ONLY -- this is one part of a split sweep; "
              f"merge with combine_results.py")
    # An epoch used to be ONE optimiser step, so `epochs` and `patience` meant
    # something very different from what they read like. Print the conversion.
    n_tr_mean = float(np.mean([len(t) for _, t, _, _ in runs]))
    steps = (max(1, int(np.ceil(n_tr_mean / args.batch_size)))
             if args.batch_size > 0 else 1)
    print(f"batch {args.batch_size if args.batch_size > 0 else 'full'} "
          f"slides/step -> ~{steps} optimiser steps/epoch, "
          f"<={steps * args.epochs:,} per run "
          f"(patience {args.patience} epochs = {steps * args.patience:,} steps)")
    print(f"{args.folds}-fold patient-grouped stratified CV, resampled per seed; "
          f"seeds {seed_list} -> {len(runs)} runs/arm")
    print(f"fold test sizes (seed {seed_list[0]}) "
          f"{[len(te) for _, _, te in fold_sets[seed_list[0]]]}\n")

    # `runs` is fixed and ordered, so scores line up index-for-index between arms
    def eval_arm(arm):
        scores, abls = [], []
        # pack once per arm and reuse across every run and epoch
        if arm == "abundance-only":
            bags = None
        else:
            t_pack = time.time()
            construction, _ = parse_arm(arm)
            # pack by ARM, not construction: a star arm reads the hypergraph
            # cache but needs the bipartite packer, so the construction half of
            # the name does not determine the encoding
            bags = [pack_bag(sb[construction], pack_mode(arm),
                             args.regions_per_batch) for sb in slide_bags]
            print(f"  [{arm}] packed {len(bags)} slides in "
                  f"{time.time() - t_pack:.1f}s (once, reused by every run)",
                  flush=True)
        # flushed per-run progress, so a walltime kill still leaves a rate to
        # size the next job by
        t_arm = time.time()
        print(f"  [{arm}] {len(runs)} runs ...", flush=True)
        for j, (s, tr, va, te) in enumerate(runs, 1):
            # seed before constructing the model, or init depends on whatever
            # RNG state the previous arm left behind
            set_seed(s)
            if arm == "abundance-only":
                m = AbundanceOnly(abund.shape[1], 32, n_classes)
                r = train_eval_mil(m, None, y, tr, va, te, n_classes, args.epochs,
                                   0.01, s, abundance=abund,
                                   device=args.device, class_weight=class_weight,
                                   patience=args.patience,
                                   select_on=args.select_on,
                                   batch_size=args.batch_size)
            else:
                m = MILClassifier(arm, in_dim, hidden[arm], n_classes,
                                  regions_per_batch=args.regions_per_batch,
                                  blend_families=args.blend_families,
                                  star_layers=args.star_layers,
                                  region_dim=args.region_dim, att_dim=args.att_dim,
                                  abundance_dim=(abund.shape[1] if args.abundance_skip else 0),
                                  path_dropout=args.path_dropout)
                r = train_eval_mil(m, bags, y, tr, va, te, n_classes, args.epochs,
                                   0.01, s,
                                   abundance=(abund if args.abundance_skip else None),
                                   device=args.device,
                                   class_weight=class_weight,
                                   patience=args.patience,
                                   select_on=args.select_on,
                                   batch_size=args.batch_size)
            scores.append(r[:2])
            if r[4]:
                abls.append(r[4])
            el = time.time() - t_arm
            # stop@ far below the cap means patience bound, not --epochs
            cap = "CAP" if r[2] >= args.epochs else "patience"
            print(f"    [{arm}] run {j}/{len(runs)} (seed {s}) "
                  f"acc={r[0]:.3f} f1={r[1]:.3f} | "
                  f"stop@{r[2]} best@{r[3]} ({cap}) | "
                  f"{el / j:.0f}s/run | eta {(len(runs) - j) * el / j / 60:.0f}min",
                  flush=True)
        a = np.array(scores)                      # (n_runs, 2) = acc, macro-F1
        if abls:
            m = {k: float(np.mean([d[k] for d in abls])) for k in abls[0]}
            full = a[:, 1].mean()
            bits = [f"full {full:.3f}"]
            if "f1_graph_permuted" in m:
                bits.append(f"graph permuted {m['f1_graph_permuted']:.3f} "
                            f"(-{full - m['f1_graph_permuted']:.3f})")
            if "f1_abundance_permuted" in m:
                bits.append(f"abundance permuted {m['f1_abundance_permuted']:.3f} "
                            f"(-{full - m['f1_abundance_permuted']:.3f})")
            print(f"    [{arm}] " + " | ".join(bits), flush=True)
            ablations[arm] = dict(m, f1_full=float(full))
        return a[:, 0], a[:, 1]

    ablations = {}
    maj = float((y == y.bincount().argmax()).float().mean())
    n_te = float(np.mean([len(te) for _, _, _, te in runs]))
    n_tr = float(np.mean([len(tr) for _, tr, _, _ in runs]))

    # macro-F1 of a majority predictor: 2p/(p+1) on one class, 0 on the rest
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
        # also in test slides: at ~23 per fold accuracy moves in steps of 0.043,
        # so "+0.044" is one slide
        slides = mean_d * n_te
        print(f"  {'':<18} vs abundance {mean_d:+.3f} ({slides:+.1f} of "
              f"{n_te:.0f} test slides) ({sig}, corrected) "
              f"| wins {beat:.0%} of paired runs")
        if abs(slides) < 1.5:
            print(f"  {'':<18} ^ under 1.5 slides per fold -- at or below the "
                  f"resolution of this test set, regardless of p")

    if args.save_results:
        # the fingerprint guards the merge: if segmentation lands between array
        # tasks the slide set changes and the parts are different experiments
        os.makedirs(os.path.dirname(os.path.abspath(args.save_results)) or ".",
                    exist_ok=True)
        with open(args.save_results, "w") as fh:
            json.dump({
                "cohort": fingerprint,
                "n_slides": n,
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
                # recorded because it is a design decision, not a nuisance
                # parameter: it sets how much reach the star baseline gets
                "star_layers": args.star_layers,
                "abundance_skip": bool(args.abundance_skip),
                "path_dropout": args.path_dropout,
                "ablations": ablations,
                "batch_size": args.batch_size,
                "region_dim": args.region_dim,
                "att_dim": args.att_dim,
                "blend_families": bool(args.blend_families),
                "arms": list(args.arms),
                "n_test_mean": n_te,
                "n_train_mean": n_tr,
                # (seed, fold) order, identical across arms, so scores stay
                # paired index-for-index after merging
                "runs": [{"seed": int(s), "n_test": int(len(te))}
                         for s, _, _, te in runs],
                "scores": per_arm,
            }, fh, indent=2)
        print(f"\nwrote {args.save_results} (cohort {fingerprint})")

if __name__ == "__main__":
    main()