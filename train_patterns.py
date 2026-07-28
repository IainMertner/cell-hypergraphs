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
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

from graphs import N_TYPES, DEFAULT_ARMS
from models import (set_seed, matched_hidden, n_params, macro_f1,
                    MILClassifier, AbundanceOnly)

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
# Isolating arrangement means CROSSING briskness, so each target class holds one
# Brisk and one Non-Brisk label and abundance carries no information about it.
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


def load_labels(csv_path, task):
    df = pd.read_csv(csv_path)
    df = df[["SlideID", "PatternLabels"]].dropna()
    if task == "arrangement":
        df = df[df.PatternLabels.isin(ARRANGEMENT)]
        df["y"] = df.PatternLabels.map(ARRANGEMENT)
        classes = ["focused", "dispersed"]
        # Show the briskness balance. If either row is all-Brisk or all-Non-Brisk
        # the target has collapsed back onto abundance and no spatial claim can
        # survive -- this is the check that would have caught the old mapping.
        brisk = df.PatternLabels.str.startswith("Brisk")
        print("briskness x arrangement (must be mixed on BOTH rows):")
        for c in classes:
            m = df.y == c
            print(f"  {c:<10} Brisk {int((m & brisk).sum()):>3} | "
                  f"Non-Brisk {int((m & ~brisk).sum()):>3}")
        if any((df.y == c).sum() and ((df.y == c) & brisk).nunique() == 1
               for c in classes):
            print("  WARNING: a class is pure in briskness -- this is an "
                  "abundance task, not an arrangement one.")
        print()
    else:
        df = df[df.PatternLabels.isin(CLASSES4)]
        df["y"] = df.PatternLabels
        classes = CLASSES4
    idx = {c: i for i, c in enumerate(classes)}
    return dict(zip(df.SlideID, df.y.map(idx))), classes


def train_eval_mil(model, bags, labels_t, tr, va, te, n_classes, epochs, lr, seed,
                   abundance=None, device="cpu", patience=20, class_weight=None):
    """Train on `tr`, early-stop on `va`, score `te`. Returns (accuracy, macro-F1).

    class_weight: per-class loss weights (inverse frequency) so the model is not
    rewarded for collapsing to the majority class -- essential with imbalanced
    labels, where plain cross-entropy sits at the majority predictor.
    Early stopping uses the VAL fold only, never the test fold, so test stays
    untouched until the final read.

    MEMORY: each slide's loss is backwarded immediately and the gradients
    accumulate in .grad, rather than summing every slide's loss into one graph
    and backwarding once at the end. Identical arithmetic -- the gradient of a
    sum is the sum of the gradients -- but peak memory is one slide's autograd
    graph instead of the whole training fold's. Summing first silently defeated
    MILClassifier's regions_per_batch cap, which bounds a single slide only.
    """
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
        bag = [(x.to(device), s.to(device)) for x, s in bags[i]]
        return model(bag)[0]

    y_np = labels_t.detach().cpu().numpy()

    def preds(idx):
        return np.array([int(run(i).argmax()) for i in idx])

    best_val, best, since = -1.0, (0.0, 0.0), 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        for i in tr:
            loss = F.cross_entropy(run(i).unsqueeze(0), labels_t[i:i + 1],
                                   weight=class_weight, reduction="sum") / denom
            loss.backward()          # accumulates into .grad; graph freed here
        opt.step()
        model.eval()
        with torch.no_grad():
            va_acc = float((preds(va) == y_np[va]).mean())
            if va_acc > best_val:
                # Report macro-F1 as well as accuracy. With imbalanced labels a
                # majority predictor scores respectable accuracy but near-floor
                # macro-F1, so the pair makes collapse visible; accuracy alone
                # cannot distinguish a real model from one that learned nothing.
                tp = preds(te)
                best_val, since = va_acc, 0
                best = (float((tp == y_np[te]).mean()),
                        macro_f1(tp, y_np[te], n_classes))
            else:
                since += 1
                if since >= patience:
                    break
    return best


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
    ap.add_argument("--task", choices=["pattern4", "arrangement"], default="pattern4")
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS,
                    help="default is the single pw-knn vs hg-knn comparison; "
                         "abundance-only always runs as the control")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5,
                    help="each seed = a fresh patient->fold assignment AND init")
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
    fingerprint = hashlib.sha1("\n".join(sorted(slide_ids)).encode()).hexdigest()[:10]
    print(f"cohort {fingerprint} | {len(files)} cached slides | {n} have a label "
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
    fold_sets = [_make_folds(y.numpy(), patient, args.folds, seed=s)
                 for s in range(args.seeds)]
    runs = [(s, tr, va, te) for s in range(args.seeds)
            for tr, va, te in fold_sets[s]]
    print(f"{args.folds}-fold patient-grouped stratified CV, resampled per seed; "
          f"{len(runs)} runs/arm")
    print(f"fold test sizes (seed 0) {[len(te) for _, _, te in fold_sets[0]]}\n")

    # evaluate one arm across all seed x fold runs -> flat list of test accuracies.
    # `runs` is a fixed ordered list, so scores line up index-for-index between
    # arms and the beat-rate below is a genuine paired comparison.
    def eval_arm(arm):
        scores = []
        bags = None if arm == "abundance-only" else [sb[arm] for sb in slide_bags]
        for s, tr, va, te in runs:
            # seed BEFORE constructing the model, for every arm including the
            # control -- otherwise the control's init depends on whatever RNG
            # state the previous arm happened to leave behind.
            set_seed(s)
            if arm == "abundance-only":
                m = AbundanceOnly(abund.shape[1], 32, n_classes)
                r = train_eval_mil(m, None, y, tr, va, te, n_classes, args.epochs,
                                   0.01, s, abundance=abund,
                                   device=args.device, class_weight=class_weight)
            else:
                m = MILClassifier(arm, in_dim, hidden[arm], n_classes,
                                  regions_per_batch=args.regions_per_batch)
                r = train_eval_mil(m, bags, y, tr, va, te, n_classes, args.epochs,
                                   0.01, s, device=args.device,
                                   class_weight=class_weight)
            scores.append(r)
        a = np.array(scores)                      # (n_runs, 2) = acc, macro-F1
        return a[:, 0], a[:, 1]

    maj = float((y == y.bincount().argmax()).float().mean())
    n_te = float(np.mean([len(te) for _, _, _, te in runs]))
    n_tr = float(np.mean([len(tr) for _, tr, _, _ in runs]))

    print(f"=== test scores over {args.seeds} seeds x {args.folds} folds "
          f"({len(runs)} runs/arm) ===")
    print(f"  {'majority baseline':<18} acc {maj:.3f}")
    ab, ab_f1 = eval_arm("abundance-only")
    print(f"  {'abundance-only':<18} acc {ab.mean():.3f} +- {ab.std():.3f}"
          f" | macroF1 {ab_f1.mean():.3f}")
    print(f"  {'':<18} {'':<4}(the bar every arm below must clear)\n")

    for arm in args.arms:
        sc, f1 = eval_arm(arm)
        beat = float((sc > ab).mean())
        mean_d, _t, p = corrected_t_test(sc - ab, n_te, n_tr)
        sig = "n/a" if np.isnan(p) else f"p={p:.3f}"
        print(f"  {arm:<18} acc {sc.mean():.3f} +- {sc.std():.3f}"
              f" | macroF1 {f1.mean():.3f}")
        print(f"  {'':<18} vs abundance {mean_d:+.3f} ({sig}, corrected) "
              f"| wins {beat:.0%} of paired runs")

    print("\nabundance-only is the bar: an arm shows spatial signal only if it")
    print("clears it. Majority baseline is the floor.")
    print("READING THE NUMBERS:")
    print("  - macroF1 near floor with respectable acc = collapsed to majority.")
    print("  - the +- is a spread, NOT a standard error: repeated-CV runs share")
    print("    training data. Use the corrected p, which accounts for that")
    print("    overlap (Nadeau & Bengio); a plain t-test here would overstate.")
    print("  - at this n, treat p as indicative and read it with the win rate.")


if __name__ == "__main__":
    main()