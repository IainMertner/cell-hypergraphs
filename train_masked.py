"""
train_masked.py
---------------
DIAGNOSTIC: masked cell-type prediction, per NODE.

30% of cells in a region have their features zeroed; the model predicts their
PanNuke type from the surrounding cells and the topology.

WHY THIS IS THE USEFUL DIAGNOSTIC RIGHT NOW. Every slide-level result is limited
by 122 labels and swamped by variance, and it cannot separate three explanations:
the encoder, the MIL pooling, or the sample size. This task removes two of them:

  - supervision is PER CELL. One region gives thousands of labels, so sample
    size is not the constraint.
  - nothing downstream of the encoder runs. No bag aggregation, no attention
    pooling. Whatever differs between arms here is the encoder.

So: if hg-* beats or matches pw-knn here but loses on the slide task, the
problem is the MIL stage. If it loses HERE too, the hypergraph representation
genuinely carries less usable signal than the pairwise one at this scale.

--features CONTROLS WHETHER THE TASK IS CIRCULAR, and this is the whole design.

  type   node features are the 5-d one-hot type ONLY. The label is then in the
         input for every unmasked cell, so predicting a masked cell's type is
         essentially reading a neighbourhood type histogram. That measures
         whether the graph propagates local composition -- and it flatters SUM
         aggregation for a trivial reason, since counting neighbour types is
         precisely what sum does. Informative about propagation, near-useless
         as evidence about representing tissue.

  morph  node features are the 5-d morphology ONLY (area, perimeter,
         circularity, eccentricity, extent). The label appears NOWHERE in the
         input, so the model must infer "this is a lymphocyte" from the SHAPES
         of surrounding cells. Non-circular, genuinely hard, and the version to
         believe. THIS IS THE DEFAULT.

  both   all 10 dims. Same circularity as `type`, plus morphology.

Note the topology is never a leak: pw-knn, hg-knn and hg-radius are built from
centroids alone and never see cell type.

CAVEAT. Under `type`/`both`, cell type is partly readable straight off the
neighbours -- tissue is locally homogeneous -- so both arms may hit a ceiling
and fail to discriminate. A tie there is weak evidence; a clear difference is
strong. Read the margin over the majority-class floor, not the raw score.

Reads the graph cache, so no re-precompute -- the node features already carry
the type as their leading one-hot block, which is both the input to mask and
the label to predict.

Usage (SUBMIT, do not run on a login node):
    qsub scripts/run_masked.sh
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from graphs import N_TYPES, DEFAULT_ARMS
from models import set_seed, matched_hidden, n_params, macro_f1, NodeClassifier, parse_arm


def select_features(x, features):
    """Slice cached node features. The LABEL is always taken from the one-hot
    block BEFORE this, so `morph` and `none` genuinely remove it from the input.

    `none` replaces every node feature with a constant 1, leaving TOPOLOGY as
    the only signal. That makes it the sharpest test of sum-vs-mean available:

      SumHyperConv      sum_{j in e} 1  ==  |e|, the hyperedge CARDINALITY
      DeepSetsHyperConv same, and it already feeds log1p(size) explicitly
      GCNConv           sum_j 1/sqrt(d_i d_j), a degree term

    So with constant features the sum layers reduce to reading set size, and a
    concrete prediction follows: hg-knn should sit AT THE FLOOR, because its
    cardinality is fixed at k+1 and therefore constant and uninformative, while
    hg-radius (median 6, max 15) still encodes local density. If that holds it
    confirms the fixed-cardinality argument rather than leaving it inferred.
    """
    if features == "none":
        return torch.ones(x.shape[0], 1, dtype=x.dtype)
    if features == "type":
        return x[:, :N_TYPES]
    if features == "morph":
        return x[:, N_TYPES:]
    return x


def load_regions(graph_cache, arms, max_regions, min_cells, features, seed=0):
    """Pull the same regions for every arm, so the comparison is paired.

    Region i of arm A and region i of arm B are the same cells with different
    topology -- that is the whole point, and it only holds if the selection is
    identical across arms.
    """
    files = [f for f in sorted(glob.glob(os.path.join(graph_cache, "*.pt")))
             if not f.endswith("_params.pt")]
    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    picked = []                       # [(x, {arm: (struct, n_he)}, y), ...]
    for f in files:
        bags = torch.load(f)["bags"]
        if any(parse_arm(a)[0] not in bags for a in arms):
            continue
        ref = parse_arm(arms[0])[0]
        for r in range(len(bags[ref])):
            x = bags[ref][r][0]
            if x.shape[0] < min_cells:
                continue
            # label from the one-hot block FIRST, then slice the inputs -- so
            # --features morph removes the label from the input entirely
            y = x[:, :N_TYPES].argmax(1)
            x = select_features(x, features)
            structs = {}
            for a in arms:
                c = parse_arm(a)[0]
                s = bags[c][r][1]
                n_he = (int(s[1].max()) + 1 if (not c.startswith("pw-")
                                                and s.numel()) else None)
                structs[a] = (s, n_he)
            picked.append((x, structs, y))
            if len(picked) >= max_regions:
                return picked
    return picked


def train_eval_region(model, x, struct, n_he, y, tr, va, te, is_pw,
                      epochs, lr, patience, device, class_weight):
    """Train on masked nodes `tr`, early-stop on `va`, score `te`."""
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    struct = struct.to(device)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    def fwd():
        return model(x, struct) if is_pw else model(x, struct, n_he)

    best_val, best_state, since = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = fwd()
        F.cross_entropy(out[tr], y[tr], weight=class_weight).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = fwd().argmax(1)
            # select on macro-F1, not accuracy: cell types are heavily
            # imbalanced (neoplastic dominates), so accuracy rewards collapse
            f1 = macro_f1(pred[va].cpu().numpy(), y[va].cpu().numpy(), N_TYPES)
            if f1 > best_val:
                best_val, since = f1, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                since += 1
                if since >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = fwd().argmax(1)
    p, t = pred[te].cpu().numpy(), y[te].cpu().numpy()
    return float((p == t).mean()), macro_f1(p, t, N_TYPES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-cache", default="graph_cache")
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
    ap.add_argument("--regions", type=int, default=10)
    ap.add_argument("--min-cells", type=int, default=2000)
    ap.add_argument("--mask-frac", type=float, default=0.30)
    ap.add_argument("--features", choices=["none", "type", "morph", "both"],
                    default="morph",
                    help="morph (default) keeps the label OUT of the input, so "
                         "the task is not circular. type/both put the one-hot "
                         "label in the features and reduce the task to reading "
                         "a neighbourhood type histogram")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"masked cell-type prediction | arms={args.arms} "
          f"| {args.regions} regions x {args.seeds} seeds")
    if args.features == "none":
        print("features=none  <- constant node features: TOPOLOGY ONLY.")
        print("  nothing is masked (there is nothing to hide), so ALL nodes are")
        print("  used, split 60/20/20 -- not the 30% subset the other modes use.")
        print("  sum aggregation reduces to reading hyperedge cardinality here,")
        print("  so hg-knn (fixed at k+1) should sit at the FLOOR while")
        print("  hg-radius (varies 6-15) still has something to read.")
    elif args.features == "morph":
        print("features=morph  <- label is NOT in the input; non-circular")
    else:
        print(f"features={args.features}  <- WARNING: the one-hot label is IN "
              "the input.\n  This reduces the task to reading a neighbourhood "
              "type histogram, and favours\n  sum aggregation trivially. Use "
              "--features morph for evidence you can rely on.")
    regions = load_regions(args.graph_cache, args.arms, args.regions,
                           args.min_cells, args.features)
    if not regions:
        raise SystemExit("no regions matched -- check --graph-cache and --arms")
    in_dim = regions[0][0].shape[1]
    print(f"{len(regions)} regions | node features {in_dim}-d "
          f"| sizes {[int(r[0].shape[0]) for r in regions]}\n")

    # capacity match every arm to the deepsets hypergraph, as train_patterns does
    def _node(a):
        return lambda i, h, o: NodeClassifier(a, i, h, o)
    ref_arm = next((a for a in args.arms if not a.startswith("pw-")), args.arms[0])
    target = n_params(_node(ref_arm)(in_dim, args.hidden, N_TYPES))
    hidden = {a: (args.hidden if a == ref_arm
                  else matched_hidden(_node(a), target, in_dim, N_TYPES))
              for a in args.arms}
    print(f"capacity target {target:,} params ({ref_arm} @ hidden={args.hidden})")
    for a in args.arms:
        print(f"  {a:<16} hidden={hidden[a]:<4} "
              f"{n_params(_node(a)(in_dim, hidden[a], N_TYPES)):>7,} params")

    scores = {a: [] for a in args.arms}
    # Track BOTH baselines across every region, not just the first. Majority
    # accuracy matters as much as the macro-F1 floor: with inverse-frequency
    # weights the model spreads predictions across classes, which RAISES
    # macro-F1 and LOWERS accuracy -- so an arm can look respectable on accuracy
    # while being worse than always answering the dominant class.
    floors, majs, presents = [], [], []
    for ri, (x, structs, y) in enumerate(regions):
        n = x.shape[0]
        counts = np.bincount(y.numpy(), minlength=N_TYPES)
        maj = counts.max() / n
        # macro-F1 a single-class predictor would score, over classes PRESENT
        present = int((counts > 0).sum())
        f_floor = (2 * maj / (maj + 1)) / present
        floors.append(f_floor)
        majs.append(maj)
        presents.append(present)
        # inverse-frequency weights over present classes
        w = torch.tensor([n / (present * c) if c else 0.0 for c in counts],
                         dtype=torch.float)

        for s in range(args.seeds):
            set_seed(s)
            rng = np.random.default_rng(s)
            # features=none masks nothing, so restricting to a 30% subset would
            # just discard 70% of the labels for no reason -- every node is
            # equally "predict this from structure". Use them all: the test set
            # goes from 6% of nodes (20% of a 30% subset) to 20%.
            #
            # The other modes still need masking. Under type/both the label is
            # literally in the node's own features; under morph its own shape is
            # a strong cue, and hiding it is what forces the prediction to come
            # from CONTEXT rather than from the cell itself.
            frac = 1.0 if args.features == "none" else args.mask_frac
            targets = rng.permutation(n)[:int(n * frac)]
            n_tr, n_va = int(len(targets) * 0.6), int(len(targets) * 0.2)
            tr = torch.from_numpy(targets[:n_tr]).long()
            va = torch.from_numpy(targets[n_tr:n_tr + n_va]).long()
            te = torch.from_numpy(targets[n_tr + n_va:]).long()
            xm = x.clone()
            if args.features != "none":
                xm[torch.from_numpy(targets).long()] = 0.0  # hide the targets
            # features=none: nothing to hide, and zeroing WOULD do harm. With
            # 0/1 inputs, sum-pooling reads the count of UNMASKED members --
            # Binomial(|e|, 0.7) -- so cardinality arrives with ~27% noise, and
            # hg-knn (constant |e|=6) picks up spurious variation that is pure
            # noise. Left at 1, `he` is exactly |e|, so "hg-knn has no
            # cardinality signal" is a clean prediction rather than a muddy one.

            for a in args.arms:
                set_seed(s)
                struct, n_he = structs[a]
                m = NodeClassifier(a, in_dim, hidden[a], N_TYPES)
                t0 = time.time()
                acc, f1 = train_eval_region(
                    m, xm, struct, n_he, y, tr, va, te,
                    a.startswith("pw-"), args.epochs, 0.01, args.patience,
                    args.device, w)
                scores[a].append((acc, f1))
                print(f"  region {ri} seed {s} {a:<16} acc={acc:.3f} "
                      f"f1={f1:.3f} | {time.time() - t0:.0f}s", flush=True)

    print(f"\n=== masked cell-type prediction, {len(regions)} regions "
          f"x {args.seeds} seeds ===")
    maj_acc, floor = float(np.mean(majs)), float(np.mean(floors))
    print(f"  BASELINES (mean over regions, {min(presents)}-{max(presents)} "
          f"classes present)")
    print(f"    majority accuracy {maj_acc:.3f}   <- an arm below this is worse "
          f"than always")
    print(f"                                         answering the dominant class")
    print(f"    macroF1 floor     {floor:.3f}   <- what that same predictor scores")
    for a in args.arms:
        v = np.array(scores[a])
        acc, f1 = v[:, 0].mean(), v[:, 1].mean()
        print(f"  {a:<20} acc {acc:.3f} +- {v[:, 0].std():.3f} "
              f"| macroF1 {f1:.3f} +- {v[:, 1].std():.3f}"
              + ("  [acc BELOW majority]" if acc < maj_acc else ""))
    print("\nPer-CELL supervision, so thousands of labels per region -- sample")
    print("size is not the constraint here, and no MIL or attention pooling is")
    print("involved. A difference between arms is the ENCODER. A tie is weak")
    print("evidence: cell type is partly readable from neighbours, so both arms")
    print("may simply be at ceiling.")


if __name__ == "__main__":
    main()
