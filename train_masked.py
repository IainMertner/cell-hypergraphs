"""DIAGNOSTIC: masked cell-type prediction, per node.

30% of a region's cells have their features zeroed; the model predicts their
PanNuke type from the surrounding cells and the topology. Supervision is per
cell and nothing downstream of the encoder runs, so this isolates the encoders
from both the sample size and the MIL stage.

--features decides whether the task is circular:

  type   5-d one-hot type only. The label is in the input for every unmasked
         cell, so the task reduces to reading a neighbourhood type histogram --
         which flatters sum aggregation trivially.
  morph  5-d morphology only (area, perimeter, circularity, eccentricity,
         extent). The label is nowhere in the input. THE DEFAULT.
  both   all 10 dims. As circular as `type`.
  none   constant features; see select_features.

The topology is never a leak -- every construction is built from centroids.

Reads the graph cache, so no re-precompute.

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
    """Slice cached node features. The label is read from the one-hot block
    before this, so `morph` and `none` genuinely remove it from the input.

    `none` sets every feature to a constant 1, leaving topology as the only
    signal: sum pooling then reduces to sum_{j in e} 1 == |e|, the hyperedge
    cardinality. So hg-knn should sit at the floor (its cardinality is fixed at
    k+1) while hg-radius (6-15) still encodes local density -- which tests the
    fixed-cardinality argument directly.
    """
    if features == "none":
        return torch.ones(x.shape[0], 1, dtype=x.dtype)
    if features == "type":
        return x[:, :N_TYPES]
    if features == "morph":
        return x[:, N_TYPES:]
    return x


def load_regions(graph_cache, arms, max_regions, min_cells, features, seed=0):
    """Pull the same regions for every arm, so region i is the same cells under
    different topology and the comparison is paired."""
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
            # label first, then slice, so --features morph really drops it
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
            # macro-F1 not accuracy: cell types are heavily imbalanced
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
                    help="morph keeps the label out of the input; type/both "
                         "put it in and make the task circular")
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
    # Both baselines, over every region. With inverse-frequency weights the
    # model spreads predictions across classes, raising macro-F1 and lowering
    # accuracy -- so an arm can beat the floor and still lose to "always answer
    # the dominant class".
    floors, majs, presents = [], [], []
    for ri, (x, structs, y) in enumerate(regions):
        n = x.shape[0]
        counts = np.bincount(y.numpy(), minlength=N_TYPES)
        maj = counts.max() / n
        # macro-F1 of a single-class predictor, over classes present
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
            # features=none masks nothing, so every node is usable and the test
            # set is 20% rather than 6%. The other modes must mask: the node's
            # own features carry the label (type/both) or a strong cue (morph).
            frac = 1.0 if args.features == "none" else args.mask_frac
            targets = rng.permutation(n)[:int(n * frac)]
            n_tr, n_va = int(len(targets) * 0.6), int(len(targets) * 0.2)
            tr = torch.from_numpy(targets[:n_tr]).long()
            va = torch.from_numpy(targets[n_tr:n_tr + n_va]).long()
            te = torch.from_numpy(targets[n_tr + n_va:]).long()
            xm = x.clone()
            if args.features != "none":
                xm[torch.from_numpy(targets).long()] = 0.0  # hide the targets
            # zeroing under features=none would actively harm: sum-pooling
            # would read the count of UNMASKED members, Binomial(|e|, 0.7), so
            # even constant-cardinality hg-knn would pick up noise

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
    print("\nPer-cell supervision and no MIL stage, so a difference between arms")
    print("is the ENCODER. A tie is weak evidence -- cell type is partly readable")
    print("off the neighbours, so both arms may be at ceiling.")


if __name__ == "__main__":
    main()
