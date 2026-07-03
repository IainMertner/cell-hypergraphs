"""
train_masked.py
---------------
Smoke-test task: masked cell-type prediction on ONE region.

Goal is to stand up an identical training loop on both arms (pairwise graph vs
neighbourhood hypergraph) and get a first comparable number. It is deliberately
the simplest valid task, NOT a headline result.

The task (and the leakage guard that makes it valid):
  * Node features ARE the one-hot cell type, so predicting a cell's type from an
    input containing that type is trivial. We therefore HIDE 30% of cells by
    zeroing their input features, and predict their type from context only
    (the visible 70% of cells + the graph topology).
  * Hidden cells are split train/val/test. Loss on train, report val/test.
  * Everything except the message-passing topology is held identical across the
    two arms (same region, same hidden set, same split, same seed, same
    hyper-params), so any difference is attributable to pairs-vs-sets.

Runs on CPU — 10k nodes trains in seconds and avoids any GPU issues.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, HypergraphConv
from torch_geometric.utils import scatter

from build_graph import (
    load_cells, densest_region, microns_to_px,
    build_knn_graph, build_neighbourhood_hypergraph,
    build_delaunay_graph, clique_expand, N_TYPES,
)
from torch_geometric.data import Data

# ---- config ----
CELLS_JSON = r"\\wsl$\Ubuntu\home\iain\cellvit\test_out\TCGA-E2-A14P\cells.json"
TILE_PX = 4000
K_VALUES = [5]     # k-sweep for k-NN / hypergraph arms (Delaunay is k-free)
FEATURES = "morph"         # "type" | "morph" | "both" -- node feature set to test
RADIUS_UM = 35.0
HIDDEN = 32
EPOCHS = 150
LR = 0.01
WEIGHT_DECAY = 5e-4
MASK_FRAC = 0.30          # fraction of cells hidden and predicted
SPLIT = (0.6, 0.2, 0.2)  # train/val/test split *within* the hidden set
SEED = 0


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def make_targets(n, mask_frac, split, seed):
    """Choose hidden target cells and split them into train/val/test index tensors."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_target = int(n * mask_frac)
    targets = perm[:n_target]
    n_tr = int(n_target * split[0])
    n_va = int(n_target * split[1])
    tr = torch.from_numpy(targets[:n_tr]).long()
    va = torch.from_numpy(targets[n_tr:n_tr + n_va]).long()
    te = torch.from_numpy(targets[n_tr + n_va:]).long()
    all_targets = torch.from_numpy(targets).long()
    return all_targets, tr, va, te


class PairwiseGNN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.c2(x, edge_index))
        return self.head(x)


class HyperGNN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.c1 = HypergraphConv(in_dim, hidden)
        self.c2 = HypergraphConv(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, hyperedge_index):
        x = F.relu(self.c1(x, hyperedge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.c2(x, hyperedge_index))
        return self.head(x)


class DeepSetsHyperConv(nn.Module):
    """Set-aggregation hyperedge layer (node -> hyperedge -> node).

    The point of difference from vanilla HypergraphConv: hyperedge pooling is a
    SUM (stage 3), not a mean. Sum preserves set-level counts/composition that a
    clique expansion cannot reconstruct -- this is the mechanism the whole
    hypergraph hypothesis rests on. Mean would collapse back toward
    clique-equivalent behaviour.

    Stages:
      1. phi: per-member MLP on node features (learns what to aggregate)
      3. sum-pool members -> hyperedge representation (+ explicit set size, so
         the model can account for size rather than be destabilised by it)
      4. rho: MLP on the set summary
      5. scatter hyperedge reps back to member nodes (mean over the hyperedges a
         node belongs to), then combine with the node's own input feature
    """

    def __init__(self, in_dim, out_dim, hidden=None):
        super().__init__()
        hidden = hidden or out_dim
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())      # stage 1
        self.rho = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU())  # stage 4 (+1 = set size)
        self.out = nn.Linear(hidden + in_dim, out_dim)                      # stage 5 combine

    def forward(self, x, hyperedge_index, num_hyperedges=None):
        node_idx, edge_idx = hyperedge_index[0], hyperedge_index[1]
        n = x.size(0)
        if num_hyperedges is None:
            num_hyperedges = int(edge_idx.max()) + 1

        m = self.phi(x)                                   # (N, hidden)  stage 1
        msg = m[node_idx]                                 # (M, hidden)  gather members
        he = scatter(msg, edge_idx, dim=0, dim_size=num_hyperedges, reduce="sum")  # stage 3: SUM
        size = scatter(torch.ones_like(edge_idx, dtype=x.dtype), edge_idx,
                       dim=0, dim_size=num_hyperedges, reduce="sum").unsqueeze(1)
        he = self.rho(torch.cat([he, size.log1p()], dim=1))                # stage 4
        back = he[edge_idx]                               # (M, hidden)
        node_msg = scatter(back, node_idx, dim=0, dim_size=n, reduce="mean")  # stage 5: back to nodes
        return self.out(torch.cat([node_msg, x], dim=1))


class DeepSetsHyperGNN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.c1 = DeepSetsHyperConv(in_dim, hidden)
        self.c2 = DeepSetsHyperConv(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, hyperedge_index):
        x = F.relu(self.c1(x, hyperedge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.c2(x, hyperedge_index))
        return self.head(x)


def macro_f1(pred, true, n_classes):
    """Macro-F1 by hand (no sklearn dep). Averages F1 over classes present in `true`."""
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        if (true == c).sum() == 0:
            continue  # class absent in this region -> skip
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def run_arm(name, model, x_input, struct, y, tr, va, te):
    """Train one arm and return (val_acc, val_f1, test_acc, test_f1)."""
    set_seed(SEED)  # identical init/dropout randomness across arms
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val, best = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(x_input, struct)
        loss = F.cross_entropy(logits[tr], y[tr])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(x_input, struct).argmax(1)
            va_acc = float((pred[va] == y[va]).float().mean())
            if va_acc > best_val:
                va_f1 = macro_f1(pred[va], y[va], N_TYPES)
                te_acc = float((pred[te] == y[te]).float().mean())
                te_f1 = macro_f1(pred[te], y[te], N_TYPES)
                best_val, best = va_acc, (va_acc, va_f1, te_acc, te_f1)
    print(f"[{name:>10}] val acc {best[0]:.3f} | val F1 {best[1]:.3f} | "
          f"test acc {best[2]:.3f} | test F1 {best[3]:.3f}")
    return best


def _feats(types, morph, n):
    """Assemble node features per FEATURES setting. Returns (x, in_dim)."""
    onehot = torch.zeros((n, N_TYPES), dtype=torch.float)
    onehot[torch.arange(n), torch.from_numpy(types - 1).long()] = 1.0
    if FEATURES == "type":
        return onehot, N_TYPES
    m = torch.from_numpy(morph).float()
    if FEATURES == "morph":
        return m, m.shape[1]
    return torch.cat([onehot, m], dim=1), N_TYPES + m.shape[1]


def main():
    set_seed(SEED)
    centroids, types, mpp, morph_all = load_cells(CELLS_JSON, with_morphology=True)
    radius_px = microns_to_px(RADIUS_UM, mpp)

    # densest region: derive ONE mask and apply to centroids, types, morph together
    _, _, (x0, y0) = densest_region(centroids, types, TILE_PX)
    mask = ((centroids[:, 0] >= x0) & (centroids[:, 0] < x0 + TILE_PX) &
            (centroids[:, 1] >= y0) & (centroids[:, 1] < y0 + TILE_PX))
    sub_c, sub_t, sub_m = centroids[mask], types[mask], morph_all[mask]
    n = len(sub_c)
    print(f"region: {n:,} cells | cap {RADIUS_UM}um | k-sweep {K_VALUES} | features={FEATURES}\n")

    y = (torch.from_numpy(sub_t).long() - 1)
    x_full, in_dim = _feats(sub_t, sub_m, n)

    all_targets, tr, va, te = make_targets(n, MASK_FRAC, SPLIT, SEED)
    x_input = x_full.clone()
    x_input[all_targets] = 0.0            # hide target cells (all features)
    print(f"hidden {len(all_targets):,} cells "
          f"(train {len(tr):,} / val {len(va):,} / test {len(te):,})")

    d = build_delaunay_graph(sub_c, sub_t, radius_px)
    print(f"\n=== pw-delaunay ===")
    run_arm("pw-delaunay", PairwiseGNN(in_dim, HIDDEN, N_TYPES),
            x_input, d.edge_index, y, tr, va, te)

    for k in K_VALUES:
        print(f"\n=== k = {k} ===")
        g = build_knn_graph(sub_c, sub_t, k, radius_px)
        h = build_neighbourhood_hypergraph(sub_c, sub_t, k, radius_px)
        pwc_ei = clique_expand(h.hyperedge_index)
        run_arm("pw-knn", PairwiseGNN(in_dim, HIDDEN, N_TYPES),
                x_input, g.edge_index, y, tr, va, te)
        run_arm("pw-clique", PairwiseGNN(in_dim, HIDDEN, N_TYPES),
                x_input, pwc_ei, y, tr, va, te)
        run_arm("hg-clique", HyperGNN(in_dim, HIDDEN, N_TYPES),
                x_input, h.hyperedge_index, y, tr, va, te)
        run_arm("hg-deepsets", DeepSetsHyperGNN(in_dim, HIDDEN, N_TYPES),
                x_input, h.hyperedge_index, y, tr, va, te)


if __name__ == "__main__":
    main()