"""
models.py
---------
The GNN arms, the training loop, and capacity matching.

Kept separate from graph construction (graphs/) and from the task harness
(train_masked.py) because all three vary independently: you swap constructions
without touching models, and swap tasks without touching either.

Model choice follows from the arm name:
    pw-*  -> PairwiseGNN      (GCN, mean aggregation)
    hg-*  -> DeepSetsHyperGNN (sum aggregation) by default
             HyperGNN         (mean aggregation) when agg="mean" -- Stage 2 only
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, HypergraphConv
from torch_geometric.utils import scatter


# ---------------------------------------------------------------- models

class PairwiseGNN(nn.Module):
    """2-layer GCN. Mean-normalised aggregation. Used by all pw-* arms."""

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
    """PyG HypergraphConv: MEAN aggregation over hyperedge members.

    Near-equivalent to a GCN on the clique expansion (its Laplacian *is* a
    clique-expansion Laplacian), so this is the aggregation control, not the
    proposed method. Stage 2 only.
    """

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
    """Set-aggregation hyperedge layer: node -> hyperedge -> node.

    The difference from HypergraphConv is that pooling is a SUM, not a mean.
    Sum preserves set-level counts and composition that a clique expansion
    cannot reconstruct; mean divides that information back out and collapses
    toward clique-equivalent behaviour. That is the whole mechanism.

    Stages: phi (per-member MLP) -> sum-pool (+ log set size, so the model can
    account for cardinality rather than be destabilised by it) -> rho (MLP on
    the set summary) -> scatter back to members -> combine with own features.
    """

    def __init__(self, in_dim, out_dim, hidden=None):
        super().__init__()
        hidden = hidden or out_dim
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU())
        self.out = nn.Linear(hidden + in_dim, out_dim)

    def forward(self, x, hyperedge_index, num_hyperedges=None):
        node_idx, edge_idx = hyperedge_index[0], hyperedge_index[1]
        n = x.size(0)
        if num_hyperedges is None:
            num_hyperedges = int(edge_idx.max()) + 1 if edge_idx.numel() else 0
        if num_hyperedges == 0:                       # degenerate region
            return self.out(torch.cat([torch.zeros(n, self.rho[0].out_features,
                                                   device=x.device), x], dim=1))
        m = self.phi(x)
        he = scatter(m[node_idx], edge_idx, dim=0,
                     dim_size=num_hyperedges, reduce="sum")
        size = scatter(torch.ones_like(edge_idx, dtype=x.dtype), edge_idx,
                       dim=0, dim_size=num_hyperedges, reduce="sum").unsqueeze(1)
        he = self.rho(torch.cat([he, size.log1p()], dim=1))
        back = scatter(he[edge_idx], node_idx, dim=0, dim_size=n, reduce="mean")
        return self.out(torch.cat([back, x], dim=1))


class DeepSetsHyperGNN(nn.Module):
    """2 Deep Sets hyperedge layers. The proposed method (all hg-* Stage 1 arms)."""

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


# ---------------------------------------------------------------- capacity

def n_params(model):
    return sum(p.numel() for p in model.parameters())


def model_for(arm, in_dim, hidden, out_dim, agg="sum"):
    """Instantiate the right model class for an arm name."""
    if arm.startswith("pw-"):
        return PairwiseGNN(in_dim, hidden, out_dim)
    if agg == "mean":
        return HyperGNN(in_dim, hidden, out_dim)
    return DeepSetsHyperGNN(in_dim, hidden, out_dim)


def struct_of(arm, data):
    """The topology tensor an arm's model consumes."""
    return data.edge_index if arm.startswith("pw-") else data.hyperedge_index


def matched_hidden(cls, target, in_dim, out_dim, lo=4, hi=4096, step=2):
    """Smallest hidden dim whose parameter count is closest to `target`.

    Used to widen the pairwise baselines until they match the Deep Sets arms, so
    a win cannot be attributed to simply having more parameters. In a pilot the
    Deep Sets model had ~4.9x the parameters of the GCN at equal hidden size,
    and an apparent advantage vanished once that was equalised.
    """
    best, best_err = lo, float("inf")
    for h in range(lo, hi, step):
        p = n_params(cls(in_dim, h, out_dim))
        err = abs(p - target)
        if err < best_err:
            best, best_err = h, err
        if p > target * 1.2:
            break
    return best


# ---------------------------------------------------------------- training

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def macro_f1(pred, true, n_classes):
    """Macro-F1 by hand. Classes absent from `true` are skipped, not counted 0."""
    f1s = []
    for c in range(n_classes):
        if (true == c).sum() == 0:
            continue
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def make_targets(n, mask_frac, split, seed):
    """Pick the hidden cells and split them train/val/test. Shared across arms."""
    rng = np.random.default_rng(seed)
    targets = rng.permutation(n)[:int(n * mask_frac)]
    n_tr = int(len(targets) * split[0])
    n_va = int(len(targets) * split[1])
    return (torch.from_numpy(targets).long(),
            torch.from_numpy(targets[:n_tr]).long(),
            torch.from_numpy(targets[n_tr:n_tr + n_va]).long(),
            torch.from_numpy(targets[n_tr + n_va:]).long())


def train_eval(model, x, struct, y, tr, va, te, n_classes,
               epochs=150, lr=0.01, weight_decay=5e-4, seed=0):
    """Train one arm; return (val_acc, val_f1, test_acc, test_f1) at best val acc."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best = -1.0, (0.0, 0.0, 0.0, 0.0)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x, struct)[tr], y[tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, struct).argmax(1)
            va_acc = float((pred[va] == y[va]).float().mean())
            if va_acc > best_val:
                best_val = va_acc
                best = (va_acc, macro_f1(pred[va], y[va], n_classes),
                        float((pred[te] == y[te]).float().mean()),
                        macro_f1(pred[te], y[te], n_classes))
    return best