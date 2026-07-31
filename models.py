"""Everything learnable: hypergraph layers, the MIL classifier, capacity matching.

For a slide:

    region graph --[encoder]--> nodes --[readout]--> region vector
    {region vectors} --[attention pool]--> slide vector --[head]--> class

Everything downstream of the encoder is shared across arms, so the arm is the
only thing that varies.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.utils import scatter

# ---------------------------------------------------------------- layer

class DeepSetsHyperConv(nn.Module):
    """Set-aggregation hyperedge layer: node -> hyperedge -> node.

    rho(sum phi(x)), the Zaheer et al. universal form for permutation-invariant
    set functions. Pooling members is a SUM, preserving set cardinality that a
    mean divides out. log(set size) is fed to rho explicitly.

    `back` controls the RETURN path: "mean" averages over the hyperedges
    containing a node and discards node degree; "sum" keeps it. GCNConv retains
    degree via 1/sqrt(d_i d_j), so back="mean" costs the hypergraph arms a
    density signal the pairwise baseline has.
    """

    def __init__(self, in_dim, out_dim, hidden=None, back="mean"):
        super().__init__()
        hidden = hidden or out_dim
        self.back = back
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
        back = scatter(he[edge_idx], node_idx, dim=0, dim_size=n,
                       reduce=self.back)
        return self.out(torch.cat([back, x], dim=1))

class SumHyperConv(nn.Module):
    """Sum-pooling hyperedge layer with ONE linear transform.

    DeepSetsHyperConv conflates two claims: that sum preserves cardinality, and
    that a learned per-member encoding expresses richer set functions. Only the
    first is the thesis, and it costs no parameters -- so this isolates it at a
    cost comparable to GCNConv. Run against DeepSetsHyperConv on the same
    construction to separate the two.

    `back` as in DeepSetsHyperConv.
    """

    def __init__(self, in_dim, out_dim, hidden=None, back="mean"):
        super().__init__()
        # `hidden` unused; signature matches DeepSetsHyperConv
        self.back = back
        self.out = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x, hyperedge_index, num_hyperedges=None):
        node_idx, edge_idx = hyperedge_index[0], hyperedge_index[1]
        n = x.size(0)
        if num_hyperedges is None:
            num_hyperedges = int(edge_idx.max()) + 1 if edge_idx.numel() else 0
        if num_hyperedges == 0:                       # degenerate region
            return self.out(torch.cat([torch.zeros_like(x), x], dim=1))
        # unnormalised: a hyperedge of 12 yields twice the magnitude of one of 6
        he = scatter(x[node_idx], edge_idx, dim=0,
                     dim_size=num_hyperedges, reduce="sum")
        back = scatter(he[edge_idx], node_idx, dim=0, dim_size=n,
                       reduce=self.back)
        return self.out(torch.cat([back, x], dim=1))


def parse_arm(arm):
    """'hg-knn@sum' -> ('hg-knn', 'sum');  'hg-knn' -> ('hg-knn', 'deepsets').

    The construction half selects which cached graph to read; the aggregation
    half selects the layer. Encoding both in the arm name means deepsets and sum
    can run in the SAME sweep on the same folds, so the comparison is paired
    run-for-run rather than across separate jobs.
    """
    construction, _, agg = arm.partition("@")
    return construction, (agg or "deepsets")


# ------------------------------------------------------------ region encoders

# Fixed width every encoder emits, so AttentionMIL and the head are identical
# across arms and capacity matching only has to equalise the encoders. When
# out_dim was 2*hidden, widening an encoder widened the pool with it.
REGION_DIM = 64


def _readout(x, batch, n_regions):
    """Per-region mean+sum pooling. batch[i] = region that node i belongs to.

    Mean captures composition, sum captures scale/count -- keeping both means a
    region vector retains how MANY as well as what FRACTION, which matters for a
    task with an abundance axis.
    """
    mean = global_mean_pool(x, batch, size=n_regions)
    summ = global_add_pool(x, batch, size=n_regions)
    return torch.cat([mean, summ], dim=-1)          # (n_regions, 2*hidden)


class PairwiseRegionEncoder(nn.Module):
    """2-layer GCN over all a slide's regions at once -> (n_regions, REGION_DIM).

    `proj` is a plain linear adapter from the 2*hidden readout to the fixed
    region width -- no nonlinearity, so it changes the interface and not the
    representational story.
    """

    def __init__(self, in_dim, hidden, out_dim=REGION_DIM):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.proj = nn.Linear(2 * hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x, edge_index, batch, n_regions):
        x = F.relu(self.c1(x, edge_index))
        x = F.relu(self.c2(x, edge_index))
        return self.proj(_readout(x, batch, n_regions))


class HyperRegionEncoder(nn.Module):
    """2-layer hypergraph encoder over all a slide's regions at once.

    Emits REGION_DIM like the pairwise encoder, via the same linear adapter.

    agg="deepsets" -> DeepSetsHyperConv, the general rho(sum phi(x)) form
    agg="sum"      -> SumHyperConv, sum pooling with one transform
    Both preserve cardinality; only the first also learns what to aggregate.
    """

    # name -> (layer, return-path reduction). `2` means sum on the way back too,
    # so node degree survives; plain names keep the mean earlier results used.
    AGGS = {"deepsets":  (DeepSetsHyperConv, "mean"),
            "deepsets2": (DeepSetsHyperConv, "sum"),
            "sum":       (SumHyperConv, "mean"),
            "sum2":      (SumHyperConv, "sum")}

    def __init__(self, in_dim, hidden, out_dim=REGION_DIM, agg="deepsets"):
        super().__init__()
        if agg not in self.AGGS:
            raise ValueError(f"unknown agg {agg!r}; expected one of "
                             f"{sorted(self.AGGS)}")
        Conv, back = self.AGGS[agg]
        self.agg = agg
        self.c1 = Conv(in_dim, hidden, back=back)
        self.c2 = Conv(hidden, hidden, back=back)
        self.proj = nn.Linear(2 * hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x, hyperedge_index, batch, n_regions, num_hyperedges):
        x = F.relu(self.c1(x, hyperedge_index, num_hyperedges))
        x = F.relu(self.c2(x, hyperedge_index, num_hyperedges))
        return self.proj(_readout(x, batch, n_regions))


# ------------------------------------------------------------ packing a slide
#
# Tensors built here (batch vector, empty-hyperedge fallback) must be created on
# x's device -- they default to CPU, and a CPU index against CUDA features fails
# inside scatter, which reads like a PyG bug.

def pack_pairwise(region_graphs):
    """Combine a slide's pairwise regions into one disconnected graph.
    region_graphs: list of (x, edge_index). Returns x, edge_index, batch, R."""
    xs, eis, batch = [], [], []
    node_off = 0
    dev = region_graphs[0][0].device
    for r, (x, ei) in enumerate(region_graphs):
        xs.append(x)
        eis.append(ei + node_off)
        batch.append(torch.full((x.size(0),), r, dtype=torch.long, device=dev))
        node_off += x.size(0)
    return (torch.cat(xs, 0), torch.cat(eis, 1),
            torch.cat(batch), len(region_graphs))


def pack_hyper(region_graphs):
    """Combine a slide's hypergraph regions into one disconnected hypergraph.
    Node ids AND hyperedge ids are offset per region so nothing merges across
    regions. Returns x, hyperedge_index, batch, R, total_hyperedges."""
    xs, his, batch = [], [], []
    node_off, edge_off = 0, 0
    dev = region_graphs[0][0].device
    for r, (x, hi) in enumerate(region_graphs):
        xs.append(x)
        if hi.numel():
            h = hi.clone()
            h[0] += node_off
            h[1] += edge_off
            his.append(h)
            edge_off += int(hi[1].max()) + 1
        batch.append(torch.full((x.size(0),), r, dtype=torch.long, device=dev))
        node_off += x.size(0)
    hyperedge_index = (torch.cat(his, 1) if his
                       else torch.empty((2, 0), dtype=torch.long, device=dev))
    return (torch.cat(xs, 0), hyperedge_index,
            torch.cat(batch), len(region_graphs), edge_off)


def pack_bag(region_graphs, is_pw, regions_per_batch=16):
    """Pack one slide's regions into memory-bounded groups, once.

    Deterministic in the region graphs, so call it per slide before training and
    reuse the result rather than repeating the torch.cat on every pass. Region
    boundaries are never split, so grouping changes no region vector.
    """
    packer = pack_pairwise if is_pw else pack_hyper
    return [packer(region_graphs[i:i + regions_per_batch])
            for i in range(0, len(region_graphs), regions_per_batch)]


# ------------------------------------------------------------ MIL aggregation

class AttentionMIL(nn.Module):
    """Gated-attention pooling over a bag of region vectors (Ilse et al. 2018)."""

    def __init__(self, dim, att_dim=64):
        super().__init__()
        self.v = nn.Linear(dim, att_dim)
        self.u = nn.Linear(dim, att_dim)
        self.w = nn.Linear(att_dim, 1)

    def forward(self, bag):                       # bag: (R, dim)
        a = torch.tanh(self.v(bag)) * torch.sigmoid(self.u(bag))
        a = torch.softmax(self.w(a), dim=0)
        return (a * bag).sum(dim=0), a.squeeze(-1)


class MILClassifier(nn.Module):
    """Region encoder (minibatched) + attention pool + linear head. One per arm.

    Pool and head are sized from encoder.out_dim, which REGION_DIM pins, so the
    encoder is the only thing that differs in parameter count across arms.
    train_patterns.py asserts this at startup.

    Regions are encoded in groups of `regions_per_batch`: full-slide batching
    OOMs on large slides with high-cardinality hypergraphs. Whole regions per
    group and no detaching, so region vectors and gradients are unchanged.
    """

    def __init__(self, arm, in_dim, hidden, n_classes, pool="attention",
                 regions_per_batch=16):
        super().__init__()
        self.arm = arm
        construction, agg = parse_arm(arm)
        self.construction, self.agg = construction, agg
        self.is_pw = construction.startswith("pw-")
        self.encoder = (PairwiseRegionEncoder(in_dim, hidden) if self.is_pw
                        else HyperRegionEncoder(in_dim, hidden, agg=agg))
        self.pool_kind = pool
        self.rpb = regions_per_batch
        if pool == "attention":
            self.pool = AttentionMIL(self.encoder.out_dim)
        self.head = nn.Linear(self.encoder.out_dim, n_classes)

    def _encode_regions(self, packed):
        """Encode pre-packed groups to (R, REGION_DIM)."""
        vecs = []
        for g in packed:
            if self.is_pw:
                x, ei, batch, R = g
                vecs.append(self.encoder(x, ei, batch, R))
            else:
                x, hi, batch, R, n_he = g
                vecs.append(self.encoder(x, hi, batch, R, n_he))
        return torch.cat(vecs, dim=0)          # attached: gradients flow through all groups

    def forward(self, bag):
        """bag: raw [(x, struct), ...] regions, or pre-packed groups from
        pack_bag(). Prefer pre-packed -- packing here repeats identical work on
        every epoch."""
        packed = (pack_bag(bag, self.is_pw, self.rpb)
                  if bag and len(bag[0]) == 2 else bag)
        region_vecs = self._encode_regions(packed)
        if self.pool_kind == "attention":
            slide_vec, att = self.pool(region_vecs)
        else:
            slide_vec, att = region_vecs.mean(dim=0), None
        return self.head(slide_vec), att


class NodeClassifier(nn.Module):
    """2 conv layers + linear head, per-NODE output. No pooling, no MIL.

    For the masked-cell-type task: supervision is per cell, so one region gives
    thousands of labels, and nothing downstream of the encoder is involved. An
    arm that underperforms here is losing in its encoder; arms that tie here but
    differ on the slide task differ in the MIL stage.

    Same conv layers as the MIL encoders, so `arm` takes the same names
    including the @agg suffix.
    """

    def __init__(self, arm, in_dim, hidden, n_classes):
        super().__init__()
        construction, agg = parse_arm(arm)
        self.is_pw = construction.startswith("pw-")
        if self.is_pw:
            self.c1, self.c2 = GCNConv(in_dim, hidden), GCNConv(hidden, hidden)
        else:
            if agg not in HyperRegionEncoder.AGGS:
                raise ValueError(f"unknown agg {agg!r}; expected one of "
                                 f"{sorted(HyperRegionEncoder.AGGS)}")
            Conv, back = HyperRegionEncoder.AGGS[agg]
            self.c1 = Conv(in_dim, hidden, back=back)
            self.c2 = Conv(hidden, hidden, back=back)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, struct, num_hyperedges=None):
        if self.is_pw:
            x = F.relu(self.c1(x, struct))
            x = F.relu(self.c2(x, struct))
        else:
            x = F.relu(self.c1(x, struct, num_hyperedges))
            x = F.relu(self.c2(x, struct, num_hyperedges))
        return self.head(x)


class AbundanceOnly(nn.Module):
    """Triviality control: predict the label from per-slide cell-type fractions
    ONLY. No spatial structure. If this matches the graph arms, the task is
    abundance-driven and the spatial claim collapses."""

    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))

    def forward(self, slide_feature):
        return self.net(slide_feature)


# ---------------------------------------------------------------- capacity

def n_params(model):
    return sum(p.numel() for p in model.parameters())


def matched_hidden(cls, target, in_dim, out_dim, lo=4, hi=4096, step=2):
    """Smallest hidden dim whose parameter count is closest to `target`.

    Widens the pairwise baselines to match the Deep Sets arms, so a win cannot
    be attributed to parameter count. At equal hidden size the Deep Sets model
    had ~4.9x the GCN's parameters, and a pilot advantage vanished once matched.
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
