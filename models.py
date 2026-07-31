"""
models.py
---------
Everything learnable: the hypergraph layer, the MIL classifier built on it, and
capacity matching.

Kept separate from graph construction (graphs/) and from the task harness
(train_patterns.py) because all three vary independently -- you swap
constructions without touching models, and swap tasks without touching either.

The chain, for a slide:

    region graph --[encoder]--> nodes --[readout]--> region vector
    {region vectors} --[attention pool]--> slide vector --[head]--> class

DeepSetsHyperConv is the mechanism the project turns on: its SUM pooling is what
a clique expansion cannot reproduce. Both region encoders are built on it or on
GCNConv, and everything downstream of the encoder is shared across arms, so the
arm is the only thing that varies.

REMOVED (recoverable via `git show ed1da7a:models.py`): the node-level
classifiers PairwiseGNN / HyperGNN / DeepSetsHyperGNN and helpers model_for /
struct_of / make_targets / train_eval, whose only caller was the masked-cell-type
task. Note HyperGNN was the MEAN-aggregation control -- the comparison that
distinguishes "hypergraph topology adds nothing" from "sum-vs-mean is the wrong
lever" -- and no equivalent exists in the MIL pipeline. If a null needs
explaining, that is the thing to bring back.
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

class SumHyperConv(nn.Module):
    """Minimal sum-pooling hyperedge layer. ONE linear transform.

    This exists because DeepSetsHyperConv conflates two separate claims:

      (a) SUM preserves set-level cardinality that MEAN divides back out.
      (b) A learned per-member encoding phi, pooled and read by rho, can express
          richer set functions than any fixed aggregation.

    The project's thesis is (a). But (a) needs only an unnormalised sum, which
    costs ZERO parameters -- note that GCNConv is already a sum, just with
    degree normalisation dividing the count out again. So the minimal edit from
    the pairwise baseline to the hypothesis is REMOVING a normalisation, not
    adding two MLPs.

    (b) is a much more ambitious claim, and rho(sum phi(x)) -- the Zaheer et al.
    universal form -- is three stacked transforms. On ~68 training slides with
    ~30 full-batch gradient steps that is not affordable, and its cost gets
    charged to (a) in every comparison.

    This layer isolates (a): sum pooling, one transform, parameter count in the
    same range as GCNConv. Run it against DeepSetsHyperConv on the SAME
    construction to separate "sum aggregation helps" from "learned set
    aggregation is unaffordable at this n" -- currently confounded everywhere.
    """

    def __init__(self, in_dim, out_dim, hidden=None):
        super().__init__()
        # signature matches DeepSetsHyperConv so the encoders are interchangeable;
        # `hidden` is unused -- there is no intermediate representation
        self.out = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x, hyperedge_index, num_hyperedges=None):
        node_idx, edge_idx = hyperedge_index[0], hyperedge_index[1]
        n = x.size(0)
        if num_hyperedges is None:
            num_hyperedges = int(edge_idx.max()) + 1 if edge_idx.numel() else 0
        if num_hyperedges == 0:                       # degenerate region
            return self.out(torch.cat([torch.zeros_like(x), x], dim=1))
        # SUM over members, unnormalised -- this is the whole mechanism. A
        # hyperedge of 12 cells yields twice the magnitude of one with 6, which
        # is exactly the information mean pooling and clique expansion destroy.
        he = scatter(x[node_idx], edge_idx, dim=0,
                     dim_size=num_hyperedges, reduce="sum")
        # MEAN back to members, so magnitude does not also scale with how many
        # hyperedges a node happens to belong to (a node-degree effect, not a
        # set-size one). Keeps the cardinality signal, drops the degree one.
        back = scatter(he[edge_idx], node_idx, dim=0, dim_size=n, reduce="mean")
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

# Width of the region vector every arm emits. Both encoders project their
# readout down to this before the MIL stage, so AttentionMIL and the head are
# byte-for-byte the same size regardless of arm or hidden dim.
#
# This exists to make a capacity confound STRUCTURALLY IMPOSSIBLE rather than
# tuned away. Previously out_dim was 2*hidden, so widening the pairwise encoder
# to match the hypergraph encoder's parameter count also widened the pool and
# head -- matching the encoders was what CREATED the downstream mismatch (pool
# 20,161 vs 8,385, head 628 vs 260, total 1.77x). Retuning `hidden` to equalise
# totals would fix the number but not the mechanism: any later change to
# att_dim, n_classes, or the readout would silently reintroduce the drift.
# Pinning the interface means the only thing capacity matching has to equalise
# is the encoders themselves.
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

    AGGS = {"deepsets": DeepSetsHyperConv, "sum": SumHyperConv}

    def __init__(self, in_dim, hidden, out_dim=REGION_DIM, agg="deepsets"):
        super().__init__()
        if agg not in self.AGGS:
            raise ValueError(f"unknown agg {agg!r}; expected one of "
                             f"{sorted(self.AGGS)}")
        Conv = self.AGGS[agg]
        self.agg = agg
        self.c1 = Conv(in_dim, hidden)
        self.c2 = Conv(hidden, hidden)
        self.proj = nn.Linear(2 * hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x, hyperedge_index, batch, n_regions, num_hyperedges):
        x = F.relu(self.c1(x, hyperedge_index, num_hyperedges))
        x = F.relu(self.c2(x, hyperedge_index, num_hyperedges))
        return self.proj(_readout(x, batch, n_regions))


# ------------------------------------------------------------ packing a slide
#
# DEVICE: every tensor created here must be built on the same device as the
# incoming features. The `batch` vector and the empty-hyperedge fallback are
# constructed from scratch rather than derived from x, so they default to CPU
# unless told otherwise -- and a CPU index against CUDA features fails inside
# scatter, not here, which makes it read like a PyG bug. CPU-only runs never
# expose it because everything agrees by accident.

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
    """Pack one slide's regions into memory-bounded groups, ONCE.

    Call this per slide before training and reuse the result for every epoch and
    every run. Packing is a deterministic function of the region graphs, so doing
    it inside forward() re-runs the same torch.cat over many small tensors on
    every train, val and test pass -- tens of thousands of times across a sweep,
    for an identical answer.

    Region boundaries are never split, so grouping does not change any region
    vector; this is purely about not repeating work.
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

    The pool and head are sized from encoder.out_dim, which REGION_DIM pins to a
    constant, so they are identical across arms and the ONLY thing that differs
    in parameter count is the encoder -- exactly what capacity matching targets.
    train_patterns.py asserts this at startup so it cannot regress unnoticed.

    Regions are encoded in GROUPS of `regions_per_batch` rather than all at once.
    Full-slide batching put every region's nodes on the GPU simultaneously, which
    OOMs on large slides with high-cardinality hypergraph constructions. Grouping
    caps peak memory at group-size while keeping most of the batching speedup.

    Region boundaries are never split (whole regions per group), so per-region
    vectors are identical to full-batch or per-region encoding. The group vectors
    are concatenated WITHOUT detaching, so gradients flow through every group and
    training is unchanged.
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
        """bag: either raw [(x, struct), ...] regions, or pre-packed groups from
        pack_bag(). Prefer pre-packed -- packing is a deterministic function of
        the region graphs, so doing it here repeats identical torch.cat work on
        every epoch for train, val AND test. Hoisting it out of the loop is a
        pure speedup with no effect on the result.
        """
        packed = (pack_bag(bag, self.is_pw, self.rpb)
                  if bag and len(bag[0]) == 2 else bag)
        region_vecs = self._encode_regions(packed)
        if self.pool_kind == "attention":
            slide_vec, att = self.pool(region_vecs)
        else:
            slide_vec, att = region_vecs.mean(dim=0), None
        return self.head(slide_vec), att


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
