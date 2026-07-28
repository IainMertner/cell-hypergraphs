"""
mil.py
------
Multiple-instance learning for SLIDE-LEVEL labels.

Each slide is a BAG of region graphs; the model encodes each region to a vector,
pools regions to a slide vector, and classifies:

    region graph --[GNN]--> nodes --[readout]--> region vector
    {region vectors} --[attention pool]--> slide vector --[head]--> class

The GNN inside a region is the arm (pairwise GCN vs Deep Sets hypergraph); the
MIL pool over regions is shared across arms, so it is not a confound.

BATCHING: a slide's regions are packed into one disconnected graph (features
concatenated, edge/hyperedge ids offset so regions never connect) and encoded in
a single forward pass, with a `batch` vector recording each node's region. Since
no edges cross between regions, message passing cannot either, so the result is
identical to encoding regions one-by-one -- it just avoids paying Python/dispatch
overhead ~50 times per slide. The readout pools PER REGION via the batch vector.

Also here: AbundanceOnly, the triviality control -- predicts the label from
per-slide cell-type fractions alone, no spatial structure. If it matches the
graph arms, the task is abundance-driven and no spatial claim holds.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool

from models import DeepSetsHyperConv          # reuse the Deep Sets layer


# ------------------------------------------------------------ region encoders

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
    """2-layer GCN over all a slide's regions at once -> (n_regions, 2*hidden)."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.out_dim = 2 * hidden

    def forward(self, x, edge_index, batch, n_regions):
        x = F.relu(self.c1(x, edge_index))
        x = F.relu(self.c2(x, edge_index))
        return _readout(x, batch, n_regions)


class HyperRegionEncoder(nn.Module):
    """2-layer Deep Sets hypergraph over all a slide's regions at once."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.c1 = DeepSetsHyperConv(in_dim, hidden)
        self.c2 = DeepSetsHyperConv(hidden, hidden)
        self.out_dim = 2 * hidden

    def forward(self, x, hyperedge_index, batch, n_regions, num_hyperedges):
        x = F.relu(self.c1(x, hyperedge_index, num_hyperedges))
        x = F.relu(self.c2(x, hyperedge_index, num_hyperedges))
        return _readout(x, batch, n_regions)


# ------------------------------------------------------------ packing a slide

def pack_pairwise(region_graphs):
    """Combine a slide's pairwise regions into one disconnected graph.
    region_graphs: list of (x, edge_index). Returns x, edge_index, batch, R."""
    xs, eis, batch = [], [], []
    node_off = 0
    for r, (x, ei) in enumerate(region_graphs):
        xs.append(x)
        eis.append(ei + node_off)
        batch.append(torch.full((x.size(0),), r, dtype=torch.long))
        node_off += x.size(0)
    return (torch.cat(xs, 0), torch.cat(eis, 1),
            torch.cat(batch), len(region_graphs))


def pack_hyper(region_graphs):
    """Combine a slide's hypergraph regions into one disconnected hypergraph.
    Node ids AND hyperedge ids are offset per region so nothing merges across
    regions. Returns x, hyperedge_index, batch, R, total_hyperedges."""
    xs, his, batch = [], [], []
    node_off, edge_off = 0, 0
    for r, (x, hi) in enumerate(region_graphs):
        xs.append(x)
        if hi.numel():
            h = hi.clone()
            h[0] += node_off
            h[1] += edge_off
            his.append(h)
            edge_off += int(hi[1].max()) + 1
        batch.append(torch.full((x.size(0),), r, dtype=torch.long))
        node_off += x.size(0)
    hyperedge_index = (torch.cat(his, 1) if his
                       else torch.empty((2, 0), dtype=torch.long))
    return (torch.cat(xs, 0), hyperedge_index,
            torch.cat(batch), len(region_graphs), edge_off)


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
        self.is_pw = arm.startswith("pw-")
        self.encoder = (PairwiseRegionEncoder(in_dim, hidden) if self.is_pw
                        else HyperRegionEncoder(in_dim, hidden))
        self.pool_kind = pool
        self.rpb = regions_per_batch
        if pool == "attention":
            self.pool = AttentionMIL(self.encoder.out_dim)
        self.head = nn.Linear(self.encoder.out_dim, n_classes)

    def _encode_regions(self, bag_graphs):
        """Encode all regions to (R, 2*hidden), in memory-bounded groups."""
        vecs = []
        for i in range(0, len(bag_graphs), self.rpb):
            group = bag_graphs[i:i + self.rpb]
            if self.is_pw:
                x, ei, batch, R = pack_pairwise(group)
                vecs.append(self.encoder(x, ei, batch, R))
            else:
                x, hi, batch, R, n_he = pack_hyper(group)
                vecs.append(self.encoder(x, hi, batch, R, n_he))
        return torch.cat(vecs, dim=0)          # attached: gradients flow through all groups

    def forward(self, bag_graphs):
        """bag_graphs: list of (x, struct) region tuples for one slide."""
        region_vecs = self._encode_regions(bag_graphs)
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