"""
mil.py
------
Multiple-instance learning for SLIDE-LEVEL labels.

The pattern task (Saltz PatternLabels: Brisk Diffuse / Brisk Band-like /
Non-Brisk Focal / Non-Brisk Multifocal) gives ONE label per slide, but a slide
is many regions. So each slide is a BAG of region graphs, and the model must:

    region graph --[GNN]--> node embeddings --[readout]--> region vector
    {region vectors} --[MIL pool]--> slide vector --[head]--> class

Two aggregations, deliberately separated so the same confound logic as the node
task applies:
  - the GNN inside a region is the arm (pairwise GCN vs Deep Sets hypergraph)
  - the MIL pool over regions is SHARED across arms, so it is not a confound

Attention MIL (Ilse et al. 2018) is used for the region pool: it learns which
regions matter and is the field standard for weakly-supervised WSI tasks. Mean
pooling is offered as a simpler fallback.

Also here: AbundanceOnly, the triviality control. It ignores all spatial
structure and predicts the label from per-slide cell-type fractions alone. If it
matches the graph arms, the task is abundance-driven and no spatial claim holds.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import scatter

from torch_geometric.nn import GCNConv
from models import DeepSetsHyperConv          # reuse the Deep Sets layer


# ------------------------------------------------------------ region encoders

def _readout(x):
    """Graph-level readout: concatenate mean and sum pooling over nodes.

    Mean captures composition, sum captures scale/count -- keeping both means the
    region vector retains how MANY as well as what FRACTION, which matters for a
    task with an abundance axis.
    """
    return torch.cat([x.mean(dim=0), x.sum(dim=0)], dim=-1)


class PairwiseRegionEncoder(nn.Module):
    """2-layer GCN over a region's cells -> region embedding (2*hidden)."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.out_dim = 2 * hidden

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.relu(self.c2(x, edge_index))
        return _readout(x)


class HyperRegionEncoder(nn.Module):
    """2-layer Deep Sets hypergraph over a region's cells -> region embedding."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.c1 = DeepSetsHyperConv(in_dim, hidden)
        self.c2 = DeepSetsHyperConv(hidden, hidden)
        self.out_dim = 2 * hidden

    def forward(self, x, hyperedge_index):
        x = F.relu(self.c1(x, hyperedge_index))
        x = F.relu(self.c2(x, hyperedge_index))
        return _readout(x)


# ------------------------------------------------------------ MIL aggregation

class AttentionMIL(nn.Module):
    """Gated-attention pooling over a bag of region vectors (Ilse et al. 2018).

    Learns a weight per region and returns the weighted sum, plus the weights
    themselves (useful for showing WHICH regions drove a prediction).
    """

    def __init__(self, dim, att_dim=64):
        super().__init__()
        self.v = nn.Linear(dim, att_dim)
        self.u = nn.Linear(dim, att_dim)
        self.w = nn.Linear(att_dim, 1)

    def forward(self, bag):                       # bag: (R, dim)
        a = torch.tanh(self.v(bag)) * torch.sigmoid(self.u(bag))
        a = torch.softmax(self.w(a), dim=0)       # (R, 1)
        return (a * bag).sum(dim=0), a.squeeze(-1)


class MILClassifier(nn.Module):
    """Region encoder + attention pool + linear head. One per arm."""

    def __init__(self, arm, in_dim, hidden, n_classes, pool="attention"):
        super().__init__()
        self.arm = arm
        if arm.startswith("pw-"):
            self.encoder = PairwiseRegionEncoder(in_dim, hidden)
        else:
            self.encoder = HyperRegionEncoder(in_dim, hidden)
        self.pool_kind = pool
        if pool == "attention":
            self.pool = AttentionMIL(self.encoder.out_dim)
        self.head = nn.Linear(self.encoder.out_dim, n_classes)

    def forward(self, bag_graphs):
        """bag_graphs: list of (x, struct) tuples, one per region in the slide."""
        region_vecs = torch.stack([self.encoder(x, s) for x, s in bag_graphs])
        if self.pool_kind == "attention":
            slide_vec, att = self.pool(region_vecs)
        else:
            slide_vec, att = region_vecs.mean(dim=0), None
        return self.head(slide_vec), att


class AbundanceOnly(nn.Module):
    """Triviality control: predict the label from per-slide cell-type fractions
    ONLY. No spatial structure, no graph. If this matches the graph arms, the
    task is abundance-driven and the spatial claim collapses.

    Input is a fixed-length vector of type fractions (+ optional total count),
    pooled across the slide -- so it is a plain MLP, not an MIL model.
    """

    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))

    def forward(self, slide_feature):
        return self.net(slide_feature)