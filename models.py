"""Everything learnable: hypergraph layers, the MIL classifier, capacity matching.

For a slide:

    region graph --[encoder]--> nodes --[readout]--> region vector
    {region vectors} --[attention pool]--> slide vector --[head]--> class

Everything downstream of the encoder is shared across arms, so the arm is the
only thing that varies.
"""

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (GCNConv, GraphConv, global_mean_pool,
                                global_add_pool)
from torch_geometric.utils import scatter

# ---------------------------------------------------------------- layer

def _back_by_family(he, node_idx, edge_idx, n, reduce, family_id, n_families):
    """Return path, split by hyperedge family. -> (n, n_families * he_dim).

    Scattering both families into ONE vector is destructive, not merely badly
    scaled: `back_i` is the sum of spatial and semantic contributions before any
    weight touches it, so no downstream linear can recover either. With the
    semantic family running ~4x the magnitude of the spatial one (measured), the
    combined arm would be semantic hyperedges plus spatial noise.

    Scattering per family and concatenating keeps them separable, so `out` can
    weight the two independently. Single-family arms take n_families=1 and this
    is exactly the old behaviour.
    """
    if n_families == 1:
        return scatter(he[edge_idx], node_idx, dim=0, dim_size=n, reduce=reduce)
    if family_id is None:
        raise ValueError(
            f"layer expects {n_families} hyperedge families but the packed bag "
            "carries no family_id. Either the cache predates family tagging "
            "(rerun precompute for this arm) or the arm name and the data "
            "disagree. Refusing to guess -- defaulting every hyperedge to "
            "family 0 would silently zero half the layer.")
    fam = family_id[edge_idx]                    # family of each INCIDENCE
    outs = []
    for f in range(n_families):
        m = fam == f
        if not bool(m.any()):                    # family absent in this region
            outs.append(he.new_zeros(n, he.size(1)))
            continue
        outs.append(scatter(he[edge_idx[m]], node_idx[m], dim=0,
                            dim_size=n, reduce=reduce))
    return torch.cat(outs, dim=1)


class DeepSetsHyperConv(nn.Module):
    """Set-aggregation hyperedge layer: node -> hyperedge -> node.

    rho(sum phi(x)), the Zaheer et al. universal form for permutation-invariant
    set functions. Pooling members is a SUM, preserving set cardinality that a
    mean divides out.

    use_size feeds log(|e|) to rho as its own channel. OFF by default, and that
    is the thesis-relevant setting: summation already carries cardinality in the
    magnitude of the total, so an explicit channel hands the model set size as a
    FEATURE rather than through the structure. Any advantage it produces ports
    to a pairwise graph in one line (GINLayer's use_degree, its twin), so
    enabling it on the primary arm would answer a feature-engineering question
    in place of a representational one. Run the two as an ablation pair instead.

    `back` controls the RETURN path: "mean" averages over the hyperedges
    containing a node and discards node degree; "sum" keeps it. GCNConv retains
    degree via 1/sqrt(d_i d_j), so back="mean" costs the hypergraph arms a
    density signal the pairwise baseline has.

    The node's own representation is combined with the returned hyperedge
    messages as (1 + eps) * x, matching GIN, rather than being concatenated.
    Concatenation is strictly more general -- W([b || x]) subsumes W((1+e)x + b)
    -- and that generality is exactly the problem: it would give the hypergraph
    arm freedom in combining a cell with its context that the pairwise arm does
    not have, in a comparison meant to isolate structure alone. rho therefore
    outputs in INPUT dimension so the addition is well-typed, again as in GIN,
    where the sum happens in input space and the outer map changes width.
    """

    def __init__(self, in_dim, out_dim, hidden=None, back="mean", n_families=1,
                 use_size=False, combine="add", rho_mlp=False):
        super().__init__()
        hidden = hidden or out_dim
        if combine not in ("add", "concat"):
            raise ValueError(f"combine must be 'add' or 'concat', got {combine!r}")
        self.combine = combine
        self.back = back
        self.n_families = n_families
        self.hidden = hidden
        self.in_dim = in_dim
        self.use_size = use_size
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        # rho stays SHARED across families -- log1p compresses 5 vs 200 members
        # to 1.8 vs 5.3, which one transform handles. Only the return path has
        # to be separated, because that is where the sum is irreversible.
        # rho_mlp gives rho a hidden layer, making it a universal approximator
        # of functions of the hyperedge total rather than one affine map and a
        # nonlinearity. It is the only transform in this layer that no stacking
        # argument reaches: GIN has one aggregation per layer, so its MLP always
        # sits between consecutive sums, whereas the node->hyperedge and
        # hyperedge->node sums here are separated by rho alone. Shaped like
        # GIN's MLP (Linear, ReLU, Linear) with no trailing nonlinearity.
        _rin = hidden + (1 if use_size else 0)
        self.rho = (nn.Sequential(nn.Linear(_rin, hidden), nn.ReLU(),
                                  nn.Linear(hidden, in_dim))
                    if rho_mlp else
                    nn.Sequential(nn.Linear(_rin, in_dim), nn.ReLU()))
        self.eps = nn.Parameter(torch.zeros(1))
        # Multi-family arms keep one vector per family so `out` can weight them
        # independently; a single projection back to in_dim restores the width
        # the addition needs while leaving the families separable inside it.
        self.proj = nn.Linear(in_dim * n_families, in_dim) if n_families > 1 else None
        self.out = nn.Linear(in_dim * (2 if combine == "concat" else 1), out_dim)

    def forward(self, x, hyperedge_index, num_hyperedges=None, family_id=None):
        node_idx, edge_idx = hyperedge_index[0], hyperedge_index[1]
        n = x.size(0)
        if num_hyperedges is None:
            num_hyperedges = int(edge_idx.max()) + 1 if edge_idx.numel() else 0
        if num_hyperedges == 0:                       # degenerate region
            # must still clear last_he: returning early with it unset leaves the
            # PREVIOUS region's groups in place, which a group readout would
            # then pool as if they belonged to this one
            self.last_he = x.new_zeros((0, self.in_dim))
            return self._combine(x, torch.zeros_like(x))
        m = self.phi(x)
        he = scatter(m[node_idx], edge_idx, dim=0,
                     dim_size=num_hyperedges, reduce="sum")
        if self.use_size:
            size = scatter(torch.ones_like(edge_idx, dtype=x.dtype), edge_idx,
                           dim=0, dim_size=num_hyperedges,
                           reduce="sum").unsqueeze(1)
            he = torch.cat([he, size.log1p()], dim=1)
        he = self.rho(he)
        # The group vector is otherwise transient: it is scattered back to the
        # member nodes and never referenced again, so nothing between it and the
        # prediction ever sees a hyperedge. Keeping it lets the encoder pool over
        # groups as well as cells -- the one thing a hypergraph offers that a
        # pairwise graph has no analogue for.
        self.last_he = he
        back = _back_by_family(he, node_idx, edge_idx, n, self.back,
                               family_id, self.n_families)
        if self.proj is not None:
            back = self.proj(back)
        return self._combine(x, back)

    def _combine(self, x, back):
        """How a node's own representation meets what came back from its edges.

        "add" is GIN's (1 + eps) x + b, and is the MATCHED setting: the pairwise
        arm combines self and context exactly this way, so the two arms differ
        in representation and in nothing else.

        "concat" is W([x ; b]), which is strictly more general -- it subsumes the
        addition -- and is therefore NOT matched. It exists for the unconstrained
        comparison: whether a hypergraph layer given freedom the pairwise arm
        does not have beats it. A win there is attributable to the combination
        as much as to the representation, which is why it is reported separately
        rather than as the primary arm.
        """
        if self.combine == "concat":
            return self.out(torch.cat([x, back], dim=1))
        return self.out((1 + self.eps) * x + back)

class SumHyperConv(nn.Module):
    """Sum-pooling hyperedge layer with ONE linear transform.

    DeepSetsHyperConv conflates two claims: that sum preserves cardinality, and
    that a learned per-member encoding expresses richer set functions. Only the
    first is the thesis, and it costs no parameters -- so this isolates it at a
    cost comparable to GCNConv. Run against DeepSetsHyperConv on the same
    construction to separate the two.

    `back` as in DeepSetsHyperConv.
    """

    def __init__(self, in_dim, out_dim, hidden=None, back="mean", n_families=1):
        super().__init__()
        # `hidden` unused; signature matches DeepSetsHyperConv
        self.back = back
        self.n_families = n_families
        self.out = nn.Linear((n_families + 1) * in_dim, out_dim)

    def forward(self, x, hyperedge_index, num_hyperedges=None, family_id=None):
        node_idx, edge_idx = hyperedge_index[0], hyperedge_index[1]
        n = x.size(0)
        if num_hyperedges is None:
            num_hyperedges = int(edge_idx.max()) + 1 if edge_idx.numel() else 0
        if num_hyperedges == 0:                       # degenerate region
            return self.out(torch.cat(
                [x.new_zeros(n, x.size(1) * self.n_families), x], dim=1))
        # unnormalised: a hyperedge of 12 yields twice the magnitude of one of 6
        he = scatter(x[node_idx], edge_idx, dim=0,
                     dim_size=num_hyperedges, reduce="sum")
        back = _back_by_family(he, node_idx, edge_idx, n, self.back,
                               family_id, self.n_families)
        return self.out(torch.cat([back, x], dim=1))


class GINLayer(nn.Module):
    """GIN (Xu et al. 2019): MLP((1+eps) x_i + sum_j x_j).

    The PAIRWISE realisation of rho(sum phi(x)) -- GIN is derived from the same
    Deep Sets result DeepSetsHyperConv uses, over an adjacency list instead of a
    hyperedge. It is therefore the arm that says whether DeepSetsHyperConv's
    margin comes from the hyperedge or just from having an MLP per hop.

    use_degree concatenates log1p(deg_i), the pairwise twin of the log1p(|e|)
    channel DeepSetsHyperConv feeds to rho. That isolates the last mechanism
    that is not shared: cardinality as its OWN input rather than as the
    magnitude of the sum. If this closes the gap, explicit set size explains the
    result and it was never a hypergraph property -- it ports to a graph in one
    line, which is this one.
    """

    def __init__(self, in_dim, out_dim, use_degree=False):
        super().__init__()
        self.use_degree = use_degree
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + (1 if use_degree else 0), out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim))

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        n = x.size(0)
        agg = scatter(x[src], dst, dim=0, dim_size=n, reduce="sum")
        h = (1 + self.eps) * x + agg
        if self.use_degree:
            # edges are symmetrised with no self-loops, so this is |N(i)|;
            # hg-radius sees |e_i| = deg_i + 1, the same quantity offset by one
            deg = scatter(torch.ones_like(dst, dtype=x.dtype), dst, dim=0,
                          dim_size=n, reduce="sum").unsqueeze(1)
            h = torch.cat([h, deg.log1p()], dim=1)
        return self.mlp(h)


def n_families(arm):
    """How many hyperedge families the construction carries.

    Derived from the arm NAME rather than from data, because the layer's weight
    shapes depend on it and must be fixed before any batch is seen. A region that
    happens to contain no semantic hyperedges would otherwise silently build a
    narrower layer than one that does.
    """
    return 2 if parse_arm(arm)[0].endswith("+semantic") else 1


def parse_arm(arm):
    """'hg-knn@sum' -> ('hg-knn', 'sum');  'hg-knn' -> ('hg-knn', None).

    The construction half selects which cached graph to read; the aggregation
    half selects the layer. Encoding both in the arm name means variants can run
    in the SAME sweep on the same folds, paired run-for-run.

    None means the arm named no aggregation, and each encoder supplies its own
    default -- deepsets for hypergraphs, gcn for pairwise.
    """
    construction, _, agg = arm.partition("@")
    return construction, (agg or None)


# ------------------------------------------------------------ region encoders

# Fixed width every encoder emits, so AttentionMIL and the head are identical
# across arms and capacity matching only has to equalise the encoders. When
# out_dim was 2*hidden, widening an encoder widened the pool with it.
REGION_DIM = 64


def _readout(x, batch, n_regions):
    """Per-region mean+sum+std pooling. batch[i] = region node i belongs to.

    Mean is composition, sum is scale, std is DISPERSION -- how much local
    structure varies across the region.

    The third moment is not a refinement. Focal (one dense aggregate, most cells
    isolated) and Diffuse (uniform moderate density) differ in the SPREAD of
    per-cell density, not its centre, so a mean+sum readout cannot represent the
    distinction the labels turn on however good the encoder is. Both are pooled
    over ~2000 cells, which is where a per-node difference goes to die.
    """
    mean = global_mean_pool(x, batch, size=n_regions)
    summ = global_add_pool(x, batch, size=n_regions)
    # E[x^2] - E[x]^2, clamped because fp error takes it slightly negative.
    # The epsilon is load-bearing: sqrt has an infinite gradient at 0, and a
    # dead ReLU channel gives var EXACTLY 0, so sqrt(var) alone yields NaN.
    var = (global_mean_pool(x * x, batch, size=n_regions)
           - mean * mean).clamp(min=0)
    return torch.cat([mean, summ, (var + 1e-6).sqrt()],
                     dim=-1)                        # (n_regions, 3*hidden)


class PairwiseRegionEncoder(nn.Module):
    """2-layer pairwise encoder over all a slide's regions at once.

    Emits REGION_DIM, via the same linear adapter as HyperRegionEncoder.

    Each agg is the pairwise twin of one hypergraph arm, so a matched pair
    isolates one mechanism at a time:

    agg="gcn"     GCNConv, degree-normalised by 1/sqrt(d_i d_j). THE DEFAULT,
                  and what every pairwise result before this used.
    agg="sum"     GraphConv (Morris et al.), W1 x_i + W2 sum_j x_j. Twin of
                  SumHyperConv -- also one linear over [aggregate ; self], same
                  parameter count. Isolates unnormalised sum.
    agg="gin"     GIN. Twin of DeepSetsHyperConv without its size channel.
                  Isolates the per-hop MLP.
    agg="gin+deg" GIN with log1p(deg) concatenated. Twin of DeepSetsHyperConv
                  WITH its size channel. Isolates explicit cardinality.

    n_layers sets the receptive field: each layer is one hop. A hypergraph layer
    over neighbourhood hyperedges covers TWO hops (node -> hyperedge -> node), so
    a 2-layer hypergraph arm reaches as far as a 4-layer pairwise one. Varying
    this measures the exchange rate between hyperedge encoding and depth -- what
    the hyperedge buys, not whether it won fairly.

    Without these the aggregation function was confounded with the construction:
    every pw arm normalised, every hg arm summed, so "hypergraphs win" and "sum
    beats normalisation" (Xu et al., GIN) were not separable.
    """

    AGGS = {"gcn": GCNConv,
            "sum": GraphConv,
            "gin": GINLayer,
            "gin+deg": partial(GINLayer, use_degree=True)}

    def __init__(self, in_dim, hidden, out_dim=REGION_DIM, agg=None, n_layers=2):
        super().__init__()
        agg = agg or "gcn"
        if agg not in self.AGGS:
            raise ValueError(f"unknown pairwise agg {agg!r}; expected one of "
                             f"{sorted(self.AGGS)}")
        if n_layers < 1:
            raise ValueError(f"n_layers={n_layers}: need at least one")
        Conv = self.AGGS[agg]
        self.agg = agg
        self.n_layers = n_layers
        dims = [in_dim] + [hidden] * n_layers
        self.convs = nn.ModuleList([Conv(dims[i], dims[i + 1])
                                    for i in range(n_layers)])
        self.proj = nn.Linear(3 * hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x, edge_index, batch, n_regions):
        for c in self.convs:
            x = F.relu(c(x, edge_index))
        return self.proj(_readout(x, batch, n_regions))


class StarRegionEncoder(nn.Module):
    """GIN over the star (bipartite incidence) expansion -> (n_regions, REGION_DIM).

    The lossless pairwise comparator.

    DEPTH DEFAULT IS AN ACTIVE DECISION, recorded here, printed at run time and
    stored in the results JSON. It is not an accident of implementation.

    Layers are not comparable units across the two architectures: a
    DeepSetsHyperConv layer is two scatters and three transforms, a star GIN
    layer is one scatter and one MLP, so two star layers roughly equal one
    hypergraph layer in both hops and transform count. At n_layers=2 the
    hypergraph arm would hold 2x the reach; n_layers=4 matches reach instead.

    DEFAULT 4 -- the setting that favours the BASELINE. Capacity matching
    equalises total parameters, so the extra depth is not extra budget: star
    trades width for depth (hidden 44 -> 34) and takes on oversmoothing risk.
    Chosen so that a hypergraph win cannot be attributed to reach the baseline
    was denied.

    Neither depth is "the fair one". Sweep 2/3/4 and report the depth at which
    star catches the hypergraph -- that exchange rate is a measurement a reader
    can argue with, where a single chosen depth is a judgement they cannot.

    GIN rather than GCNConv: the star graph is bipartite and wildly
    degree-imbalanced (a 196-member hyperedge node against a cell in ~8
    hyperedges), so 1/sqrt(d_i d_j) would divide out exactly the cardinality
    signal the comparison exists to test.

    The readout pools over CELL nodes only -- see pack_star.
    """

    def __init__(self, in_dim, hidden, out_dim=REGION_DIM, n_layers=4):
        super().__init__()
        if n_layers < 2:
            raise ValueError(
                f"n_layers={n_layers}: information cannot reach a cell in under "
                "two hops on a star graph (cell -> hyperedge -> cell), and "
                "hyperedge nodes start at zero, so one layer tells cells nothing")
        dims = [in_dim] + [hidden] * n_layers
        self.convs = nn.ModuleList([GINLayer(dims[i], dims[i + 1])
                                    for i in range(n_layers)])
        self.n_layers = n_layers
        self.proj = nn.Linear(3 * hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x, edge_index, batch, n_regions, cell_mask):
        for c in self.convs:
            x = F.relu(c(x, edge_index))
        return self.proj(_readout(x[cell_mask], batch[cell_mask], n_regions))


class HyperRegionEncoder(nn.Module):
    """2-layer hypergraph encoder over all a slide's regions at once.

    Emits REGION_DIM like the pairwise encoder, via the same linear adapter.

    agg="deepsets" -> DeepSetsHyperConv, the general rho(sum phi(x)) form
    agg="sum"      -> SumHyperConv, sum pooling with one transform
    Both preserve cardinality; only the first also learns what to aggregate.
    """

    # name -> (layer, return-path reduction, explicit size channel). `2` means
    # sum on the way back too, so node incidence degree survives; plain names
    # keep the mean earlier results used. "+size" feeds log|e| to rho as its own
    # input -- the ablation twin of the pairwise "gin+deg", and off everywhere
    # else so that cardinality reaches the model through the structure rather
    # than as a hand-supplied feature.
    # (layer, return path, size channel, self/context combine, rho MLP, group readout)
    AGGS = {"deepsets":       (DeepSetsHyperConv, "mean", False, "add", False, False),
            "deepsets2":      (DeepSetsHyperConv, "sum",  False, "add", False, False),
            "deepsets+size":  (DeepSetsHyperConv, "mean", True,  "add", False, False),
            "deepsets2+size": (DeepSetsHyperConv, "sum",  True,  "add", False, False),
            # UNMATCHED: more general than GIN's combination, so it answers
            # "does a hypergraph layer win when unconstrained", not the
            # representational question the primary arms are built for
            "deepsets2+cat":  (DeepSetsHyperConv, "sum",  False, "concat", False, False),
            # rho as a perceptron: the strictly section-3.5-compliant arm. Run
            # against deepsets2 to measure what the simplification costs
            "deepsets2+mlp":  (DeepSetsHyperConv, "sum",  False, "add", True, False),
            # UNMATCHED: pools over hyperedges as well as cells, so the region is
            # summarised by its groups and not only by its members. The pairwise
            # arm has no counterpart -- GIN computes no edge representation -- so
            # this answers "does the group representation help when it is
            # actually used", not the matched question the primary arms are for
            "deepsets2+group": (DeepSetsHyperConv, "sum", False, "add", False, True),
            "sum":            (SumHyperConv, "mean", False, "add", False, False),
            "sum2":           (SumHyperConv, "sum",  False, "add", False, False)}

    def __init__(self, in_dim, hidden, out_dim=REGION_DIM, agg=None,
                 n_families=1):
        super().__init__()
        agg = agg or "deepsets"
        if agg not in self.AGGS:
            raise ValueError(f"unknown hypergraph agg {agg!r}; expected one of "
                             f"{sorted(self.AGGS)}")
        Conv, back, use_size, combine, rho_mlp, group = self.AGGS[agg]
        self.agg = agg
        self.n_families = n_families
        self.group_readout = group
        kw = ({"use_size": use_size, "combine": combine, "rho_mlp": rho_mlp}
              if Conv is DeepSetsHyperConv else {})
        self.c1 = Conv(in_dim, hidden, back=back, n_families=n_families, **kw)
        self.c2 = Conv(hidden, hidden, back=back, n_families=n_families, **kw)
        # six moments rather than three when groups are pooled alongside cells
        self.proj = nn.Linear((6 if group else 3) * hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x, hyperedge_index, batch, n_regions, num_hyperedges,
                family_id=None):
        x = F.relu(self.c1(x, hyperedge_index, num_hyperedges, family_id))
        x = F.relu(self.c2(x, hyperedge_index, num_hyperedges, family_id))
        r = _readout(x, batch, n_regions)
        if self.group_readout:
            he = self.c2.last_he
            # a hyperedge lies wholly within one region, so its region is that of
            # any member; take the first incidence of each
            he_batch = torch.zeros(he.size(0), dtype=torch.long, device=x.device)
            he_batch.scatter_(0, hyperedge_index[1], batch[hyperedge_index[0]])
            r = torch.cat([r, _readout(he, he_batch, n_regions)], dim=1)
        return self.proj(r)


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
    regions. Returns x, hyperedge_index, batch, R, total_hyperedges, family_id.

    Regions are (x, hyperedge_index) or (x, hyperedge_index, family_id). The
    family vector is indexed by GLOBAL hyperedge id, so it has to be padded to
    edge_off for every region -- a region whose hyperedge count exceeds its own
    tags would otherwise shift every later region's families by the shortfall.
    family_id is None when no region carried one.
    """
    xs, his, batch, fams = [], [], [], []
    node_off, edge_off = 0, 0
    dev = region_graphs[0][0].device
    any_fam = any(len(g) > 2 and g[2] is not None for g in region_graphs)
    for r, g in enumerate(region_graphs):
        x, hi = g[0], g[1]
        fam = g[2] if len(g) > 2 else None
        xs.append(x)
        if hi.numel():
            h = hi.clone()
            h[0] += node_off
            h[1] += edge_off
            his.append(h)
            n_he = int(hi[1].max()) + 1
            if any_fam:
                fams.append(fam.to(dev) if fam is not None
                            else torch.zeros(n_he, dtype=torch.long, device=dev))
            edge_off += n_he
        batch.append(torch.full((x.size(0),), r, dtype=torch.long, device=dev))
        node_off += x.size(0)
    hyperedge_index = (torch.cat(his, 1) if his
                       else torch.empty((2, 0), dtype=torch.long, device=dev))
    family_id = torch.cat(fams) if fams else None
    return (torch.cat(xs, 0), hyperedge_index,
            torch.cat(batch), len(region_graphs), edge_off, family_id)


def permute_within_regions(chunk, generator):
    """Relabel a packed chunk's nodes within each region, structure only.

    Every other ablation replaces an input with another slide's, which for the
    graph branch removes the node features along with the structure and so
    cannot say which of the two the model was using. This one leaves the
    features, the region membership and the topology exactly as they were, and
    changes only WHICH cell sits at each position in the graph. A drop is
    therefore attributable to the correspondence between a cell and its
    neighbourhood -- it cannot be explained by composition at any scale, since
    the multiset of features in every region is untouched.

    Permuting within regions rather than across the whole packed chunk matters:
    a global permutation would connect cells in different regions, which changes
    the topology instead of preserving it.
    """
    x, struct, batch = chunk[0], chunk[1], chunk[2]
    perm = torch.empty_like(batch)
    for r in range(int(batch.max()) + 1 if batch.numel() else 0):
        idx = (batch == r).nonzero(as_tuple=True)[0]
        # CPU randperm then move: a CPU generator with a CUDA device errors,
        # and seeding per device would make the ablation depend on where it ran
        order = torch.randperm(idx.numel(), generator=generator).to(idx.device)
        perm[idx] = idx[order]
    if struct.numel() == 0:
        return chunk
    new = struct.clone()
    new[0] = perm[struct[0]]                 # node row, for both packings
    if new.size(0) > 1 and chunk[1].size(0) == 2 and len(chunk) == 4:
        new[1] = perm[struct[1]]             # pairwise: both rows are nodes
    return (x, new) + tuple(chunk[2:])


def pack_star(region_graphs):
    """Star-expand a slide's hypergraph regions into ONE bipartite graph.

    Adds a node per hyperedge, joined to its members. This is the only LOSSLESS
    pairwise encoding of a hypergraph, and unlike clique expansion it is O(1) per
    incidence rather than quadratic in cardinality -- so it is the tractable
    pairwise comparator for arms whose hyperedges reach 196 members.

    It is also, transparently, the hypergraph drawn differently: one hypergraph
    layer is node->hyperedge->node, which is exactly TWO hops here. A star arm
    therefore needs 2x the layers to match a hypergraph arm's reach, and that
    factor is the measurement, not an inconvenience.

    Hyperedge nodes get ZERO features. Seeding them with the mean of their
    members would perform the aggregation before the network does, which is the
    thing under test. Returns x, edge_index, batch, R, cell_mask -- cell_mask is
    load-bearing: the readout must pool over CELLS only, or every region vector
    gets diluted by however many hyperedges it happens to contain.
    """
    xs, eis, batch, mask = [], [], [], []
    node_off = 0
    dev = region_graphs[0][0].device
    for r, g in enumerate(region_graphs):
        x, hi = g[0], g[1]
        n_cells = x.size(0)
        n_he = (int(hi[1].max()) + 1) if hi.numel() else 0
        xs.append(torch.cat([x, x.new_zeros(n_he, x.size(1))], dim=0))
        if hi.numel():
            # incidence -> undirected edge between member and its hyperedge node
            src = hi[0] + node_off
            dst = hi[1] + node_off + n_cells
            eis.append(torch.stack([torch.cat([src, dst]),
                                    torch.cat([dst, src])]))
        total = n_cells + n_he
        batch.append(torch.full((total,), r, dtype=torch.long, device=dev))
        m = torch.zeros(total, dtype=torch.bool, device=dev)
        m[:n_cells] = True
        mask.append(m)
        node_off += total
    edge_index = (torch.cat(eis, 1) if eis
                  else torch.empty((2, 0), dtype=torch.long, device=dev))
    return (torch.cat(xs, 0), edge_index, torch.cat(batch),
            len(region_graphs), torch.cat(mask))


def pack_mode(arm):
    """'pw' | 'hyper' | 'star' -- which packer an arm needs.

    Star arms read the HYPERGRAPH cache (the incidence structure is the bipartite
    edge list), so the construction half of the name is unchanged and only the
    aggregation half selects the encoding.
    """
    construction, agg = parse_arm(arm)
    if construction.startswith("pw-"):
        return "pw"
    return "star" if agg == "star" else "hyper"


def pack_bag(region_graphs, mode, regions_per_batch=16):
    """Pack one slide's regions into memory-bounded groups, once.

    Deterministic in the region graphs, so call it per slide before training and
    reuse the result rather than repeating the torch.cat on every pass. Region
    boundaries are never split, so grouping changes no region vector.

    mode: 'pw' | 'hyper' | 'star', from pack_mode(arm). A bool is accepted for
    the old is_pw calling convention.
    """
    if isinstance(mode, bool):
        mode = "pw" if mode else "hyper"
    packer = {"pw": pack_pairwise, "hyper": pack_hyper,
              "star": pack_star}[mode]
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

    Pool and head are sized from encoder.out_dim, which region_dim pins, so the
    encoder is the only thing that differs in parameter count across arms.
    train_patterns.py asserts this at startup.

    region_dim and att_dim are the REAL capacity knob, not `hidden`. The pool and
    head cost 2*region_dim*att_dim + ... regardless of hidden, which at the
    defaults is 8,645 parameters -- 80% of the whole model once hidden drops to
    8. Shrinking `hidden` alone cannot take total capacity below ~9,700, so on
    ~73 training slides it is tuning a fifth of the model.

    Regions are encoded in groups of `regions_per_batch`: full-slide batching
    OOMs on large slides with high-cardinality hypergraphs. Whole regions per
    group and no detaching, so region vectors and gradients are unchanged.
    """

    def __init__(self, arm, in_dim, hidden, n_classes, pool="attention",
                 regions_per_batch=16, blend_families=False, star_layers=4,
                 region_dim=REGION_DIM, att_dim=64,
                 abundance_dim=0, path_dropout=0.25, abundance_hidden=32,
                 pw_layers=2):
        super().__init__()
        self.arm = arm
        construction, agg = parse_arm(arm)
        self.construction, self.agg = construction, agg
        self.is_pw = construction.startswith("pw-")
        # blend_families is the ABLATION: force one family so both scatter into
        # the same vector, which is what the arm does if you ignore that a
        # 200-cell semantic hyperedge and a 5-cell spatial one are different
        # objects. Run it once to show it fails; it is not a usable setting.
        nf = 1 if blend_families else n_families(arm)
        self.mode = pack_mode(arm)
        if self.mode == "pw":
            self.encoder = PairwiseRegionEncoder(in_dim, hidden, region_dim,
                                                 agg=agg, n_layers=pw_layers)
        elif self.mode == "star":
            self.encoder = StarRegionEncoder(in_dim, hidden, region_dim,
                                             n_layers=star_layers)
        else:
            self.encoder = HyperRegionEncoder(in_dim, hidden, region_dim,
                                              agg=agg, n_families=nf)
        self.pool_kind = pool
        self.rpb = regions_per_batch
        if pool == "attention":
            self.pool = AttentionMIL(self.encoder.out_dim, att_dim)
        # Abundance skip: concatenate the per-slide cell-type fractions to the
        # pooled slide vector, so the head starts from at-least-abundance and
        # topology can only add. Turns "can a graph model beat a composition
        # baseline from scratch" (an optimisation question) into "given
        # composition, does structure add" (the question actually of interest).
        #
        # path_dropout zeroes ONE path at random per training step -- abundance
        # with probability p, the graph vector with probability p, neither
        # otherwise. It must be SYMMETRIC. Dropping only abundance leaves the
        # head having never seen a zeroed graph vector, so the graph-zeroed
        # ablation measures distribution shift rather than information loss and
        # reads far below the floor. That breaks the one comparison the skip
        # exists to make: full versus graph-zeroed, within the same model.
        # Mutually exclusive so at least one path always carries signal.
        #
        # The abundance branch gets its OWN hidden layer, matching AbundanceOnly
        # (Linear -> ReLU -> Linear, where the head is the second Linear).
        # Feeding the 6 raw fractions straight into the head instead makes the
        # abundance path linear while the graph path is deep, so `graph-zeroed`
        # measures a crippled model rather than what composition carries -- and
        # `full - graph-zeroed` then OVERSTATES structure's contribution, which
        # is the direction that flatters the hypothesis. Measured on synthetic
        # data where abundance is near-perfectly predictive: 0.24 linear against
        # 0.97 for the control.
        if not 0 <= 2 * path_dropout <= 1:
            raise ValueError(f"path_dropout={path_dropout}: each path is "
                             "dropped with this probability and they are "
                             "mutually exclusive, so it cannot exceed 0.5")
        self.abundance_dim = abundance_dim
        self.path_dropout = path_dropout
        ab_out = 0
        if abundance_dim:
            ab_out = abundance_hidden
            self.abundance_enc = nn.Sequential(
                nn.Linear(abundance_dim, ab_out), nn.ReLU())
        self.head = nn.Linear(self.encoder.out_dim + ab_out, n_classes)

    def _encode_regions(self, packed):
        """Encode pre-packed groups to (R, REGION_DIM)."""
        vecs = []
        for g in packed:
            if self.mode == "pw":
                x, ei, batch, R = g
                vecs.append(self.encoder(x, ei, batch, R))
            elif self.mode == "star":
                x, ei, batch, R, cell_mask = g
                vecs.append(self.encoder(x, ei, batch, R, cell_mask))
            else:
                x, hi, batch, R, n_he, fam = g
                vecs.append(self.encoder(x, hi, batch, R, n_he, fam))
        return torch.cat(vecs, dim=0)          # attached: gradients flow through all groups

    def forward(self, bag, abundance=None, drop_graph=False):
        """bag: raw regions, or pre-packed groups from pack_bag(). Prefer
        pre-packed -- packing here repeats identical work on every epoch.

        Raw regions are (x, struct), or (x, struct, family_id) for multi-family
        arms; packed groups are 4-tuples (pairwise) or 5-tuples (hypergraph).
        The arity test must accept BOTH raw widths or a 3-tuple region list gets
        mistaken for pre-packed and fails somewhere unrelated.

        abundance: per-slide cell-type fractions, used only when abundance_dim>0.
        Pass None (or drop_graph=True) to run an ablation; see path_dropout for
        why both ablations are in-distribution.

        drop_graph: zero the pooled slide vector, leaving composition only.
        Together the two ablations decompose the model from one set of weights:
        full vs graph-zeroed is what structure adds, full vs abundance-zeroed is
        what composition adds.
        """
        packed = (pack_bag(bag, self.mode, self.rpb)
                  if bag and len(bag[0]) in (2, 3) else bag)
        region_vecs = self._encode_regions(packed)
        if self.pool_kind == "attention":
            slide_vec, att = self.pool(region_vecs)
        else:
            slide_vec, att = region_vecs.mean(dim=0), None

        if self.abundance_dim:
            drop_a = abundance is None
            if self.training and self.path_dropout > 0:
                r = float(torch.rand(()))
                if r < self.path_dropout:
                    drop_a = True
                elif r < 2 * self.path_dropout:
                    drop_graph = True
            a = self.abundance_enc(
                slide_vec.new_zeros(self.abundance_dim) if drop_a else abundance)
            if drop_graph:
                slide_vec = torch.zeros_like(slide_vec)
            slide_vec = torch.cat([slide_vec, a])
        elif drop_graph:
            slide_vec = torch.zeros_like(slide_vec)
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

    def __init__(self, arm, in_dim, hidden, n_classes, blend_families=False):
        super().__init__()
        construction, agg = parse_arm(arm)
        self.is_pw = construction.startswith("pw-")
        if self.is_pw:
            agg = agg or "gcn"
            if agg not in PairwiseRegionEncoder.AGGS:
                raise ValueError(f"unknown pairwise agg {agg!r}; expected one "
                                 f"of {sorted(PairwiseRegionEncoder.AGGS)}")
            Conv = PairwiseRegionEncoder.AGGS[agg]
            self.c1, self.c2 = Conv(in_dim, hidden), Conv(hidden, hidden)
        else:
            agg = agg or "deepsets"
            if agg not in HyperRegionEncoder.AGGS:
                raise ValueError(f"unknown hypergraph agg {agg!r}; expected one "
                                 f"of {sorted(HyperRegionEncoder.AGGS)}")
            Conv, back, use_size = HyperRegionEncoder.AGGS[agg]
            nf = 1 if blend_families else n_families(arm)
            kw = {"use_size": use_size} if Conv is DeepSetsHyperConv else {}
            self.c1 = Conv(in_dim, hidden, back=back, n_families=nf, **kw)
            self.c2 = Conv(hidden, hidden, back=back, n_families=nf, **kw)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, struct, num_hyperedges=None, family_id=None):
        if self.is_pw:
            x = F.relu(self.c1(x, struct))
            x = F.relu(self.c2(x, struct))
        else:
            x = F.relu(self.c1(x, struct, num_hyperedges, family_id))
            x = F.relu(self.c2(x, struct, num_hyperedges, family_id))
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
