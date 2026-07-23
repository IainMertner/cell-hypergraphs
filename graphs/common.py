"""Shared helpers. Everything here is used by more than one construction."""

import numpy as np
import torch
from torch_geometric.data import Data

# PanNuke taxonomy (CellViT): type indices are 1..5
TYPE_MAP = {1: "Neoplastic", 2: "Inflammatory", 3: "Connective",
            4: "Dead", 5: "Epithelial"}
N_TYPES = 5
N_MORPH = 5  # area, perimeter, circularity, eccentricity, extent


def microns_to_px(um, mpp):
    """Convert a distance in microns to slide pixels."""
    return um / mpp


def node_features(types, morph=None):
    """(N, 5) one-hot type, optionally concatenated with (N, 5) morphology."""
    n = len(types)
    x = torch.zeros((n, N_TYPES), dtype=torch.float)
    x[torch.arange(n), torch.from_numpy(types - 1).long()] = 1.0
    if morph is None:
        return x
    return torch.cat([x, torch.from_numpy(morph).float()], dim=1)


def make_pairwise(centroids, types, edge_index, morph=None):
    """Assemble a pairwise-graph Data object."""
    d = Data(x=node_features(types, morph),
             pos=torch.tensor(centroids, dtype=torch.float),
             edge_index=edge_index)
    d.cell_type = torch.from_numpy(types).long()
    return d


def make_hyper(centroids, types, hyperedge_index, num_hyperedges, morph=None,
               family_id=None):
    """Assemble a hypergraph Data object.

    hyperedge_index : (2, n_incidences) -- row 0 node id, row 1 hyperedge id
    """
    d = Data(x=node_features(types, morph),
             pos=torch.tensor(centroids, dtype=torch.float))
    d.hyperedge_index = hyperedge_index
    d.num_hyperedges = num_hyperedges
    d.cell_type = torch.from_numpy(types).long()
    if family_id is not None:
        d.family_id = family_id
    return d


def symmetrise(pairs):
    """(E,2) unique undirected pairs -> (2, 2E) directed edge_index, deduped."""
    if len(pairs) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    both = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    both = np.unique(both.T, axis=1)
    return torch.from_numpy(both).long().contiguous()


def incidences_from_groups(groups, n_nodes):
    """[array_of_member_ids, ...] -> (hyperedge_index, num_hyperedges).

    Groups with fewer than 2 members are dropped: a singleton hyperedge carries
    no set information.
    """
    keep = [g for g in groups if len(g) >= 2]
    if not keep:
        return torch.empty((2, 0), dtype=torch.long), 0
    node = np.concatenate(keep)
    edge = np.concatenate([np.full(len(g), i, dtype=np.int64)
                           for i, g in enumerate(keep)])
    hi = torch.from_numpy(np.stack([node, edge])).long().contiguous()
    return hi, len(keep)


def structural_stats(name, data, n_nodes):
    """Deterministic structural statistics (pre-reg 8.2). No seeds, no training.

    Works for both pairwise (edge_index) and hypergraph (hyperedge_index) Data.
    'clique_edges' is the number of undirected pairwise edges needed to encode
    the same grouping; 'expansion' is that cost relative to the hypergraph's
    incidence count -- the measured version of "how badly does clique expansion
    blow up for this construction".
    """
    if hasattr(data, "hyperedge_index") and data.hyperedge_index.numel():
        hi = data.hyperedge_index
        n_he = int(data.num_hyperedges)
        n_inc = int(hi.shape[1])
        sizes = np.bincount(hi[1].numpy(), minlength=n_he)
        degrees = np.bincount(hi[0].numpy(), minlength=n_nodes)
        clique_edges = int((sizes * (sizes - 1) // 2).sum())
        return dict(name=name, kind="hyper", units=n_he, incidences=n_inc,
                    mean_size=float(sizes.mean()),
                    median_size=float(np.median(sizes)),
                    max_size=int(sizes.max()),
                    mean_degree=float(degrees.mean()),
                    clique_edges=clique_edges,
                    expansion=clique_edges / n_inc)
    ei = data.edge_index
    n_dir = int(ei.shape[1])
    und = n_dir // 2
    degrees = np.bincount(ei[0].numpy(), minlength=n_nodes)
    return dict(name=name, kind="pair", units=und, incidences=n_dir,
                mean_size=2.0, median_size=2.0, max_size=2,
                mean_degree=float(degrees.mean()),
                clique_edges=und, expansion=1.0)


def print_stats_table(rows):
    hdr = (f"{'construction':<18}{'units':>9}{'incid':>10}"
           f"{'mean|med|max':>16}{'node deg':>10}{'clique edges':>14}{'expansion':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        s = f"{r['mean_size']:.1f}|{r['median_size']:.0f}|{r['max_size']}"
        print(f"{r['name']:<18}{r['units']:>9,}{r['incidences']:>10,}"
              f"{s:>16}{r['mean_degree']:>10.1f}"
              f"{r['clique_edges']:>14,}{r['expansion']:>10.1f}x")