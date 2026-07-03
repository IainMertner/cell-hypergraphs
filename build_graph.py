"""
build_graph.py
--------------
Minimal pure-cell-graph construction for CellViT output.

Pipeline:
  cells.json  ->  (centroids, types)  ->  one region  ->  radius graph (PyG Data)

Design choices (deliberately simple v1):
  * Nodes are cells. Node position = CellViT 'centroid'. Node feature = one-hot
    PanNuke type (5 classes). Nothing else yet.
  * Edges: an undirected edge between two cells whose centroids are within
    `radius` microns of each other (a "radius graph").
  * Edges are built with a CPU KD-tree (scipy), NOT torch-cluster, so there is
    no compiled-CUDA dependency to break. Plenty fast at region scale.
  * "Tiling" just bounds the spatial extent. Same construction on a sub-region
    as on the whole slide; the region is one labelled sample.
"""

import json
import numpy as np
import torch
from scipy.spatial import cKDTree, Delaunay
from torch_geometric.data import Data

# PanNuke taxonomy from CellViT: type indices are 1..5
TYPE_MAP = {1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead", 5: "Epithelial"}
N_TYPES = 5


def _poly_features(contour):
    """Nuclear shape descriptors from a cell contour (CGC-Net style).
    Returns [area, perimeter, circularity, eccentricity, extent]."""
    c = np.asarray(contour, dtype=np.float64)
    if len(c) < 3:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    x, y = c[:, 0], c[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    dd = np.diff(c, axis=0, append=c[:1])
    perim = float(np.sqrt((dd ** 2).sum(1)).sum())
    circ = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0.0
    cen = c - c.mean(0)
    ev = np.linalg.eigvalsh(np.cov(cen.T))
    ev = np.clip(ev, 1e-9, None)
    minor, major = 2 * np.sqrt(ev[0]), 2 * np.sqrt(ev[1])
    ecc = np.sqrt(1 - (minor ** 2) / (major ** 2)) if major > 0 else 0.0
    bb = (x.max() - x.min()) * (y.max() - y.min())
    extent = area / bb if bb > 0 else 0.0
    return [area, perim, circ, ecc, extent]


N_MORPH = 5  # area, perimeter, circularity, eccentricity, extent


def load_cells(path, with_morphology=False):
    """Read CellViT cells.json -> centroids (N,2), types (N,) in 1..5, mpp.

    If with_morphology=True, also returns a z-scored (N,5) morphology matrix
    computed from each cell's contour. (One-time loop over all cells; this is
    the same extraction that will run per-slide in the cohort pipeline.)
    """
    with open(path) as f:
        d = json.load(f)
    mpp = float(d["wsi_metadata"]["base_mpp"])
    cells = d["cells"]
    centroids = np.fromiter(
        (coord for c in cells for coord in c["centroid"]), dtype=np.float64
    ).reshape(-1, 2)
    types = np.fromiter((c["type"] for c in cells), dtype=np.int64)
    if not with_morphology:
        return centroids, types, mpp

    morph = np.array([_poly_features(c["contour"]) for c in cells], dtype=np.float64)
    mu, sd = morph.mean(0), morph.std(0)
    sd[sd == 0] = 1.0
    morph = (morph - mu) / sd            # z-score each descriptor
    return centroids, types, mpp, morph


def grid_tiles(centroids, tile_px):
    """Yield (x0, y0) lower corners of a regular grid covering all cells."""
    mins = centroids.min(axis=0)
    maxs = centroids.max(axis=0)
    xs = np.arange(mins[0], maxs[0] + tile_px, tile_px)
    ys = np.arange(mins[1], maxs[1] + tile_px, tile_px)
    for x0 in xs:
        for y0 in ys:
            yield float(x0), float(y0)


def cells_in_region(centroids, types, x0, y0, tile_px):
    """Boolean-mask the cells whose centroid falls in [x0,x0+tile)x[y0,y0+tile)."""
    m = (
        (centroids[:, 0] >= x0)
        & (centroids[:, 0] < x0 + tile_px)
        & (centroids[:, 1] >= y0)
        & (centroids[:, 1] < y0 + tile_px)
    )
    return centroids[m], types[m]


def densest_region(centroids, types, tile_px):
    """Return (sub_centroids, sub_types, (x0,y0)) for the grid tile with most cells.

    Handy for a demo run so we don't accidentally pick an empty background tile.
    """
    best = None
    best_n = -1
    for x0, y0 in grid_tiles(centroids, tile_px):
        c, t = cells_in_region(centroids, types, x0, y0, tile_px)
        if len(c) > best_n:
            best_n, best = len(c), (c, t, (x0, y0))
    return best


def build_knn_graph(centroids, types, k, radius_px):
    """Build an undirected k-NN graph (capped by a distance cutoff) as PyG Data.

    Each cell connects to its <=k nearest neighbours, keeping only edges shorter
    than radius_px (this kills implausibly long links in sparse areas; with k
    doing the real limiting, the cutoff is just a safety rail). The graph is
    symmetrised: an undirected edge exists if EITHER endpoint had the other in
    its k-nearest set, so node degree can slightly exceed k.

    x        : (N, 5) one-hot cell type
    pos      : (N, 2) centroid coordinates (pixels)
    edge_index: (2, E) undirected edges (both directions stored)
    cell_type: (N,) raw type index 1..5
    """
    n = len(centroids)
    pos = torch.tensor(centroids, dtype=torch.float)

    # one-hot type features (types are 1..5 -> columns 0..4)
    x = torch.zeros((n, N_TYPES), dtype=torch.float)
    idx_oh = torch.from_numpy(types - 1).long()
    x[torch.arange(n), idx_oh] = 1.0

    # k nearest neighbours via CPU KD-tree. Query k+1 because each point's
    # nearest "neighbour" is itself (distance 0); we drop that first column.
    tree = cKDTree(centroids)
    kq = min(k + 1, n)
    if kq < 2:                                  # only itself -> no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        dist, nbr = tree.query(centroids, k=kq)
        dist = np.atleast_2d(dist)[:, 1:]       # drop self column
        nbr = np.atleast_2d(nbr)[:, 1:]
        src = np.repeat(np.arange(n), dist.shape[1])
        dst = nbr.reshape(-1)
        d = dist.reshape(-1)
        keep = d <= radius_px                   # cutoff (also drops inf padding)
        directed = np.stack([src[keep], dst[keep]], axis=0)
        both = np.concatenate([directed, directed[::-1]], axis=1)  # symmetrise
        both = np.unique(both, axis=1)                              # dedupe
        edge_index = torch.from_numpy(both).long().contiguous()

    data = Data(x=x, pos=pos, edge_index=edge_index)
    data.cell_type = torch.from_numpy(types).long()
    return data


def build_neighbourhood_hypergraph(centroids, types, k, radius_px):
    """Build a neighbourhood hypergraph as a PyG Data object.

    The exact higher-order analogue of build_knn_graph: for each cell we form
    ONE hyperedge containing that cell plus its <=k nearest neighbours within
    radius_px. So there are N hyperedges (one centred on each cell), and a cell
    belongs to its own hyperedge plus every neighbour's hyperedge that reaches
    it -> hyperedges overlap, which is the point.

    Stored in PyG's hypergraph format:
      hyperedge_index : (2, num_incidences) -- row 0 = node id, row 1 = hyperedge id
      num_hyperedges  : int (= N here)
    x / pos / cell_type are identical to the pairwise graph (same node set).
    """
    n = len(centroids)
    pos = torch.tensor(centroids, dtype=torch.float)

    x = torch.zeros((n, N_TYPES), dtype=torch.float)
    idx_oh = torch.from_numpy(types - 1).long()
    x[torch.arange(n), idx_oh] = 1.0

    # every cell is in its own hyperedge (hyperedge i is centred on cell i)
    self_nodes = np.arange(n)
    self_edges = np.arange(n)

    tree = cKDTree(centroids)
    kq = min(k + 1, n)
    if kq < 2:                                  # no neighbours -> singleton hyperedges
        node_row, edge_row = self_nodes, self_edges
    else:
        dist, nbr = tree.query(centroids, k=kq)
        dist = np.atleast_2d(dist)[:, 1:]       # drop self column
        nbr = np.atleast_2d(nbr)[:, 1:]
        keep = (dist <= radius_px).reshape(-1)  # distance cap (also drops inf padding)
        # hyperedge id for a neighbour incidence = the centre cell's index
        nbr_edges = np.repeat(np.arange(n), nbr.shape[1])[keep]
        nbr_nodes = nbr.reshape(-1)[keep]
        node_row = np.concatenate([self_nodes, nbr_nodes])
        edge_row = np.concatenate([self_edges, nbr_edges])

    hyperedge_index = torch.from_numpy(np.stack([node_row, edge_row])).long().contiguous()

    data = Data(x=x, pos=pos)
    data.hyperedge_index = hyperedge_index
    data.num_hyperedges = n
    data.cell_type = torch.from_numpy(types).long()
    return data


def _onehot(types, n):
    x = torch.zeros((n, N_TYPES), dtype=torch.float)
    x[torch.arange(n), torch.from_numpy(types - 1).long()] = 1.0
    return x


def build_delaunay_graph(centroids, types, radius_px):
    """Standard Delaunay-triangulation cell graph (order-invariant), distance-capped.

    Parameter-free alternative to k-NN: edges come from the triangulation, then
    long edges spanning tissue gaps are pruned by radius_px. Same PyG format and
    same node features as build_knn_graph, so it's a drop-in pairwise baseline.
    """
    n = len(centroids)
    pos = torch.tensor(centroids, dtype=torch.float)
    x = _onehot(types, n)

    try:
        tri = Delaunay(centroids)
        s = tri.simplices                                      # (M, 3) triangles
        e = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]], axis=0)
        e = np.unique(np.sort(e, axis=1), axis=0)              # undirected, unique
        d = np.linalg.norm(centroids[e[:, 0]] - centroids[e[:, 1]], axis=1)
        e = e[d <= radius_px]                                  # distance cap
        both = np.concatenate([e, e[:, ::-1]], axis=0)         # symmetrise
        edge_index = torch.from_numpy(both.T).long().contiguous()
    except Exception:                                          # <4 pts / degenerate
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, pos=pos, edge_index=edge_index)
    data.cell_type = torch.from_numpy(types).long()
    return data


def clique_expand(hyperedge_index):
    """Flatten a hypergraph to a pairwise graph: connect all member pairs within
    each hyperedge. This is the information-matched pairwise CONTROL (pw-clique),
    not a real-world baseline. Returns a symmetric edge_index.
    """
    node = hyperedge_index[0].numpy()
    edge = hyperedge_index[1].numpy()
    order = np.argsort(edge, kind="stable")
    node, edge = node[order], edge[order]
    _, start, counts = np.unique(edge, return_index=True, return_counts=True)

    src_list, dst_list = [], []
    for s0, c in zip(start, counts):
        if c < 2:
            continue
        members = node[s0:s0 + c]
        a = np.repeat(members, c)
        b = np.tile(members, c)
        m = a != b                                             # drop self-pairs
        src_list.append(a[m])
        dst_list.append(b[m])
    if not src_list:
        return torch.empty((2, 0), dtype=torch.long)
    ei = np.stack([np.concatenate(src_list), np.concatenate(dst_list)])
    ei = np.unique(ei, axis=1)                                 # dedup overlaps
    return torch.from_numpy(ei).long().contiguous()


def microns_to_px(radius_um, mpp):
    return radius_um / mpp


if __name__ == "__main__":
    # ---- knobs ----
    CELLS_JSON = r"\\wsl$\Ubuntu\home\iain\cellvit\test_out\TCGA-E2-A14P\cells.json"
    TILE_PX = 4000              # region size in pixels (~1 mm at 0.25 mpp)
    K = 10                      # connect each cell to its <=K nearest neighbours
    RADIUS_UM = 35.0            # distance cap: drop edges longer than this (microns)

    centroids, types, mpp = load_cells(CELLS_JSON)
    print(f"loaded {len(centroids):,} cells | mpp={mpp} | "
          f"type counts={ {TYPE_MAP[i]: int((types==i).sum()) for i in TYPE_MAP} }")

    radius_px = microns_to_px(RADIUS_UM, mpp)
    print(f"k={K} | distance cap {RADIUS_UM} um -> {radius_px:.1f} px")

    sub_c, sub_t, (x0, y0) = densest_region(centroids, types, TILE_PX)
    print(f"densest {TILE_PX}px region at ({x0:.0f},{y0:.0f}) has {len(sub_c):,} cells")

    # --- pairwise arm ---
    g = build_knn_graph(sub_c, sub_t, K, radius_px)
    nd = g.num_nodes
    print(f"\n[pairwise]   {nd:,} nodes | {g.num_edges:,} directed edges | "
          f"mean degree {g.num_edges / nd:.1f}")
    print("  ", g)

    # --- hypergraph arm (same node set) ---
    h = build_neighbourhood_hypergraph(sub_c, sub_t, K, radius_px)
    n_he = h.num_hyperedges
    n_inc = h.hyperedge_index.shape[1]
    print(f"\n[hypergraph] {h.num_nodes:,} nodes | {n_he:,} hyperedges | "
          f"{n_inc:,} incidences | mean hyperedge size {n_inc / n_he:.1f}")
    print("  ", h)