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
from scipy.spatial import cKDTree
from torch_geometric.data import Data

# PanNuke taxonomy from CellViT: type indices are 1..5
TYPE_MAP = {1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead", 5: "Epithelial"}
N_TYPES = 5


def load_cells(path):
    """Read CellViT cells.json -> centroids (N,2) float, types (N,) int in 1..5, mpp float.

    Only pulls centroid + type; drops contours/bbox to keep memory small.
    """
    with open(path) as f:
        d = json.load(f)
    mpp = float(d["wsi_metadata"]["base_mpp"])
    cells = d["cells"]
    centroids = np.fromiter(
        (coord for c in cells for coord in c["centroid"]), dtype=np.float64
    ).reshape(-1, 2)
    types = np.fromiter((c["type"] for c in cells), dtype=np.int64)
    return centroids, types, mpp


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

    data = build_knn_graph(sub_c, sub_t, K, radius_px)
    n, e = data.num_nodes, data.num_edges
    avg_deg = e / n if n else 0
    print(f"graph: {n:,} nodes | {e:,} directed edges | mean degree {avg_deg:.1f}")
    print(data)