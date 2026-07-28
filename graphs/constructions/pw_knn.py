"""pw-knn: k-nearest-neighbour cell graph. The field-standard baseline.

Each cell connects to its k nearest neighbours within a distance cap, then the
graph is symmetrised (an edge exists if EITHER endpoint had the other in its
k-nearest set), so node degree can slightly exceed k.

Note this construction is NOT invariant to node ordering: ties in the k-nearest
set are broken by array position, so the same tissue can yield non-isomorphic
graphs. pw_delaunay is the order-invariant alternative.
"""

import numpy as np
from scipy.spatial import cKDTree

from .common import make_pairwise, symmetrise

DEFAULTS = dict(k=5, radius_um=35.0)


def build(centroids, types, k, radius_px, morph=None):
    n = len(centroids)
    kq = min(k + 1, n)
    if kq < 2:
        return make_pairwise(centroids, types, symmetrise(np.empty((0, 2), int)), morph)
    tree = cKDTree(centroids)
    dist, nbr = tree.query(centroids, k=kq)
    dist, nbr = np.atleast_2d(dist)[:, 1:], np.atleast_2d(nbr)[:, 1:]  # drop self
    src = np.repeat(np.arange(n), dist.shape[1])
    keep = dist.reshape(-1) <= radius_px
    pairs = np.stack([src[keep], nbr.reshape(-1)[keep]], axis=1)
    return make_pairwise(centroids, types, symmetrise(pairs), morph)