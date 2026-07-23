"""pw-delaunay: Delaunay-triangulation cell graph. Order-invariant baseline.

Edges come from the triangulation, so the graph depends only on cell positions,
not on the order cells appear in the array. Long edges spanning tissue gaps are
pruned by the distance cap. Parameter-free apart from that cap; mean degree is
~6 by geometry.
"""

import numpy as np
from scipy.spatial import Delaunay

from .common import make_pairwise, symmetrise

DEFAULTS = dict(radius_um=35.0)


def build(centroids, types, radius_px, morph=None):
    try:
        s = Delaunay(centroids).simplices
    except Exception:                      # <4 points, or degenerate/collinear
        return make_pairwise(centroids, types, symmetrise(np.empty((0, 2), int)), morph)
    e = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    d = np.linalg.norm(centroids[e[:, 0]] - centroids[e[:, 1]], axis=1)
    return make_pairwise(centroids, types, symmetrise(e[d <= radius_px]), morph)