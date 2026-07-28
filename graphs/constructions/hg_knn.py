"""hg-knn: k-nearest-neighbour hypergraph. The primary construction.

Direct higher-order analogue of pw_knn: for each cell, ONE hyperedge containing
{cell + its k nearest neighbours within the cap}. N cells -> N hyperedges, which
overlap heavily (a cell belongs to its own hyperedge plus every neighbour's that
reaches it).

Fixed cardinality (k+1), which is what hg_radius varies.
"""

import numpy as np
from scipy.spatial import cKDTree

from .common import make_hyper, incidences_from_groups

DEFAULTS = dict(k=5, radius_um=35.0)


def build(centroids, types, k, radius_px, morph=None):
    n = len(centroids)
    kq = min(k + 1, n)
    if kq < 2:
        hi, nh = incidences_from_groups([], n)
        return make_hyper(centroids, types, hi, nh, morph)
    tree = cKDTree(centroids)
    dist, nbr = tree.query(centroids, k=kq)
    dist, nbr = np.atleast_2d(dist)[:, 1:], np.atleast_2d(nbr)[:, 1:]
    groups = [np.concatenate([[i], nbr[i][dist[i] <= radius_px]])
              for i in range(n)]
    hi, nh = incidences_from_groups(groups, n)
    return make_hyper(centroids, types, hi, nh, morph)