"""hg-knn: one hyperedge per cell containing {cell + its k nearest}.

The direct higher-order analogue of pw_knn -- same k, same neighbours, grouped
rather than paired. Cardinality is FIXED at k+1 (fewer only where the distance
cap bites), so sum and mean aggregation differ here only by a constant.
"""

import numpy as np
from scipy.spatial import cKDTree

from ..common import make_hyper, incidences_from_groups

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
