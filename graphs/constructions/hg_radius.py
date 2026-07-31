"""hg-radius: one hyperedge per cell containing all cells within radius r.

Cardinality VARIES with local density (median 6, max 15 at 12.5um), unlike
hg_knn's fixed k+1 -- so this is the arm where sum-vs-mean aggregation has
something to act on.
"""

import numpy as np
from scipy.spatial import cKDTree

from ..common import make_hyper, incidences_from_groups


def build(centroids, types, radius_px, morph=None, max_size=None):
    n = len(centroids)
    tree = cKDTree(centroids)
    groups = []
    for i, members in enumerate(tree.query_ball_point(centroids, r=radius_px)):
        m = np.asarray(members, dtype=np.int64)
        if max_size is not None and len(m) > max_size:
            d = np.linalg.norm(centroids[m] - centroids[i], axis=1)
            m = m[np.argsort(d)[:max_size]]            # keep the nearest
        groups.append(m)
    hi, nh = incidences_from_groups(groups, n)
    return make_hyper(centroids, types, hi, nh, morph)
