"""hg-radius: one hyperedge per cell containing ALL cells within radius r.

The point of this construction is VARIABLE cardinality -- dense regions produce
large hyperedges, sparse regions small ones -- where hg_knn is fixed at k+1.
That makes it the direct test of whether the Deep Sets set-size handling
contributes anything mean-pooling structurally cannot.

Radius is much smaller than the 35um cap used elsewhere: at 35um this produces
mean cardinality ~106 in dense tissue, which averages over most of the
neighbourhood and washes out the signal. 10um gives median 6 (matching k=5)
while retaining real spread (mean ~10, max ~40).
"""

import numpy as np
from scipy.spatial import cKDTree

from .common import make_hyper, incidences_from_groups

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