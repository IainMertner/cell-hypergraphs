"""hg-radius: one hyperedge per cell containing ALL cells within radius r.

The point of this construction is VARIABLE cardinality -- dense regions produce
large hyperedges, sparse regions small ones -- where hg_knn is fixed at k+1.
That makes it the direct test of whether the Deep Sets set-size handling
contributes anything mean-pooling structurally cannot.

Radius is much smaller than the 35um cap used elsewhere: 35um averages over most
of the neighbourhood and washes the signal out. The setting is hg_radius_um=12.5
(graphs.PARAMS), chosen so median cardinality matches hg-knn's k+1=6 while
cardinality still VARIES.

Measured on TCGA-E2-A14P, the three densest 4000px regions (9.2k-10.3k cells),
mean | median | max cardinality and clique-expansion ratio:

    10.0um    4.3-5.1 |  4-5 | 10-11    2.0-2.4x
    12.5um    6.1-7.4 |  6-8 |    15    3.1-3.8x   <- the setting
    35.0um   40.4-51.1| 40-56| 83-91   21.8-29.1x
    hg-knn        6.0 |    6 |     6    2.5x       (reference, FIXED size)

WHY THIS ARM MATTERS. hg-knn's cardinality is constant at k+1, so the sum-vs-mean
mechanism -- sum preserves set-level counts that mean divides back out -- has
nothing to preserve there. This construction is the only one where cardinality
actually varies (median 6, max 15), so it is the direct test of whether Deep Sets
set-size handling contributes anything. If a hypergraph advantage exists and is
mechanistic, it should appear HERE and not in hg-knn.

The spread is real but modest -- max 15 against a median of 6 -- so read a null
here as "the effect is small at this cardinality range", not as "no effect".
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