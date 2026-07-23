"""hg-semantic: attribute-based grouping -- one hyperedge per (window, cell type).

e.g. "all inflammatory cells in this 100um window" is a hyperedge. Cells are
grouped by WHAT THEY ARE as well as where, which is the construction the project
proposal actually argued for: semantic families are natural to express as
hyperedges and awkward as pairwise edges.

WARNING -- do not use standalone. A tiled partition puts each cell in exactly
one group, so node degree is 1, hyperedges do not overlap, and message passing
cannot propagate beyond a single group. Combine it with a spatial construction
(see combine.combine_families) so it ADDS information rather than replacing the
structure.

There is no faithful clique expansion for these hyperedges: expansion is
quadratic in cardinality (which reaches the hundreds) and semantically wrong,
since it asserts every member is mutually adjacent. The honest pairwise
comparator is a star/bipartite expansion instead.
"""

import numpy as np

from .common import make_hyper, incidences_from_groups

DEFAULTS = dict(window_um=100.0, min_size=2)


def build(centroids, types, window_px, morph=None, min_size=2):
    n = len(centroids)
    origin = centroids.min(axis=0)
    gx = ((centroids[:, 0] - origin[0]) // window_px).astype(np.int64)
    gy = ((centroids[:, 1] - origin[1]) // window_px).astype(np.int64)
    keys = np.stack([gx, gy, types], axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                   return_counts=True)
    order = np.argsort(inverse, kind="stable")
    groups, start = [], 0
    for c in np.bincount(inverse):
        if c >= min_size:
            groups.append(order[start:start + c])
        start += c
    hi, nh = incidences_from_groups(groups, n)
    return make_hyper(centroids, types, hi, nh, morph)