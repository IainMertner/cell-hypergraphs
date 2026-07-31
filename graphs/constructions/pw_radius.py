"""pw-radius: pairwise edges between all cells within radius r.

Control for hg-radius: same neighbourhood, pairwise form. Node degree scales
with local density the way hyperedge cardinality does.
"""

import numpy as np
from scipy.spatial import cKDTree

from ..common import make_pairwise, symmetrise


def build(centroids, types, radius_px, morph=None):
    if len(centroids) < 2:
        return make_pairwise(centroids, types,
                             symmetrise(np.empty((0, 2), int)), morph)
    pairs = cKDTree(centroids).query_pairs(r=radius_px, output_type="ndarray")
    if len(pairs) == 0:
        return make_pairwise(centroids, types,
                             symmetrise(np.empty((0, 2), int)), morph)
    return make_pairwise(centroids, types, symmetrise(pairs), morph)
