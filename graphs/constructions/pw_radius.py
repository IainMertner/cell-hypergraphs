"""pw-radius: pairwise graph connecting every pair of cells within radius r.

THE CONTROL FOR hg-radius. Without it, "hg-radius beats pw-knn" confounds two
changes at once: hypergraph-vs-pairwise AND radius-vs-kNN neighbourhood. Both
could produce the observed effect, because the mechanistic story -- a radius
neighbourhood in dense tissue contains more cells, so its size reads out local
density -- applies to a pairwise radius graph too, where node DEGREE scales with
density in exactly the same way.

Completing the 2x2:

                    fixed-size          radius
    pairwise        pw-knn              pw-radius
    hypergraph      hg-knn              hg-radius

A CAVEAT ON WHAT A NULL HERE WOULD MEAN. GCNConv normalises by degree
(1/sqrt(d_i d_j)), so it divides out precisely the density signal this
construction makes available. If pw-radius lands at the floor, that is
consistent with two different explanations -- hypergraph structure matters, OR
the pairwise arm's normalisation destroys the signal -- and separating those
needs a pairwise layer with unnormalised sum aggregation. Do not read a null
here as "hypergraph structure wins" on its own.

Same radius as hg-radius (hg_radius_um, 12.5um) so the neighbourhoods are the
same cells; only the representation differs.
"""

import numpy as np
from scipy.spatial import cKDTree

from ..common import make_pairwise, symmetrise


def build(centroids, types, radius_px, morph=None):
    n = len(centroids)
    if n < 2:
        return make_pairwise(centroids, types,
                             symmetrise(np.empty((0, 2), int)), morph)
    tree = cKDTree(centroids)
    # query_pairs returns each undirected pair once, as i < j
    pairs = tree.query_pairs(r=radius_px, output_type="ndarray")
    if len(pairs) == 0:
        return make_pairwise(centroids, types,
                             symmetrise(np.empty((0, 2), int)), morph)
    return make_pairwise(centroids, types, symmetrise(pairs), morph)
