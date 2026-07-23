"""hg-delaunay: each Delaunay simplex is a hyperedge. Cardinality exactly 3.

The least "higher-order" of the constructions, and deliberately so: a 3-cell
hyperedge clique-expands to a triangle with no information lost (expansion ratio
1.0), so it anchors the low end of the clique-expansibility spectrum. If a
hypergraph advantage exists and grows along that spectrum, this construction is
where it should be smallest.

Order-invariant, parameter-free apart from the distance cap.
"""

import numpy as np
from scipy.spatial import Delaunay

from .common import make_hyper, incidences_from_groups

DEFAULTS = dict(radius_um=35.0)


def build(centroids, types, radius_px, morph=None):
    n = len(centroids)
    try:
        s = Delaunay(centroids).simplices
    except Exception:
        hi, nh = incidences_from_groups([], n)
        return make_hyper(centroids, types, hi, nh, morph)
    p = centroids[s]                                   # (T, 3, 2)
    longest = np.maximum.reduce([
        np.linalg.norm(p[:, 0] - p[:, 1], axis=1),
        np.linalg.norm(p[:, 1] - p[:, 2], axis=1),
        np.linalg.norm(p[:, 0] - p[:, 2], axis=1)])
    keep = s[longest <= radius_px]
    hi, nh = incidences_from_groups(list(keep), n)
    return make_hyper(centroids, types, hi, nh, morph)