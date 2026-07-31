"""Cell-graph and cell-hypergraph constructions.

    from graphs import build, ARMS
    data = build("hg-knn", centroids, types, mpp, morph=morph)

Every builder returns a PyG Data with the same node set and node features; only
the topology differs. Pairwise arms carry `edge_index`, hypergraph arms
`hyperedge_index` + `num_hyperedges`.

    pw-knn      baseline, field-standard k-NN graph
    pw-radius   control for hg-radius: same neighbourhood, pairwise form
    hg-knn      {cell + k nearest} as one hyperedge. Cardinality FIXED at k+1
    hg-radius   all cells within r. Cardinality VARIES with density

Removed constructions (pw-delaunay, pw-clique, hg-delaunay, hg-knn+semantic)
are in git history at ed1da7a.
"""

from . import cells, common
from .constructions import pw_knn, pw_radius, hg_knn, hg_radius
from .common import (N_TYPES, microns_to_px,
                     structural_stats, print_stats_table)
from .cells import load_cache, regions, region_mask, grid_tiles, zscore_morph

ARMS = ["pw-knn", "pw-radius", "hg-knn", "hg-radius"]

# Kept distinct from ARMS so adding a construction does not silently enlarge the
# default experiment. Request others explicitly with --arms.
DEFAULT_ARMS = ["pw-knn", "hg-knn"]

# Microns where applicable. hg_radius_um=12.5 matches hg-knn's median
# cardinality, so the two differ in cardinality VARIANCE, not scale.
PARAMS = dict(k=5, radius_um=35.0, hg_radius_um=12.5, max_size=None)

# Bump when the node feature encoding changes. The cache stores finished feature
# tensors, so such a change cannot be repaired at load time -- it must be rebuilt.
#   1 -> morphology concatenated RAW    2 -> morphology z-scored per slide
FEATURE_VERSION = 2


def build(arm, centroids, types, mpp, morph=None, params=None):
    """Build one arm by name.

    arm    : one of ARMS
    mpp    : microns per pixel, for converting the micron parameters
    morph  : optional (N,5) morphology features, concatenated to node features
    params : overrides for PARAMS
    """
    p = dict(PARAMS)
    if params:
        p.update(params)
    cap = microns_to_px(p["radius_um"], mpp)

    if arm == "pw-knn":
        return pw_knn.build(centroids, types, p["k"], cap, morph)
    if arm == "hg-knn":
        return hg_knn.build(centroids, types, p["k"], cap, morph)
    if arm == "pw-radius":
        # same radius as hg-radius, so only the representation differs
        return pw_radius.build(centroids, types,
                               microns_to_px(p["hg_radius_um"], mpp), morph)
    if arm == "hg-radius":
        return hg_radius.build(centroids, types,
                               microns_to_px(p["hg_radius_um"], mpp),
                               morph, p["max_size"])
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
