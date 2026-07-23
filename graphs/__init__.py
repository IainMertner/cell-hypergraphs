"""Cell-graph and cell-hypergraph constructions.

One module per construction, so arms can be mixed and matched by name:

    from graphs import build, ARMS
    data = build("hg-knn", centroids, types, mpp, morph=morph)

Every builder returns a PyG Data with the same node set and node features; only
the topology differs. Pairwise arms carry `edge_index`; hypergraph arms carry
`hyperedge_index` + `num_hyperedges`.

Arms
----
pw-knn          field baseline, k-NN graph (order-dependent)
pw-delaunay     field baseline, Delaunay graph (order-invariant)
pw-clique       CONTROL, clique expansion of a hypergraph (needs one passed in)
hg-delaunay     simplex hyperedges, cardinality 3, expansion ~1.0x
hg-knn          PRIMARY, {cell + k nearest}, fixed cardinality
hg-radius       all cells within r, VARIABLE cardinality
hg-knn+semantic hg-knn plus a (window, cell-type) semantic family

Ordered by measured clique-expansibility: hg-delaunay ~1.0x, hg-knn ~2.5x,
hg-radius ~8x, hg-knn+semantic ~8x. That spectrum is the argument -- if a
hypergraph advantage tracks it, the effect is mechanistic rather than incidental.
"""

from . import (cells, common, combine,
               pw_knn, pw_delaunay, pw_clique,
               hg_knn, hg_delaunay, hg_radius, hg_semantic)
from .common import (N_TYPES, N_MORPH, TYPE_MAP, microns_to_px,
                     structural_stats, print_stats_table)
from .cells import load_cells, load_cache, regions, region_mask, grid_tiles
from .combine import combine_families

# Stage 1 arms (see pre-registration section 6)
STAGE1 = ["pw-knn", "pw-delaunay", "hg-delaunay", "hg-knn", "hg-radius",
          "hg-knn+semantic"]
# Stage 2 arms, run only if Stage 1 is positive
STAGE2 = ["pw-clique"]
ARMS = STAGE1 + STAGE2

# default construction parameters, in microns where applicable
PARAMS = dict(k=5, radius_um=35.0, hg_radius_um=12.5, window_um=100.0,
              max_size=None, min_size=2)


def build(arm, centroids, types, mpp, morph=None, params=None, hypergraph=None):
    """Build one arm by name.

    arm         : one of ARMS
    mpp         : microns per pixel, for converting the micron parameters
    morph       : optional (N,5) morphology features, concatenated to node features
    params      : overrides for PARAMS
    hypergraph  : required only for "pw-clique", which expands an existing
                  hypergraph and must be regenerated per construction
    """
    p = dict(PARAMS)
    if params:
        p.update(params)
    cap = microns_to_px(p["radius_um"], mpp)

    if arm == "pw-knn":
        return pw_knn.build(centroids, types, p["k"], cap, morph)
    if arm == "pw-delaunay":
        return pw_delaunay.build(centroids, types, cap, morph)
    if arm == "pw-clique":
        if hypergraph is None:
            raise ValueError("pw-clique needs the hypergraph it expands "
                             "(pass hypergraph=...); never cache it across arms")
        return pw_clique.build(centroids, types, hypergraph.hyperedge_index, morph)
    if arm == "hg-knn":
        return hg_knn.build(centroids, types, p["k"], cap, morph)
    if arm == "hg-delaunay":
        return hg_delaunay.build(centroids, types, cap, morph)
    if arm == "hg-radius":
        return hg_radius.build(centroids, types,
                               microns_to_px(p["hg_radius_um"], mpp),
                               morph, p["max_size"])
    if arm == "hg-knn+semantic":
        base = hg_knn.build(centroids, types, p["k"], cap, morph)
        sem = hg_semantic.build(centroids, types,
                                microns_to_px(p["window_um"], mpp),
                                morph, p["min_size"])
        return combine_families(base, sem, centroids, types, morph)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")