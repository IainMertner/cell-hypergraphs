"""Cell-graph and cell-hypergraph constructions.

    from graphs import build, ARMS
    data = build("hg-knn", centroids, types, mpp, morph=morph)

Both builders return a PyG Data with the same node set and node features; only
the topology differs. The pairwise arm carries `edge_index`, the hypergraph arm
`hyperedge_index` + `num_hyperedges`.

Arms
----
pw-knn      BASELINE, field-standard k-NN cell graph. Cardinality 2.
hg-knn      PRIMARY, {cell + its k nearest} as one hyperedge -- the direct
            higher-order analogue: same k, same neighbours, grouped not paired.
            Cardinality FIXED at k+1 = 6.
hg-radius   MECHANISM TEST, all cells within r as one hyperedge. Cardinality
            VARIES (median 6, max 15) at the same median as hg-knn.

pw-knn vs hg-knn holds k and the neighbour set fixed, so the only difference is
set-vs-pairs. But note what it CANNOT test: hg-knn's cardinality is constant, so
the sum-vs-mean mechanism -- sum preserves set-level counts that mean divides
back out -- has no set-size variation to act on. A null there does not falsify
the mechanism; it may just mean the construction cannot express it.

hg-radius is the arm where cardinality varies, so it is where a mechanistic
advantage should appear. Comparing hg-knn against hg-radius isolates cardinality
VARIANCE from cardinality SCALE, since 12.5um is chosen to match hg-knn's median.

Four further constructions (pw-delaunay, pw-clique, hg-delaunay, hg-radius,
hg-knn+semantic) were removed once it was clear they only serve a follow-up
question -- whether an advantage tracks clique-expansibility -- which is
meaningless until some arm clears the abundance control. They are recoverable
from git history (`git show ed1da7a:graphs/constructions/hg_radius.py`).

For the record, measured expansion ratios before removal (TCGA-E2-A14P, three
densest 4000px regions): pw-* 1.0x, hg-delaunay 1.0x, hg-knn 2.5x, hg-radius
3.1-3.8x, hg-knn+semantic 5.6-9.3x. Note that spectrum is compressed at the
bottom -- three of six arms sat within 1.0-2.5x -- so it supported far less of a
trend argument than the design assumed. Worth knowing before reinstating them.
"""

from . import cells, common
from .constructions import pw_knn, hg_knn, hg_radius
from .common import (N_TYPES, microns_to_px,
                     structural_stats, print_stats_table)
from .cells import load_cache, regions, region_mask, grid_tiles, zscore_morph

ARMS = ["pw-knn", "hg-knn", "hg-radius"]

# What runs unless you ask otherwise. Kept distinct from ARMS so reinstating a
# construction does not silently enlarge the default experiment.
#
# hg-radius is DELIBERATELY not a default. It exists to answer one question:
# hg-knn has FIXED cardinality k+1, so the sum-vs-mean mechanism this project
# rests on has no set-size variation to act on there. hg-radius varies (median 6,
# max 15), so it is where a mechanistic advantage should appear if it exists.
# Run it explicitly:  --arms pw-knn hg-knn hg-radius
DEFAULT_ARMS = ["pw-knn", "hg-knn"]

# default construction parameters, in microns where applicable.
# hg_radius_um=12.5 puts hg-radius at the same MEDIAN cardinality as hg-knn (6),
# so the two differ in cardinality VARIANCE rather than in scale -- which is the
# comparison that isolates the mechanism.
PARAMS = dict(k=5, radius_um=35.0, hg_radius_um=12.5, max_size=None)

# Bump whenever the NODE FEATURE encoding changes in a way that invalidates an
# existing graph cache. precompute_graphs.py stores finished feature tensors,
# not raw cells, so a feature-encoding change CANNOT be repaired at load time --
# the cache has to be rebuilt. Recording the version makes that detectable
# instead of silent, which the geometry parameters alone would not catch.
#   1 -> morphology concatenated RAW to the one-hot type columns
#   2 -> morphology z-scored per slide (graphs.cells.zscore_morph)
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
    if arm == "hg-radius":
        # its OWN radius, not the 35um cap -- see hg_radius.py for the measured
        # cardinalities at 10 / 12.5 / 35um and why 12.5 is the setting
        return hg_radius.build(centroids, types,
                               microns_to_px(p["hg_radius_um"], mpp),
                               morph, p["max_size"])
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
