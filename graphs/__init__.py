"""Cell-graph and cell-hypergraph constructions.

    from graphs import build, ARMS
    data = build("hg-knn", centroids, types, mpp, morph=morph)

Both builders return a PyG Data with the same node set and node features; only
the topology differs. The pairwise arm carries `edge_index`, the hypergraph arm
`hyperedge_index` + `num_hyperedges`.

Arms
----
pw-knn   BASELINE, field-standard k-NN cell graph
hg-knn   PRIMARY, {cell + its k nearest} as one hyperedge -- the direct
         higher-order analogue: same k, same neighbours, grouped not paired

That pairing is the whole experiment. Holding k and the neighbour set fixed
means the ONLY difference between the arms is whether a neighbourhood is
represented as a set or as a collection of pairs, so a difference in performance
is attributable to that and little else.

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
from .constructions import pw_knn, hg_knn
from .common import (N_TYPES, microns_to_px,
                     structural_stats, print_stats_table)
from .cells import load_cache, regions, region_mask, grid_tiles, zscore_morph

ARMS = ["pw-knn", "hg-knn"]

# What runs unless you ask otherwise. Currently everything, but kept distinct
# from ARMS so reinstating a construction does not silently enlarge the default
# experiment -- one comparison at n~99 is already thin; a dozen uncorrected ones
# would let something clear the control by chance.
DEFAULT_ARMS = ["pw-knn", "hg-knn"]

# default construction parameters, in microns where applicable
PARAMS = dict(k=5, radius_um=35.0)

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
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
