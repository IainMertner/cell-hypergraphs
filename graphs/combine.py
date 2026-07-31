"""Merging hyperedge families over a shared node set.

Carry spatial hyperedges AND semantic hyperedges together, so a cell belongs to
both a spatial neighbourhood and an attribute group. Hyperedge ids from the
second family are offset so they stay distinct, and `family_id` records which
family each came from.

family_id is not bookkeeping. The families are badly mismatched in cardinality --
spatial ~6 cells, semantic up to ~200 -- and SumHyperConv aggregates members
UNNORMALISED, so pooling both with shared weights lets the semantic family
contribute ~30x the magnitude and dominate outright. Expect exploding activations
if the layer ignores the tag. Family-aware weights are the fix; a size cap is not,
since the cardinality spread is the thing under test.
"""

import torch

from .common import make_hyper


def combine_families(base, extra, centroids, types, morph=None):
    b, e = base.hyperedge_index, extra.hyperedge_index
    nb, ne = int(base.num_hyperedges), int(extra.num_hyperedges)
    if ne == 0 or e.numel() == 0:
        return make_hyper(centroids, types, b, nb, morph,
                          family_id=torch.zeros(nb, dtype=torch.long))
    e = e.clone()
    e[1] = e[1] + nb                                   # offset so ids don't collide
    hi = torch.cat([b, e], dim=1)
    fam = torch.cat([torch.zeros(nb, dtype=torch.long),
                     torch.ones(ne, dtype=torch.long)])
    return make_hyper(centroids, types, hi, nb + ne, morph, family_id=fam)
