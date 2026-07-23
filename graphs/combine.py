"""Merging hypergraph families over a shared node set.

The multi-family design from the project proposal: carry spatial hyperedges AND
semantic hyperedges together, so a cell belongs to both a spatial neighbourhood
and a semantic group. Hyperedge ids from the second family are offset so they
stay distinct, and family_id records which family each came from.

family_id exists so you can do family-aware aggregation later. It matters
because the families are badly mismatched in cardinality (spatial ~6 cells,
semantic up to hundreds); pooling them with shared weights lets the large ones
dominate. Stage 1 ignores the tag and treats all hyperedges alike -- if that
underperforms, family-aware routing is the first thing to try.
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