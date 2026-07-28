"""pw-clique: clique expansion of a hypergraph. A CONTROL, not a baseline.

Every hyperedge {a,b,c,...} becomes all pairwise edges among its members. This
carries the hypergraph's grouping information in pairwise form, so comparing it
against the hypergraph isolates higher-order-ness from information content.

Nobody builds cell graphs this way, which is exactly why it is a control rather
than a field baseline.

IMPORTANT: this must be regenerated from whichever hypergraph you are currently
comparing against -- never cached and reused across constructions, or it stops
being information-matched.

Clique expansion is not injective: {a,b,c} and the three separate pairs
{a,b},{b,c},{a,c} expand to the identical graph. That lost information is the
theoretical basis for expecting any hypergraph advantage at all.
"""

import numpy as np
import torch

from .common import make_pairwise


def build(centroids, types, hyperedge_index, morph=None):
    node = hyperedge_index[0].numpy()
    edge = hyperedge_index[1].numpy()
    order = np.argsort(edge, kind="stable")
    node, edge = node[order], edge[order]
    _, start, counts = np.unique(edge, return_index=True, return_counts=True)

    src, dst = [], []
    for s0, c in zip(start, counts):
        if c < 2:
            continue
        m = node[s0:s0 + c]
        a = np.repeat(m, c)
        b = np.tile(m, c)
        keep = a != b
        src.append(a[keep])
        dst.append(b[keep])
    if not src:
        ei = torch.empty((2, 0), dtype=torch.long)
    else:
        ei = np.unique(np.stack([np.concatenate(src), np.concatenate(dst)]), axis=1)
        ei = torch.from_numpy(ei).long().contiguous()
    return make_pairwise(centroids, types, ei, morph)