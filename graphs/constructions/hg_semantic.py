"""hg-semantic: one hyperedge per (window, cell type) -- attribute grouping.

"All inflammatory cells in this 100um window" is a hyperedge. Cells are grouped
by WHAT THEY ARE as well as where, which is the construction the project proposal
argued for and the only one here whose membership is not a distance threshold.

Cardinality spans roughly 3 to 200, an order of magnitude more spread than
hg-radius (6-15), so this is where set-size handling has something to act on.

OVERLAPPING WINDOWS. stride_frac=0.5 runs the grid at four offsets, so each cell
lands in ~4 groups instead of 1. This is not a refinement -- a strict tile is a
PARTITION, and a partition gives incidence degree 1, which means no message can
cross between groups and a second layer does nothing a first did not. Overlap
also blurs the boundary artefact (a hard grid cuts a real aggregate in half) and
makes the cover non-conformal, so it stops being recoverable from its own clique
expansion by taking connected components. stride_frac=1.0 restores the original
partition behaviour, which is worth running once to show it fails.

DO NOT evaluate this on the masked cell-type task. Membership is keyed on `types`,
so the topology encodes the label exactly -- worse leakage than any node feature.
Slide-level tasks only, where cell type is a legitimate input.

Clique expansion is quadratic in cardinality, so a 200-cell group needs ~20,000
edges against 200 incidences. See stats_table.py for the measured ratio; that
infeasibility is a reported result, not a reason to cap the construction.
"""

import numpy as np

from ..common import make_hyper, incidences_from_groups

DEFAULTS = dict(window_um=100.0, min_size=3, stride_frac=0.5)


def _groups_for_offset(rel, types, window_px, ox, oy, min_size, max_size,
                       centroids):
    gx = ((rel[:, 0] + ox) // window_px).astype(np.int64)
    gy = ((rel[:, 1] + oy) // window_px).astype(np.int64)
    keys = np.stack([gx, gy, types], axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")

    out, start = [], 0
    for c in np.bincount(inverse):
        if c >= min_size:
            m = order[start:start + c]
            if max_size is not None and len(m) > max_size:
                # keep the members nearest the group's own centroid, so a cap is
                # deterministic and spatially coherent rather than index order
                d = np.linalg.norm(centroids[m] - centroids[m].mean(0), axis=1)
                m = m[np.argsort(d)[:max_size]]
            out.append(m)
        start += c
    return out


def build(centroids, types, window_px, morph=None, min_size=3, max_size=None,
          stride_frac=0.5):
    n = len(centroids)
    rel = centroids - centroids.min(axis=0)
    n_off = max(1, int(round(1.0 / stride_frac)))
    step = window_px * stride_frac

    groups = []
    for i in range(n_off):
        for j in range(n_off):
            groups.extend(_groups_for_offset(rel, types, window_px,
                                             i * step, j * step,
                                             min_size, max_size, centroids))
    hi, nh = incidences_from_groups(groups, n)
    return make_hyper(centroids, types, hi, nh, morph)
