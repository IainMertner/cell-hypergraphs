"""Figures 3.1 and 3.2: the same cells as a pairwise graph and as a hypergraph.

The two panels MUST share node positions -- the point of the pair is that a
reader sees one set of cells organised two ways, so the hypergraph panel is the
graph panel with edges replaced by the neighbourhoods they came from. Drawing
them independently loses the comparison, and with it the observation that a
radius hyperedge covers exactly what the radius edges connect.

Cells come from a real segmentation cache and the constructions from the same
code the experiments use, so the figure cannot drift from what was run.

    python scripts/fig_constructions.py \
        --cells ~/Scratch/cellvit_out/TCGA-XX-XXXX-01Z-00-DX1/cells_cache.npz \
        --out figs/

Without --cells it falls back to a synthetic point set, which is fine for a
schematic but says nothing about the real construction.
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from graphs import PARAMS, load_cache                       # noqa: E402
from graphs import microns_to_px                          # noqa: E402
from graphs.constructions import pw_radius, hg_radius       # noqa: E402

# PanNuke order, matching N_TYPES in graphs/__init__.py
TYPE_NAMES = ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]
TYPE_COLOURS = ["#c0392b", "#2980b9", "#27ae60", "#7f8c8d", "#8e44ad"]


def pick_patch(centroids, n_target, radius_px, seed, target_degree=3.3,
               tries=300):
    """A patch of n_target cells whose density matches the cohort's.

    Taking the n cells nearest a random one fixes the COUNT but not the AREA,
    so in a sparse part of a slide those n cells span hundreds of microns and
    almost none fall within the radius of each other -- a figure of isolated
    dots, which is not what the construction looks like anywhere it is used.
    A typical region holds about 6,000 cells per square millimetre, a mean
    spacing near the 12.5 um radius itself, which is why the mean degree over
    the cohort is about 3.3.

    Candidates are therefore scored on the degree they would produce and the
    closest to the cohort mean is kept. Targeting the mean rather than the
    maximum matters: the densest patch would show hyperedges at their largest
    and misrepresent the median cardinality the text reports.
    """
    rng = np.random.default_rng(seed)
    tree = cKDTree(centroids)
    k = min(n_target, len(centroids))
    best = None
    for _ in range(tries):
        i = int(rng.integers(len(centroids)))
        _d, idx = tree.query(centroids[i], k=k)
        idx = np.atleast_1d(idx)
        pts = centroids[idx]
        pairs = cKDTree(pts).query_pairs(r=radius_px)
        deg = 2 * len(pairs) / k
        score = abs(deg - target_degree)
        if best is None or score < best[0]:
            best = (score, idx, deg)
    m = np.zeros(len(centroids), dtype=bool)
    m[best[1]] = True
    print(f"patch mean degree {best[2]:.1f} (cohort ~{target_degree})")
    return m


def _in_hull(ring, q):
    """Is q inside the closed convex polygon `ring`? Sign of the cross product
    at every edge, which is enough because a rounded hull is convex."""
    d = ring[1:] - ring[:-1]
    w = q - ring[:-1]
    c = d[:, 0] * w[:, 1] - d[:, 1] * w[:, 0]
    return bool((c >= -1e-9).all() or (c <= 1e-9).all())


def draw_nodes(ax, pos, types, size=70):
    for t in range(len(TYPE_NAMES)):
        m = types == t + 1 if types.max() > 0 else types == t
        if m.any():
            ax.scatter(pos[m, 0], pos[m, 1], s=size, zorder=3,
                       color=TYPE_COLOURS[t], edgecolor="white", linewidth=0.8,
                       label=TYPE_NAMES[t])


def finish(ax, title):
    # the arm names as the writeup uses them, so figure and text agree
    ax.set_title(title, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None, help="a cells_cache.npz")
    ap.add_argument("--n-cells", "--n", type=int, default=28, dest="n",
                    help="how many CELLS the patch holds. Edges and hyperedges "
                         "follow from the construction: at the cohort's mean "
                         "degree of 3.3 expect about 1.6 edges per cell and at "
                         "most one hyperedge per cell")
    ap.add_argument("--radius-um", type=float, default=PARAMS["hg_radius_um"])
    ap.add_argument("--n-hyperedges", type=int, default=5,
                    help="how many of the one-per-cell hyperedges to draw. "
                         "The construction gives one per cell, which at any "
                         "useful node count is an unreadable pile of overlapping "
                         "discs; a spread-out handful shows what a hyperedge IS, "
                         "which is what the figure is for. 0 draws all of them")
    ap.add_argument("--pad", type=float, default=0.35,
                    help="blob margin, as a fraction of the median "
                         "nearest-neighbour distance. Too large and a group "
                         "encloses cells it does not contain")
    ap.add_argument("--node-size", type=float, default=55,
                    help="marker area. At 12.5 um the radius is about one cell "
                         "spacing, so edges are short: large markers hide them "
                         "and the graph reads as unconnected when it is not")
    ap.add_argument("--target-degree", type=float, default=3.3,
                    help="mean degree the patch should show. The cohort's value "
                         "at 12.5 um; a patch far from it is not representative")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figs")
    ap.add_argument("--format", default="pdf")
    args = ap.parse_args()

    if args.cells:
        centroids, types, mpp, _morph = load_cache(args.cells)
        radius_px = microns_to_px(args.radius_um, mpp)
        m = pick_patch(centroids, args.n, radius_px, args.seed,
                       target_degree=args.target_degree)
        pos, types = centroids[m].astype(float), types[m]
    else:
        # scale the box with n so density -- and therefore mean degree -- is
        # held at the cohort's value whatever n is. A fixed box would make the
        # preview sparser the fewer cells were asked for, which says nothing
        # about the real construction
        radius_px = 22.0
        rng = np.random.default_rng(args.seed)
        side = float(np.sqrt(args.n * np.pi * radius_px ** 2
                             / max(args.target_degree, 0.1)))
        pos = rng.uniform(0, side, size=(args.n, 2))
        types = rng.integers(1, len(TYPE_NAMES) + 1, size=args.n)
        print("no --cells given: synthetic points, schematic only")

    # the experiments' own constructions, so the figure cannot drift from them
    g = pw_radius.build(pos, types, radius_px)
    h = hg_radius.build(pos, types, radius_px, None, PARAMS["max_size"])
    ei = g.edge_index.numpy()
    hi = h.hyperedge_index.numpy()
    deg = ei.shape[1] / max(len(pos), 1)
    card = np.bincount(hi[1]) if hi.size else np.array([0])
    n_he = int(hi[1].max()) + 1 if hi.size else 0
    print(f"{len(pos)} cells | {ei.shape[1] // 2} edges | {n_he} hyperedges "
          f"| mean degree {deg:.1f} | cardinality median "
          f"{int(np.median(card))} max {card.max()}")
    if n_he > ei.shape[1] // 2:
        print("  NOTE: more hyperedges than edges, which happens below mean "
              "degree 2 -- cells pairing off give one edge and two hyperedges. "
              "The patch is sparser than the cohort.")
    if deg < 1.0:
        print("  WARNING: mean degree under 1 -- this patch is sparser than the "
              "cohort's 3.3, try another --seed")

    os.makedirs(args.out, exist_ok=True)
    span = pos.max(axis=0) - pos.min(axis=0)
    pad = 0.10 * span.max() + radius_px

    # ---- 3.1 pairwise
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for a, b in ei.T[ei[0] < ei[1]]:
        ax.plot(pos[[a, b], 0], pos[[a, b], 1], color="#34495e",
                linewidth=1.4, alpha=0.85, zorder=1)
    draw_nodes(ax, pos, types, args.node_size)
    finish(ax, "pw-radius")
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_graph.{args.format}", bbox_inches="tight")

    # ---- 3.2 hypergraph, SAME positions and limits
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    # Membership is read straight off the incidence matrix. Reconstructing
    # which cell generated which hyperedge is not possible from the cache:
    # incidences_from_groups drops singletons and renumbers what survives, so
    # hyperedge j is the j-th cell that had a neighbour, not cell j. Rebuilding
    # that mapping from a fresh radius query got it wrong by however many cells
    # had been dropped, and every group was drawn centred on the wrong cell --
    # which is how a hyperedge came to appear to contain one cell when the
    # construction guarantees at least two.
    n_he = int(hi[1].max()) + 1 if hi.size else 0
    members = [pos[hi[0][hi[1] == e]] for e in range(n_he)]
    assert all(len(m) >= 2 for m in members), "singleton hyperedge in the cache"

    shown = list(range(n_he))
    if args.n_hyperedges:
        centre = np.array([m.mean(axis=0) for m in members])
        chosen, remaining = [0], list(range(1, n_he))
        while len(chosen) < min(args.n_hyperedges, n_he) and remaining:
            # farthest-point sampling, so the discs drawn overlap as little as
            # the construction allows
            d = np.min([np.linalg.norm(centre[remaining] - centre[c], axis=1)
                        for c in chosen], axis=0)
            chosen.append(remaining.pop(int(np.argmax(d))))
        shown = chosen

    # Each hyperedge is drawn as the disc that defines it: radius r about its
    # generating cell. The generator cannot be assumed to be cell e --
    # incidences_from_groups drops singletons and renumbers, so hyperedge e is
    # the e-th cell that had a neighbour. Getting that wrong centres every disc
    # on the wrong cell, which is what made groups appear to contain one cell.
    # Recover it instead: the generator is the member whose radius ball IS the
    # member set.
    balls = [frozenset(v) for v in cKDTree(pos).query_ball_point(pos, r=radius_px)]
    generator = []
    for mem in members:
        want = frozenset(np.flatnonzero(
            (pos[:, None] == mem[None]).all(-1).any(1)).tolist())
        gen = [c for c in want if balls[c] == want]
        assert gen, "no member generates this hyperedge"
        generator.append(gen[0])

    # a disc holds every cell within r of its centre, so a non-member inside one
    # means the centre is wrong -- the exact failure this figure had before
    for e, c in enumerate(generator):
        assert balls[c] == frozenset(hi[0][hi[1] == e].tolist()),             f"hyperedge {e} does not match the ball of cell {c}"

    fa = min(0.13, 1.6 / max(len(shown), 1))
    for e in shown:
        ax.add_patch(Circle(pos[generator[e]], radius_px, facecolor="#2980b9",
                            alpha=fa, edgecolor="#2980b9", linewidth=0.9,
                            zorder=1))

    draw_nodes(ax, pos, types, args.node_size)
    finish(ax, "hg-radius")
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_hypergraph.{args.format}", bbox_inches="tight")
    print(f"wrote {args.out}/fig_graph.{args.format} and "
          f"{args.out}/fig_hypergraph.{args.format}")


if __name__ == "__main__":
    main()
