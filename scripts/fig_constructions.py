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


def pick_window(centroids, n_target, seed):
    """A small square window holding roughly n_target cells.

    Chosen at random rather than by density: the densest window would show
    hyperedges at their largest and misrepresent the median cardinality of six
    that the text reports.
    """
    rng = np.random.default_rng(seed)
    lo, hi = centroids.min(axis=0), centroids.max(axis=0)
    density = len(centroids) / max(np.prod(hi - lo), 1.0)
    side = float(np.sqrt(n_target / max(density, 1e-12)))
    for _ in range(200):
        x0 = rng.uniform(lo[0], max(hi[0] - side, lo[0] + 1))
        y0 = rng.uniform(lo[1], max(hi[1] - side, lo[1] + 1))
        m = ((centroids[:, 0] >= x0) & (centroids[:, 0] < x0 + side) &
             (centroids[:, 1] >= y0) & (centroids[:, 1] < y0 + side))
        if n_target * 0.6 <= m.sum() <= n_target * 1.6:
            return m
    return m


def draw_nodes(ax, pos, types):
    for t in range(len(TYPE_NAMES)):
        m = types == t + 1 if types.max() > 0 else types == t
        if m.any():
            ax.scatter(pos[m, 0], pos[m, 1], s=70, zorder=3,
                       color=TYPE_COLOURS[t], edgecolor="white", linewidth=0.8,
                       label=TYPE_NAMES[t])


def finish(ax, title):
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None, help="a cells_cache.npz")
    ap.add_argument("--n", type=int, default=28, help="cells to show")
    ap.add_argument("--radius-um", type=float, default=PARAMS["hg_radius_um"])
    ap.add_argument("--n-hyperedges", type=int, default=5,
                    help="how many of the one-per-cell hyperedges to draw. "
                         "The construction gives one per cell, which at any "
                         "useful node count is an unreadable pile of overlapping "
                         "discs; a spread-out handful shows what a hyperedge IS, "
                         "which is what the figure is for. 0 draws all of them")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figs")
    ap.add_argument("--format", default="pdf")
    args = ap.parse_args()

    if args.cells:
        centroids, types, mpp, _morph = load_cache(args.cells)
        m = pick_window(centroids, args.n, args.seed)
        pos, types = centroids[m].astype(float), types[m]
        radius_px = microns_to_px(args.radius_um, mpp)
    else:
        rng = np.random.default_rng(args.seed)
        pos = rng.uniform(0, 100, size=(args.n, 2))
        types = rng.integers(1, len(TYPE_NAMES) + 1, size=args.n)
        radius_px = 22.0
        print("no --cells given: synthetic points, schematic only")

    # the experiments' own constructions, so the figure cannot drift from them
    g = pw_radius.build(pos, types, radius_px)
    h = hg_radius.build(pos, types, radius_px, None, PARAMS["max_size"])
    ei = g.edge_index.numpy()
    hi = h.hyperedge_index.numpy()
    print(f"{len(pos)} cells | {ei.shape[1] // 2} edges | "
          f"{int(hi[1].max()) + 1} hyperedges")

    os.makedirs(args.out, exist_ok=True)
    span = pos.max(axis=0) - pos.min(axis=0)
    pad = 0.10 * span.max() + radius_px

    # ---- 3.1 pairwise
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for a, b in ei.T[ei[0] < ei[1]]:
        ax.plot(pos[[a, b], 0], pos[[a, b], 1], color="#34495e",
                linewidth=1.0, alpha=0.7, zorder=1)
    draw_nodes(ax, pos, types)
    finish(ax, "pairwise graph")
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_graph.{args.format}", bbox_inches="tight")

    # ---- 3.2 hypergraph, SAME positions and limits
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    # hg_radius builds one hyperedge per cell IN CELL ORDER, so hyperedge e is
    # exactly the disc of radius r about cell e -- drawing that disc is the
    # construction itself rather than a hull fitted to its members
    shown = [i for i in range(len(pos))
             if (hi[1] == i).sum() >= 2] if args.n_hyperedges != 1 else []
    if args.n_hyperedges:
        chosen, remaining = [shown[0]], shown[1:]
        while len(chosen) < min(args.n_hyperedges, len(shown)) and remaining:
            # farthest-point sampling, so the discs drawn overlap as little as
            # the construction allows
            d = np.min([np.linalg.norm(pos[remaining] - pos[c], axis=1)
                        for c in chosen], axis=0)
            k = int(np.argmax(d))
            chosen.append(remaining.pop(k))
        shown = chosen
    for e in shown:
        ax.add_patch(Circle(pos[e], radius_px, facecolor="#2980b9", alpha=0.13,
                            edgecolor="#2980b9", linewidth=1.0, zorder=1))
    draw_nodes(ax, pos, types)
    finish(ax, "hypergraph")
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_hypergraph.{args.format}", bbox_inches="tight")
    print(f"wrote {args.out}/fig_graph.{args.format} and "
          f"{args.out}/fig_hypergraph.{args.format}")


if __name__ == "__main__":
    main()
