"""Figure for §4: an H&E crop, the cells detected in it, and the graph built on them.

Three panels over the same pixels, so a reader can see what survives each step.
The point is as much what is discarded as what is kept: the third panel holds
ten numbers per nucleus and nothing else, which is the whole input to every
model in this study.

    python scripts/fig_region.py \
        --slide ~/Scratch/tcga_brca_slides/<uuid>/<name>.svs \
        --cells ~/Scratch/cellvit_out/<slide-id>/cells_cache.npz \
        --um 250 --out figs

Needs openslide, which the segmentation environment already has. Cell contours
are not stored by the pipeline -- only centroids and types -- so cells are drawn
as points rather than outlines.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from graphs import PARAMS, load_cache, microns_to_px          # noqa: E402
from graphs.constructions import pw_radius                    # noqa: E402

TYPE_NAMES = ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]
TYPE_COLOURS = ["#c0392b", "#2980b9", "#27ae60", "#7f8c8d", "#8e44ad"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True, help="the .svs")
    ap.add_argument("--cells", required=True, help="its cells_cache.npz")
    ap.add_argument("--um", type=float, default=250.0, help="crop side, microns")
    ap.add_argument("--radius-um", type=float, default=PARAMS["hg_radius_um"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figs")
    ap.add_argument("--format", default="png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    try:
        import openslide
    except ImportError:
        raise SystemExit("openslide not importable -- run this in the "
                         "segmentation environment, which has it")

    centroids, types, mpp, _m = load_cache(args.cells)
    side = microns_to_px(args.um, mpp)

    # centre on a cell rather than on a random point: a random crop of a slide
    # is mostly fat and glass, and would show a construction with no edges
    rng = np.random.default_rng(args.seed)
    tree = cKDTree(centroids)
    for _ in range(500):
        c = centroids[rng.integers(len(centroids))]
        x0, y0 = c[0] - side / 2, c[1] - side / 2
        m = ((centroids[:, 0] >= x0) & (centroids[:, 0] < x0 + side) &
             (centroids[:, 1] >= y0) & (centroids[:, 1] < y0 + side))
        if m.sum() >= 40:
            break
    pos, t = centroids[m] - np.array([x0, y0]), types[m]
    print(f"{int(m.sum())} cells in a {args.um:.0f} um crop "
          f"({side:.0f} px at {mpp:.3f} um/px)")

    sl = openslide.OpenSlide(args.slide)
    img = np.asarray(sl.read_region((int(x0), int(y0)), 0,
                                    (int(side), int(side))).convert("RGB"))

    g = pw_radius.build(pos, t, microns_to_px(args.radius_um, mpp))
    ei = g.edge_index.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3))
    titles = ["H&E", "detected cells", "cell graph"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, side); ax.set_ylim(side, 0)      # image row order
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_color("#95a5a6")

    axes[0].imshow(img, extent=(0, side, side, 0))
    axes[1].imshow(img, extent=(0, side, side, 0), alpha=0.35)
    for k in range(len(TYPE_NAMES)):
        sel = t == k + 1
        if sel.any():
            axes[1].scatter(pos[sel, 0], pos[sel, 1], s=4, linewidth=0,
                            color=TYPE_COLOURS[k], label=TYPE_NAMES[k])
            axes[2].scatter(pos[sel, 0], pos[sel, 1], s=4, linewidth=0,
                            color=TYPE_COLOURS[k], zorder=3)
    seg = ei.T[ei[0] < ei[1]]
    axes[2].plot(np.stack([pos[seg[:, 0], 0], pos[seg[:, 1], 0]]),
                 np.stack([pos[seg[:, 0], 1], pos[seg[:, 1], 1]]),
                 color="#34495e", linewidth=0.4, alpha=0.6, zorder=1)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3,
                   frameon=False, fontsize=6, handletextpad=0.2,
                   columnspacing=0.8, markerscale=1.6)

    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/fig_region.{args.format}"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=args.dpi)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
