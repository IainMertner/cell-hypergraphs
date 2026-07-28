#!/usr/bin/env python
"""
cache_cells.py <cellvit_output_dir>

Parse a slide's cells.json ONCE and cache the only things the graph pipeline
needs -- centroids, type indices, type probabilities, and morphology features --
as a compact .npz.

Why: cells.json is hundreds of MB per slide (contours dominate). Re-parsing it
every time you build a graph is slow and memory-hungry. This reduces ~1M cells
to a few tens of MB that load in milliseconds.

Writes <dir>/cells_cache.npz containing:
    centroids  (N,2) float32  -- x, y in slide pixels
    types      (N,)  int8     -- PanNuke class 1..5
    type_prob  (N,)  float32  -- classifier confidence
    morph      (N,5) float32  -- area, perimeter, circularity, eccentricity, extent
    mpp        scalar float   -- microns per pixel (for micron<->pixel conversion)
"""

import json
import sys
from pathlib import Path

import numpy as np


def poly_features(contour):
    """Nuclear shape descriptors: area, perimeter, circularity, eccentricity, extent."""
    c = np.asarray(contour, dtype=np.float64)
    if len(c) < 3:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    x, y = c[:, 0], c[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    dd = np.diff(c, axis=0, append=c[:1])
    perim = float(np.sqrt((dd ** 2).sum(1)).sum())
    circ = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0.0
    cen = c - c.mean(0)
    ev = np.clip(np.linalg.eigvalsh(np.cov(cen.T)), 1e-9, None)
    minor, major = 2 * np.sqrt(ev[0]), 2 * np.sqrt(ev[1])
    ecc = np.sqrt(1 - (minor ** 2) / (major ** 2)) if major > 0 else 0.0
    bb = (x.max() - x.min()) * (y.max() - y.min())
    extent = area / bb if bb > 0 else 0.0
    return [area, perim, circ, ecc, extent]


def main(outdir):
    outdir = Path(outdir)
    src = outdir / "cells.json"
    if not src.exists():
        # CellViT nests output under a slide-named subdir in some versions
        found = list(outdir.rglob("cells.json"))
        if not found:
            print(f"ERROR: no cells.json under {outdir}")
            return 1
        src = found[0]

    print(f"reading {src} ({src.stat().st_size / 1e6:.0f} MB)")
    with open(src) as f:
        d = json.load(f)

    mpp = float(d["wsi_metadata"]["base_mpp"])
    cells = d["cells"]
    n = len(cells)
    print(f"{n:,} cells | mpp={mpp}")

    centroids = np.empty((n, 2), dtype=np.float32)
    types = np.empty(n, dtype=np.int8)
    tprob = np.empty(n, dtype=np.float32)
    morph = np.empty((n, 5), dtype=np.float32)

    for i, c in enumerate(cells):
        centroids[i] = c["centroid"]
        types[i] = c["type"]
        tprob[i] = c.get("type_prob", 1.0)
        morph[i] = poly_features(c["contour"])

    dst = outdir / "cells_cache.npz"
    np.savez_compressed(dst, centroids=centroids, types=types,
                        type_prob=tprob, morph=morph, mpp=mpp)
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")

    counts = {int(t): int((types == t).sum()) for t in np.unique(types)}
    print(f"type counts: {counts}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
