"""Loading CellViT output and carving slides into regions.

Not a graph construction -- this is the data layer everything else sits on.
"""

import json
import numpy as np

from .common import N_MORPH


def _poly_features(contour):
    """Nuclear shape descriptors from a cell contour (CGC-Net style).
    Returns [area, perimeter, circularity, eccentricity, extent]."""
    c = np.asarray(contour, dtype=np.float64)
    if len(c) < 3:
        return [0.0] * N_MORPH
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


def load_cells(path, with_morphology=False):
    """cells.json -> centroids (N,2), types (N,) in 1..5, mpp [, morph (N,5)].

    Only reads centroid/type (and contour if morphology is wanted); the contours
    dominate file size so skipping them keeps memory sane.
    """
    with open(path) as f:
        d = json.load(f)
    mpp = float(d["wsi_metadata"]["base_mpp"])
    cells = d["cells"]
    centroids = np.fromiter(
        (v for c in cells for v in c["centroid"]), dtype=np.float64
    ).reshape(-1, 2)
    types = np.fromiter((c["type"] for c in cells), dtype=np.int64)
    if not with_morphology:
        return centroids, types, mpp
    morph = np.array([_poly_features(c["contour"]) for c in cells],
                     dtype=np.float64)
    mu, sd = morph.mean(0), morph.std(0)
    sd[sd == 0] = 1.0
    return centroids, types, mpp, (morph - mu) / sd


def load_cache(path):
    """Load the compact .npz written by cache_cells.py (cluster pipeline)."""
    d = np.load(path)
    return (d["centroids"].astype(np.float64), d["types"].astype(np.int64),
            float(d["mpp"]), d["morph"].astype(np.float64))


def grid_tiles(centroids, tile_px):
    """Yield (x0, y0) lower corners of a regular grid covering all cells."""
    mins, maxs = centroids.min(axis=0), centroids.max(axis=0)
    for x0 in np.arange(mins[0], maxs[0] + tile_px, tile_px):
        for y0 in np.arange(mins[1], maxs[1] + tile_px, tile_px):
            yield float(x0), float(y0)


def region_mask(centroids, x0, y0, tile_px):
    return ((centroids[:, 0] >= x0) & (centroids[:, 0] < x0 + tile_px) &
            (centroids[:, 1] >= y0) & (centroids[:, 1] < y0 + tile_px))


def regions(centroids, tile_px, min_cells=2000, top_n=None):
    """List of (mask, (x0,y0), n_cells) for regions with >= min_cells.

    Sorted densest first. top_n limits how many are returned.
    """
    out = []
    for x0, y0 in grid_tiles(centroids, tile_px):
        m = region_mask(centroids, x0, y0, tile_px)
        n = int(m.sum())
        if n >= min_cells:
            out.append((m, (x0, y0), n))
    out.sort(key=lambda r: -r[2])
    return out[:top_n] if top_n else out