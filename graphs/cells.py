"""Loading CellViT output and carving slides into regions.

Reads the compact .npz written by segmentation/cache_cells.py.
"""

import numpy as np


def zscore_morph(morph):
    """Per-slide z-score of the morphology block.

    Required on every path feeding morphology to a model: raw areas run to
    hundreds of px^2 against 0/1 one-hot type columns, and would dominate the
    first linear layer. Per-slide because mpp and staining vary between slides.
    """
    morph = np.asarray(morph, dtype=np.float64)
    mu, sd = morph.mean(0), morph.std(0)
    sd[sd == 0] = 1.0
    return (morph - mu) / sd


def load_cache(path):
    """Load the compact .npz written by cache_cells.py. Morphology is stored
    raw there, so it is z-scored on the way out."""
    d = np.load(path)
    return (d["centroids"].astype(np.float64), d["types"].astype(np.int64),
            float(d["mpp"]), zscore_morph(d["morph"]))


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