"""Loading CellViT output and carving slides into regions.

Not a graph construction -- this is the data layer everything else sits on.

Reads only the compact .npz written by segmentation/cache_cells.py. Parsing raw
cells.json directly used to live here too, but nothing needs it any more: the
cluster pipeline caches once and every downstream stage reads the cache.
"""

import numpy as np


def zscore_morph(morph):
    """Per-slide z-score of the morphology block.

    MUST be applied on every path that feeds morphology to a model. Raw
    poly_features are in slide pixels -- area runs to hundreds or thousands of
    px^2, perimeter to tens -- while the one-hot type columns they are
    concatenated with are 0/1. Left raw, those two columns dominate the first
    linear layer of every arm and the type signal is swamped.

    Per-slide (not global) because mpp and staining vary between slides, and
    every graph is built within one slide, so a slide-local scale is the
    comparable one.
    """
    morph = np.asarray(morph, dtype=np.float64)
    mu, sd = morph.mean(0), morph.std(0)
    sd[sd == 0] = 1.0
    return (morph - mu) / sd


def load_cache(path):
    """Load the compact .npz written by cache_cells.py (cluster pipeline).

    cache_cells.py stores morphology RAW -- area in px^2 runs to the hundreds
    while the one-hot type columns it is concatenated with are 0/1 -- so it is
    z-scored here. Without this the morphology columns dominate the first linear
    layer of every arm and the type signal is swamped.
    """
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