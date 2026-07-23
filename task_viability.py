"""
task_viability.py
-----------------
Before downloading TIL maps for 99 slides, check the task is even alive.

The proposed task is: predict the SPATIAL ORGANISATION of lymphocytes in a
region (clustered vs dispersed), not how many there are. That only works if
two things hold:

  (1) organisation actually VARIES across regions -- if every region has the
      same arrangement there is nothing to predict.
  (2) organisation is DECORRELATED from abundance -- if the two move together,
      a model can just count lymphocytes and we have rebuilt the trivial task.

This script tests both using your OWN CellViT inflammatory cells as a stand-in
for the real TIL map. That is not the label (using your own cells as the label
would be circular) -- it is a cheap proxy to check whether lymphocyte
arrangement in this tissue has any interesting variance at all.

Metric: Moran's I on a binary grid, exactly as planned for the real TIL maps.
Each region is divided into ~50um cells (matching Saltz patch size); a grid cell
is "positive" if it contains >= 1 inflammatory cell. Moran's I then measures
whether positives clump together (I > 0) or spread out (I < 0), with I ~ 0
meaning random.

Run:  python task_viability.py
"""

import numpy as np
from scipy import stats

from build_graph import load_cells, grid_tiles

CELLS_JSON = r"\\wsl$\Ubuntu\home\iain\cellvit\test_out\TCGA-E2-A14P\cells.json"
TILE_PX = 4000          # region size, as used everywhere else
PATCH_UM = 50.0         # grid cell size in microns (matches Saltz TIL patches)
INFLAMMATORY = 2        # PanNuke index for Inflammatory
MIN_CELLS = 2000        # skip near-empty regions (unstable statistics)


def morans_i(binary_grid):
    """Moran's I for a binary lattice using rook (4-neighbour) contiguity.

    I > 0 : positives clump together   (organised / clustered)
    I ~ 0 : positives randomly placed
    I < 0 : positives avoid each other (dispersed / checkerboard)
    """
    x = binary_grid.astype(float)
    n = x.size
    xbar = x.mean()
    dev = x - xbar
    denom = (dev ** 2).sum()
    if denom == 0:                      # all-same grid -> undefined
        return np.nan

    # rook adjacency: sum of dev[i]*dev[j] over horizontally/vertically adjacent pairs
    num = 0.0
    num += (dev[:, :-1] * dev[:, 1:]).sum()      # horizontal pairs
    num += (dev[:-1, :] * dev[1:, :]).sum()      # vertical pairs
    num *= 2                                      # each pair counted both ways
    # W = total number of adjacency links (both directions)
    w = 2 * (x.shape[0] * (x.shape[1] - 1) + (x.shape[0] - 1) * x.shape[1])
    return (n / w) * (num / denom)


def region_stats(centroids, types, x0, y0, tile_px, patch_px):
    """Return (n_cells, infl_fraction, morans_I, grid_positive_fraction) for a region."""
    m = ((centroids[:, 0] >= x0) & (centroids[:, 0] < x0 + tile_px) &
         (centroids[:, 1] >= y0) & (centroids[:, 1] < y0 + tile_px))
    c, t = centroids[m], types[m]
    n = len(c)
    if n < MIN_CELLS:
        return None

    infl_frac = float((t == INFLAMMATORY).mean())

    # bin inflammatory cells onto the patch grid
    nb = int(np.ceil(tile_px / patch_px))
    ic = c[t == INFLAMMATORY]
    grid = np.zeros((nb, nb), dtype=np.int32)
    if len(ic):
        gx = np.clip(((ic[:, 0] - x0) / patch_px).astype(int), 0, nb - 1)
        gy = np.clip(((ic[:, 1] - y0) / patch_px).astype(int), 0, nb - 1)
        np.add.at(grid, (gy, gx), 1)
    binary = (grid >= 1)
    pos_frac = float(binary.mean())

    return n, infl_frac, morans_i(binary), pos_frac


def main():
    centroids, types, mpp = load_cells(CELLS_JSON)
    patch_px = PATCH_UM / mpp
    nb = int(np.ceil(TILE_PX / patch_px))
    print(f"slide: {len(centroids):,} cells | mpp={mpp}")
    print(f"region {TILE_PX}px | patch {PATCH_UM}um = {patch_px:.0f}px "
          f"-> {nb}x{nb} grid per region\n")

    rows = []
    for x0, y0 in grid_tiles(centroids, TILE_PX):
        r = region_stats(centroids, types, x0, y0, TILE_PX, patch_px)
        if r is not None and not np.isnan(r[2]):
            rows.append(r)

    if len(rows) < 5:
        print(f"only {len(rows)} usable regions - not enough to judge. "
              f"Try lowering MIN_CELLS or TILE_PX.")
        return

    arr = np.array(rows)
    n_cells, infl_frac, moran, pos_frac = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    print(f"{len(rows)} usable regions (>= {MIN_CELLS} cells)\n")
    print("--- (1) does organisation VARY across regions? ---")
    print(f"Moran's I : mean {moran.mean():+.3f} | sd {moran.std():.3f} | "
          f"range {moran.min():+.3f} to {moran.max():+.3f}")
    print(f"           IQR {np.percentile(moran,25):+.3f} to {np.percentile(moran,75):+.3f}")
    print("  (want: clear spread, not all clustered at one value)\n")

    print("--- (2) is organisation DECORRELATED from abundance? ---")
    r_p, p_p = stats.pearsonr(infl_frac, moran)
    r_s, p_s = stats.spearmanr(infl_frac, moran)
    print(f"inflammatory fraction: mean {infl_frac.mean():.3f} | "
          f"range {infl_frac.min():.3f} to {infl_frac.max():.3f}")
    print(f"corr(infl_fraction, Moran's I): pearson r={r_p:+.3f} (p={p_p:.3g}) | "
          f"spearman rho={r_s:+.3f} (p={p_s:.3g})")
    # also vs the grid-level positive fraction, the more direct confound
    r_g, p_g = stats.spearmanr(pos_frac, moran)
    print(f"corr(grid positive fraction, Moran's I): spearman rho={r_g:+.3f} (p={p_g:.3g})")
    print("  (want: |rho| well below ~0.5, else the task is really about counting)\n")

    print("--- verdict ---")
    ok_var = moran.std() > 0.05
    ok_dec = abs(r_g) < 0.5
    print(f"  varies enough?      {'YES' if ok_var else 'NO '}  (sd={moran.std():.3f})")
    print(f"  decorrelated?       {'YES' if ok_dec else 'NO '}  (|rho|={abs(r_g):.3f})")
    if ok_var and ok_dec:
        print("\n  -> task looks viable. Worth pulling the real TIL map for this slide.")
    else:
        print("\n  -> task looks weak on this slide. Rethink the metric or the target"
              "\n     before downloading 99 TIL maps.")

    print(f"grid positive fraction: mean {pos_frac.mean():.3f} | range {pos_frac.min():.3f}-{pos_frac.max():.3f}")


if __name__ == "__main__":
    main()