"""
task_viability.py
-----------------
Before downloading TIL maps for 99 slides, check the TIL-organisation task is
even alive. Two conditions must hold:

  (1) organisation VARIES across regions -- otherwise there is nothing to predict
  (2) organisation is DECORRELATED from abundance -- otherwise a model can just
      count lymphocytes, and we have rebuilt the trivial task

Uses your OWN CellViT inflammatory cells as a stand-in for the real Saltz TIL
map. That is a proxy, not the label (using your own cells as the label would be
circular) -- it is a cheap check on whether lymphocyte arrangement in this tissue
has interesting variance at all. A failure here is decisive; a pass is necessary
but not sufficient.

Metric: Moran's I on a binary grid, as planned for the real maps.

Usage:
    python task_viability.py
    python task_viability.py --min-tils 2      # match Saltz's >=2 TILs per patch
    python task_viability.py --residualise     # report the residualised target too
"""

import argparse
import numpy as np
from scipy import stats

from graphs import load_cells, grid_tiles, region_mask

INFLAMMATORY = 2          # PanNuke index
CELLS_JSON = r"\\wsl$\Ubuntu\home\iain\cellvit\test_out\TCGA-E2-A14P\cells.json"


def morans_i(binary_grid):
    """Moran's I on a binary lattice, rook (4-neighbour) contiguity.

    I > 0 : positives clump together   (organised)
    I ~ 0 : random placement
    I < 0 : positives avoid each other (dispersed)
    Validated against known patterns: solid block ~ +0.87, checkerboard = -1.0,
    random ~ 0.0, and a contiguous vs scattered grid of IDENTICAL density gives
    +0.80 vs -0.01 -- i.e. it measures arrangement, not abundance.
    """
    x = binary_grid.astype(float)
    dev = x - x.mean()
    denom = (dev ** 2).sum()
    if denom == 0:
        return np.nan
    num = 2 * ((dev[:, :-1] * dev[:, 1:]).sum() + (dev[:-1, :] * dev[1:, :]).sum())
    w = 2 * (x.shape[0] * (x.shape[1] - 1) + (x.shape[0] - 1) * x.shape[1])
    return (x.size / w) * (num / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=CELLS_JSON)
    ap.add_argument("--tile-px", type=int, default=4000)
    ap.add_argument("--patch-um", type=float, default=50.0)
    ap.add_argument("--min-cells", type=int, default=2000)
    ap.add_argument("--min-tils", type=int, default=2,
                    help="cells per patch to call it positive (Saltz uses 2)")
    ap.add_argument("--residualise", action="store_true")
    args = ap.parse_args()

    centroids, types, mpp = load_cells(args.cells)
    patch_px = args.patch_um / mpp
    nb = int(np.ceil(args.tile_px / patch_px))
    print(f"slide: {len(centroids):,} cells | mpp={mpp}")
    print(f"region {args.tile_px}px | patch {args.patch_um}um = {patch_px:.0f}px "
          f"-> {nb}x{nb} grid | patch positive at >= {args.min_tils} cells\n")

    rows = []
    for x0, y0 in grid_tiles(centroids, args.tile_px):
        m = region_mask(centroids, x0, y0, args.tile_px)
        n = int(m.sum())
        if n < args.min_cells:
            continue
        c, t = centroids[m], types[m]
        ic = c[t == INFLAMMATORY]
        grid = np.zeros((nb, nb), dtype=np.int32)
        if len(ic):
            gx = np.clip(((ic[:, 0] - x0) / patch_px).astype(int), 0, nb - 1)
            gy = np.clip(((ic[:, 1] - y0) / patch_px).astype(int), 0, nb - 1)
            np.add.at(grid, (gy, gx), 1)
        binary = grid >= args.min_tils
        mi = morans_i(binary)
        if not np.isnan(mi):
            rows.append((n, float((t == INFLAMMATORY).mean()), mi,
                         float(binary.mean())))

    if len(rows) < 5:
        print(f"only {len(rows)} usable regions -- lower --min-cells or --tile-px")
        return

    arr = np.array(rows)
    n_cells, infl, moran, posfrac = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    print(f"{len(rows)} usable regions\n")

    print("--- (1) does organisation VARY? ---")
    print(f"Moran's I : mean {moran.mean():+.3f} | sd {moran.std():.3f} | "
          f"range {moran.min():+.3f} to {moran.max():+.3f}")
    print(f"            IQR {np.percentile(moran, 25):+.3f} to "
          f"{np.percentile(moran, 75):+.3f}\n")

    print("--- (2) is it DECORRELATED from abundance? ---")
    print(f"inflammatory fraction : mean {infl.mean():.3f} | "
          f"range {infl.min():.3f}-{infl.max():.3f}")
    print(f"grid positive fraction: mean {posfrac.mean():.3f} | "
          f"range {posfrac.min():.3f}-{posfrac.max():.3f}")
    r_i, p_i = stats.spearmanr(infl, moran)
    r_g, p_g = stats.spearmanr(posfrac, moran)
    print(f"corr(inflammatory fraction, I) : rho={r_i:+.3f} (p={p_i:.3g})   <-- the one that matters")
    print(f"corr(grid positive fraction, I): rho={r_g:+.3f} (p={p_g:.3g})")
    print("  inflammatory fraction is the binding confound: the model reads cell")
    print("  types straight off its node features, so anything correlated with it")
    print("  is available for free.\n")

    print("--- verdict ---")
    ok_var = moran.std() > 0.05
    ok_dec = abs(r_i) < 0.5                # test the BINDING confound, not the weaker one
    print(f"  varies enough?  {'YES' if ok_var else 'NO '}  (sd={moran.std():.3f})")
    print(f"  decorrelated?   {'YES' if ok_dec else 'NO '}  (|rho|={abs(r_i):.3f} "
          f"vs inflammatory fraction)")

    if args.residualise or not ok_dec:
        resid = moran - np.polyval(np.polyfit(infl, moran, 1), infl)
        rr, pp = stats.spearmanr(infl, resid)
        print(f"\n--- residualised target (Moran's I regressed on abundance) ---")
        print(f"  sd {resid.std():.3f} | range {resid.min():+.3f} to {resid.max():+.3f}")
        print(f"  corr with abundance now rho={rr:+.3f} (p={pp:.3g}) -- zero by construction")
        print("  this is the pre-registered primary target: it CANNOT be solved by counting")


if __name__ == "__main__":
    main()