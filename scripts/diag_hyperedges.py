import sys, importlib.util
import numpy as np
from scipy.spatial import cKDTree

spec = importlib.util.spec_from_file_location("fc", "scripts/fig_constructions.py")
fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)
from graphs import load_cache, microns_to_px, PARAMS
from graphs.constructions import pw_radius, hg_radius

cells, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
centroids, types, mpp, _ = load_cache(cells)
r = microns_to_px(PARAMS["hg_radius_um"], mpp)
m = fc.pick_patch(centroids, n, r, seed)
pos = centroids[m].astype(float); t = types[m]

g = pw_radius.build(pos, t, r); h = hg_radius.build(pos, t, r, None, None)
ei = g.edge_index.numpy(); hi = h.hyperedge_index.numpy()
n_he = int(hi[1].max()) + 1 if hi.size else 0
sets = [frozenset(hi[0][hi[1] == e].tolist()) for e in range(n_he)]

ball = cKDTree(pos).query_ball_point(pos, r=r)
expected = [frozenset(b) for b in ball if len(b) >= 2]

print(f"cells {len(pos)} | edges {ei.shape[1]//2} | hyperedges {n_he}")
print(f"cells with >=1 neighbour: {sum(len(b) >= 2 for b in ball)}")
print(f"distinct hyperedge SETS : {len(set(sets))}   (duplicates: {n_he - len(set(sets))})")
print(f"sizes: {sorted(len(s) for s in sets)}")
missing = [s for s in set(expected) if s not in set(sets)]
print(f"expected groups not built: {len(missing)}")
for s in missing[:5]:
    print("   missing", sorted(s))
extra = [s for s in set(sets) if s not in set(expected)]
print(f"built groups not expected: {len(extra)}")
for s in extra[:5]:
    print("   extra  ", sorted(s))
