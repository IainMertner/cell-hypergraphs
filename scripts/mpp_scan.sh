#!/bin/bash -l
# One-off: what magnifications are in the segmented cohort?
# Regions are cut in PIXELS (tile_px) while constructions are in MICRONS, so a
# 4000px region is 1mm^2 at 0.25 mpp and 4mm^2 at 0.50. This measures the spread.
#$ -N mpp_scan
#$ -l h_rt=0:10:0
#$ -l mem=4G
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/mpp_scan.$JOB_ID.out
#$ -j y

ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"

python - <<'PY'
import collections, glob, os
import numpy as np

files = sorted(glob.glob(os.path.expanduser("~/Scratch/cellvit_out/*/cells_cache.npz")))
c = collections.Counter()
for f in files:
    with np.load(f) as d:
        c[round(float(d["mpp"]), 4)] += 1

print(f"{len(files)} caches\n")
for mpp, n in sorted(c.items()):
    side = 4000 * mpp / 1000
    print(f"  mpp {mpp:<7} {n:>4} slides   region {side:.2f} mm/side, {side**2:.2f} mm^2")
PY
