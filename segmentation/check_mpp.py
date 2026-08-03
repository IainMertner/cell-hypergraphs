#!/usr/bin/env python
"""Print a slide's microns-per-pixel, or exit 1 if it cannot be determined.

Some TCGA slides have a truncated Aperio header carrying neither MPP nor
magnification. CellViT then refuses to run ("MPP must be defined..."), and
supplying a guess would be worse than skipping: 20x and 40x differ by a factor
of two, so a wrong value silently rescales every micron-denominated construction
parameter and produces graphs that are not comparable with the rest of the
cohort.

Every source below is real metadata. Nothing is inferred from convention.

    python check_mpp.py slide.svs        # prints e.g. 0.2500, or exits 1
"""

import sys

import openslide

# 1 inch = 25400 um, 1 cm = 10000 um
UNIT_UM = {"inch": 25400.0, "centimeter": 10000.0, "cm": 10000.0}


def mpp_of(path):
    """(mpp, source) or (None, reason)."""
    s = openslide.OpenSlide(path)
    p = s.properties

    v = p.get(openslide.PROPERTY_NAME_MPP_X)
    if v:
        return float(v), "openslide.mpp-x"

    v = p.get("aperio.MPP")
    if v:
        return float(v), "aperio.MPP"

    # TIFF resolution is pixels per unit, so um/px is unit-length / resolution
    xres, unit = p.get("tiff.XResolution"), (p.get("tiff.ResolutionUnit") or "").lower()
    if xres and unit in UNIT_UM:
        try:
            x = float(xres)
            if x > 0:
                return UNIT_UM[unit] / x, f"tiff.XResolution ({unit})"
        except ValueError:
            pass

    return None, "no openslide.mpp-x, aperio.MPP or usable tiff.XResolution"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    mpp, why = mpp_of(sys.argv[1])
    if mpp is None:
        print(f"NO MPP: {why}", file=sys.stderr)
        return 1
    # sanity: brightfield WSIs run roughly 0.1-1.0 um/px. Outside that the
    # metadata is more likely wrong than the slide exotic.
    if not 0.05 <= mpp <= 2.0:
        print(f"IMPLAUSIBLE MPP {mpp} from {why}", file=sys.stderr)
        return 1
    print(f"{mpp:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
