"""Run CellViT with GDAL's DLLs loaded first. Windows only problem.

torchvision's wheels bundle their own libjpeg/libpng, and GDAL needs those too.
On Windows the first library to load a DLL wins for the life of the process, and
CellViT imports torchvision (via cellvit.inference.inference) before pathopatch,
which imports rasterio. So rasterio always loses, and every slide dies with

    ImportError: DLL load failed while importing _base:
    The operating system cannot run %1.

despite `import rasterio` working perfectly on its own. Importing rasterio here,
before anything else, reverses the order: GDAL binds its DLLs first and
torchvision then loads its own copies without conflict.

Linux resolves shared objects differently and does not need this, so the bash
pipeline still calls cellvit-inference directly.

Arguments are passed through untouched:

    python cellvit_launch.py --model SAM --nuclei_taxonomy pannuke ... process_wsi --wsi_path X.svs
"""

import rasterio.features  # noqa: F401  -- MUST be the first import

from cellvit.detect_cells import main

if __name__ == "__main__":
    main()
