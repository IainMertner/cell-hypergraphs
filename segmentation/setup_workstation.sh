#!/bin/bash
# Rebuild the segmentation environment on a CS GPU workstation from scratch.
#
# /scratch0 is wiped when a reservation ends, so this runs once per booking.
# Roughly 30-40 minutes, nearly all of it downloads.
#
#     curl -sL https://raw.githubusercontent.com/IainMertner/cell-hypergraphs/main/segmentation/setup_workstation.sh | bash
#
# or, if the repo is already cloned:
#     bash cell-hypergraphs/segmentation/setup_workstation.sh
#
# Idempotent -- every step is skipped if already done, so re-run it after an
# interruption rather than starting over.
#
# Afterwards:
#     source /scratch0/<you>/env.sh
#     tmux new -s seg
#     bash /scratch0/<you>/cell-hypergraphs/segmentation/workstation_segment.sh batch.tsv

set -uo pipefail

# The CS session cannot resolve its own uid, so whoami/id -un and $USER all fail.
# $HOME is set and is named after the account, which is the only reliable source.
ME=$(basename "$HOME")
[ -n "$ME" ] && [ "$ME" != "/" ] || {
    echo "FATAL: could not derive a username from HOME=$HOME" >&2; exit 1; }

# /scratch0 is the CS workstations' local disk. Anywhere else -- a WSL2 box,
# a lab machine -- set ROOT to somewhere writable that survives a reboot.
ROOT="${ROOT:-/scratch0/$ME}"
CONDA="$ROOT/miniconda3"
ENVDIR="$ROOT/envs/cellvit"
REPO="$ROOT/cell-hypergraphs"
GITURL="${GITURL:-https://github.com/IainMertner/cell-hypergraphs.git}"

step() { echo; echo "=== $* ==="; }

step "user $ME | root $ROOT"
mkdir -p "$ROOT" || { echo "FATAL: cannot write to $ROOT" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || echo "WARNING: no GPU visible"
df -h "$ROOT" | tail -1

step "miniconda"
if [ -x "$CONDA/bin/conda" ]; then
    echo "already installed"
else
    cd "$ROOT"
    curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA"
    rm -f Miniconda3-latest-Linux-x86_64.sh
fi
source "$CONDA/etc/profile.d/conda.sh"

step "python 3.10 env"
# conda-forge only, with --override-channels: avoids the Anaconda channel Terms
# of Service prompt, which is non-interactive-fatal, and gives one provenance
# for every package instead of a mix.
if [ -d "$ENVDIR" ]; then
    echo "already exists"
else
    conda create -y -p "$ENVDIR" -c conda-forge --override-channels python=3.10
fi
conda activate "$ENVDIR"

step "rasterio / gdal (conda, BEFORE pip)"
# rasterio needs GDAL headers to build from source and pip fails on it. Must be
# installed before the pip packages so pip's shapely wins on the import path.
if python -c "import rasterio" 2>/dev/null; then
    echo "already present"
else
    conda install -y -c conda-forge --override-channels rasterio libjpeg-turbo gdal
fi

step "pytorch (cu121)"
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "already present, CUDA available"
else
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi
python -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available())"

step "cellvit and pins"
if python -c "import cellvit" 2>/dev/null; then
    echo "already present"
else
    pip install -q cellvit
    # Installed together so pip resolves them jointly. Separately, shapely drags
    # numpy up to 2.x and breaks cellvit, pathopatch, opentile and wsidicom.
    pip install -q "numpy<2.0" "pathopatch>=1.0.9" "pydantic>=1.10.16,<2.0"
    pip install -q openslide-bin
fi

step "verify"
python - <<'EOF'
import sys
import numpy, shapely, rasterio, torch, openslide, cellvit
ok = True
print(f"  numpy    {numpy.__version__}")
print(f"  shapely  {shapely.__version__}")
print(f"  torch    {torch.__version__}  cuda={torch.cuda.is_available()}")
if numpy.__version__.startswith("2."):
    print("  FAIL: numpy 2.x breaks cellvit/pathopatch/opentile"); ok = False
if not shapely.__version__.startswith("2."):
    # 1.8 sends CellViT down _remove_overlap_shapely_1_8, which crashes at the
    # LAST step of a slide -- after all the GPU work is done.
    print("  FAIL: shapely must be 2.x -- pip install --no-deps "
          "--force-reinstall 'shapely==2.0.5'"); ok = False
if not torch.cuda.is_available():
    print("  FAIL: no CUDA"); ok = False
sys.exit(0 if ok else 1)
EOF
VERIFY=$?


step "rclone"
# The CS session cannot resolve its own uid, so ssh/scp/rsync abort before they
# even dial. rclone speaks SFTP itself and reads $HOME from the environment, so
# it is the only thing here that can reach Myriad. /scratch0 is wiped between
# reservations, hence refetching it each time.
if [ -x "$ROOT/rclone/rclone" ]; then
    echo "already installed"
else
    cd "$ROOT"
    curl -fsSLO https://downloads.rclone.org/rclone-current-linux-amd64.zip
    python -c "import zipfile; zipfile.ZipFile('rclone-current-linux-amd64.zip').extractall('.')"
    mkdir -p "$ROOT/rclone"
    cp rclone-v*-linux-amd64/rclone "$ROOT/rclone/"
    chmod +x "$ROOT/rclone/rclone"
    rm -rf rclone-v*-linux-amd64 rclone-current-linux-amd64.zip
fi
"$ROOT/rclone/rclone" version | head -1

step "repo"
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only || echo "  (pull failed, keeping what is there)"
else
    git clone "$GITURL" "$REPO"
fi

step "env.sh"
cat > "$ROOT/env.sh" <<EOF
export USER=$ME
source $CONDA/etc/profile.d/conda.sh
conda activate $ENVDIR
export PATH="$ROOT/rclone:\$PATH"
export RCLONE_SFTP_HOST=myriad.rc.ucl.ac.uk
export RCLONE_SFTP_USER=ucabim3
export RCLONE_SFTP_ASK_PASSWORD=true
export RCLONE_SFTP_KEY_USE_AGENT=false
export RCLONE_SFTP_KNOWN_HOSTS_FILE=none
EOF
echo "wrote $ROOT/env.sh"

echo
if [ "$VERIFY" -eq 0 ]; then
    echo "=== ready ==="
    echo "  source $ROOT/env.sh"
    echo "  tmux new -s seg"
    echo "  bash $REPO/segmentation/workstation_segment.sh <batch.tsv>"
    echo
    echo "Model weights (~1-2GB) download to \$HOME/.cache on the first slide."
    echo "HOME persists across reservations, so that cost is paid once."
else
    echo "=== SETUP INCOMPLETE -- see FAIL lines above ===" >&2
    exit 1
fi
