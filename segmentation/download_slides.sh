#!/bin/bash -l
# =============================================================================
# Download TCGA-BRCA diagnostic slides from GDC using a manifest.
# Submit with:  qsub download_slides.sh
# =============================================================================

# --- SGE directives (Myriad uses Grid Engine, so "#$" not "#SBATCH") ---
#$ -N gdc_download           # job name (shows in qstat)
#$ -l h_rt=12:0:0            # walltime: 12 hours (downloads are slow; be generous)
#$ -l mem=4G                 # memory per core - download is I/O bound, not RAM hungry
#$ -pe smp 4                 # 4 cores (gdc-client parallelises across files)
#$ -wd /home/ucabim3/Scratch # working directory for the job
#$ -o /home/ucabim3/Scratch/logs/gdc_download.out   # stdout log
#$ -e /home/ucabim3/Scratch/logs/gdc_download.err   # stderr log

# NOTE: no "-l gpu=1" here on purpose. This is a pure download; requesting a GPU
# would waste a scarce resource and put you in a much slower queue.

set -euo pipefail

# Defaults to whatever next_batch.sh last wrote, so the two compose:
#     bash next_batch.sh 200 && qsub download_slides.sh
# Override for a one-off:  qsub -v MANIFEST=/path/to/other.txt download_slides.sh
MANIFEST="${MANIFEST:-/home/ucabim3/manifest_batch.txt}"
DEST=/home/ucabim3/Scratch/tcga_brca_slides
CLIENT=/home/ucabim3/Scratch/gdc-client

mkdir -p "$DEST" /home/ucabim3/Scratch/logs

echo "=== job started: $(date) on $(hostname) ==="
echo "manifest : $MANIFEST  ($(($(wc -l < "$MANIFEST") - 1)) files)"
echo "dest     : $DEST"
df -h "$DEST" | tail -1
echo

cd "$DEST"

# -n 4 : 4 parallel download processes (matches the 4 cores requested)
# --retry-amount : GDC connections drop occasionally; retry rather than fail
"$CLIENT" download \
    -m "$MANIFEST" \
    -n 4 \
    --retry-amount 3 \
    --wait-time 5

echo
echo "=== download finished: $(date) ==="
echo "slides retrieved:"
find "$DEST" -name "*.svs" | wc -l
du -sh "$DEST"
