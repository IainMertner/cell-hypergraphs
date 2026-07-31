#!/bin/bash -l
# Download TCGA-BRCA diagnostic slides from GDC using a manifest.
#
#     bash next_batch.sh 200 && qsub segmentation/download_slides.sh
#
# No gpu request: a download would only sit in a slower queue for it.

#$ -N gdc_download
#$ -l h_rt=12:0:0
#$ -l mem=4G                 # I/O bound, not RAM hungry
#$ -pe smp 4                 # gdc-client parallelises across files
#$ -wd /home/ucabim3/Scratch
#$ -o /home/ucabim3/Scratch/logs/gdc_download.out
#$ -e /home/ucabim3/Scratch/logs/gdc_download.err

set -euo pipefail

# defaults to whatever next_batch.sh last wrote; override with
#     qsub -v MANIFEST=/path/to/other.txt segmentation/download_slides.sh
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

# -n 4 matches the cores requested; GDC connections drop, so retry
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
