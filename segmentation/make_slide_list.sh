#!/bin/bash
# Build a flat list of slide paths, one per line, for the array job to index.
# gdc-client nests each .svs in its own UUID directory, so we just collect paths
# rather than moving/symlinking files around.
#
# Run on the login node (it's instant):   bash make_slide_list.sh

SLIDES=/home/ucabim3/Scratch/tcga_brca_slides
OUT=/home/ucabim3/Scratch/slide_list.txt

find "$SLIDES" -name "*.svs" | sort > "$OUT"

N=$(wc -l < "$OUT")
echo "wrote $OUT with $N slides"
echo
echo "first 3:"
head -3 "$OUT"
echo
echo "submit the array with:"
echo "    qsub -t 1-$N /home/ucabim3/cellvit_array.sh"
