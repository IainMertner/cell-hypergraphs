#!/bin/bash
# Flat list of slide paths for the segmentation array job to index -- gdc-client
# nests each .svs in its own UUID dir, so collect paths rather than move files.
#
#     bash segmentation/make_slide_list.sh

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
echo "    qsub -t 1-$N segmentation/cellvit_chunked.sh"
