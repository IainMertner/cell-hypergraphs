#!/bin/bash -l
# =============================================================================
# STAGE 2: train the MIL pattern classifier off the precomputed graph cache.
# GPU job -- message passing over ~200k cells/slide is what GPUs are for; on CPU
# this took ~14 CPU-hours, on GPU it should be minutes.
#
# SUBMIT:  qsub run_patterns.sh
# =============================================================================

#$ -N patterns
#$ -l h_rt=2:0:0
#$ -l mem=16G
#$ -l gpu=1
#$ -pe smp 4
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/patterns.out
#$ -e /home/ucabim3/Scratch/logs/patterns.err

source /home/ucabim3/cellvit_env.sh
mkdir -p /home/ucabim3/Scratch/logs
echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

python -u train_patterns.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --labels /home/ucabim3/Scratch/til_indices.csv \
    --task pattern4 \
    --seeds 10

echo "=== done: $(date) ==="
