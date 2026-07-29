#!/bin/bash -l
# =============================================================================
# STAGE 2: train the MIL pattern classifier off the precomputed graph cache.
# GPU job -- message passing over ~200k cells/slide is what GPUs are for; on CPU
# this took ~14 CPU-hours, on GPU it should be minutes.
#
# SUBMIT:  qsub run_patterns.sh
# =============================================================================

#$ -N patterns
#$ -l h_rt=6:0:0
#$ -l mem=16G
#$ -l gpu=1
#$ -pe smp 4
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/patterns.out
#$ -e /home/ucabim3/Scratch/logs/patterns.err

# Source from the repo, not a copy in $HOME, and abort if it is not there.
ENV_SH=/home/ucabim3/Scratch/cell-hypergraphs/segmentation/cellvit_env.sh
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH" >&2; exit 1; }
source "$ENV_SH"
python -c "import torch" 2>/dev/null || {
    echo "FATAL: torch not importable after sourcing $ENV_SH" >&2; exit 1; }
mkdir -p /home/ucabim3/Scratch/logs
echo "=== $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

python -u train_patterns.py \
    --graph-cache /home/ucabim3/Scratch/graph_cache \
    --labels /home/ucabim3/Scratch/til_indices.csv \
    --task pattern4 \
    --seeds 10

echo "=== done: $(date) ==="
