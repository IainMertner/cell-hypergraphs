#!/bin/bash -l
#$ -N fig
#$ -l h_rt=0:20:0
#$ -l mem=4G
#$ -wd /home/ucabim3/Scratch/cell-hypergraphs
#$ -o /home/ucabim3/Scratch/logs/fig.$JOB_ID.out
#$ -e /home/ucabim3/Scratch/logs/fig.$JOB_ID.err
source /home/ucabim3/Scratch/cell-hypergraphs/env.sh
R=/home/ucabim3/Scratch/results
python scripts/fig_results.py --out /home/ucabim3/Scratch/figs --format pdf \
  --task "TIL arrangement|$R/tilC|abundance-only|pw-radius@gin|hg-radius@deepsets2" \
  --task "TIL arr. (brisk)|$R/tilbr|abundance-only|pw-radius@gin|hg-radius@deepsets2" \
  --task "PAM50 luminal A|$R/lumaClr|abundance-only|pw-radius@gin|hg-radius@deepsets2" \
  --task "PAM50 4-class|$R/pam4|abundance-only|pw-radius@gin|hg-radius@deepsets2"
