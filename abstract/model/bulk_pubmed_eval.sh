#!/bin/bash
set -e

EXPS="pubmed_min_metric_rel pubmed_max_metric_rel pubmed_max_margin_rel pubmed_min_margin_rel pubmed_max_diversity_rel"

for EXP in $EXPS
do
  python inference.py --experiment $EXP --dataset pubmed --split test --max_examples 16 --device $1
  cd ../eval
  FN="$HOME/weights/$EXP/ckpt_1000_steps/test_predictions.csv"
  CUDA_VISIBLE_DEVICES=$1 bash run_all.sh pubmed $FN all
  cd ../model
done
