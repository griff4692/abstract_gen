#!/bin/bash
set -e

EXPS="pubmed_min_metric_rel pubmed_max_metric_rel pubmed_max_margin_rel pubmed_min_margin_rel pubmed_max_diversity_rel"
CKPT="ckpt_1000_steps"

for EXP in $EXPS
do
  python inference.py --experiment $EXP --dataset pubmed --split test --max_examples 99999999 --device $1 --ckpt_name $CKPT --results_name $CKPT
  cd ../eval
  FN="$HOME/data_tmp/weights/$EXP/$CKPT/test_predictions.csv"
  CUDA_VISIBLE_DEVICES=$1 bash run_all.sh pubmed $FN all
  cd ../model
done
