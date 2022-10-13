#!/bin/bash
set -e

#FP='/home/ga2530/data_tmp/pubmed/intrinsic_swaps.csv'
DATASET=$1  # "pubmed"
FP=$2  # '/home/ga2530/data_tmp/pubmed/mask_and_fill/span_fills.csv'
#METRICS="rouge extractive_fragments bert_score bart_score fact_score"
METRICS="rouge extractive_fragments bert_score bart_score fact_score"

echo $DATASET
echo $FP

for metric in $METRICS
do
  echo "Running ${metric}..."
  python run.py --mode evaluate --dataset $DATASET --fp $FP --metric $metric
done

echo "Merging Metrics into a single dataframe..."
python run.py --mode merge_metrics --dataset $DATASET --fp $FP -erase_after_merge
