#!/bin/bash
DATA_DIR=/home/t-gadams/data_tmp/weights
EXPERIMENT=$1  # bs_unlikelihood_3_3_all_random
INFERENCE_BATCH_SIZE=$2
FP="${DATA_DIR}/${EXPERIMENT}/results/predictions.csv"
echo "Running metrics for ${EXPERIMENT} (${FP})"

if [[ -f $FP ]]
then
    echo "Evaluating $FP..." 
else
    echo "Generating inferences and saving to $FP..."
    python model/inference.py --experiment $EXPERIMENT --batch_size $INFERENCE_BATCH_SIZE  # --hf_model t5
    echo "Evaluating $FP..."
fi

python eval/run.py --mode evaluate --fp $FP --metric bart_score
python eval/run.py --mode evaluate --fp $FP --metric fact_score
python eval/run.py --mode evaluate --fp $FP --metric extractive_fragments
python eval/run.py --mode evaluate --fp $FP --metric bert_score

echo "Merging metrics for ${EXPERIMENT} (${FP})"
python eval/run.py --mode merge_metrics --fp $FP
