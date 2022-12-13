#!/bin/bash
set -e

# Training Arguments
DEVICE=$1
GRAD_ACCUM=8

# Data arguments
DATASET=$2
SAMPLE_STRATEGY=$3
EXPERIMENT=$4

# Model Arguments
OBJECTIVE="contrast"  # unlikelihood, contrast, margin_rank
METRICS="faithful"
POS_METHODS="all"
NEG_METHODS="all"
HF_MODEL="primera"
CONTRAST_CKPT="primera_ft_$DATASET"
MAX_TARGET_LENGTH=512
STEPS_PER_VALIDATION=1000
MAX_STEPS=10000

NUM_CAND=4
NUM_POS=2
NUM_NEG=2

if [[ $DATASET == "clinical" ]]
then
  MLE_WEIGHT=1.0
  CONTRAST_WEIGHT=1.0
  REF_STATUS="positive"
elif [[ $DATASET == "chemistry" ]]
then
  MLE_WEIGHT=1.0
  CONTRAST_WEIGHT=10.0
  REF_STATUS="positive"
else
  MLE_WEIGHT=1.0
  CONTRAST_WEIGHT=1.0
  REF_STATUS="ensure"
fi

PROGRAM_ARGS="--per_device_eval_batch_size 4 --mle_weight $MLE_WEIGHT --contrast_weight $CONTRAST_WEIGHT -save_every_time --reference_status $REF_STATUS -contrast --contrast_ckpt $CONTRAST_CKPT --max_num_rank $NUM_CAND --max_num_positive $NUM_POS --max_num_negative $NUM_NEG --positive_methods $POS_METHODS --negative_methods $NEG_METHODS --contrast_objective ${OBJECTIVE} --max_target_length $MAX_TARGET_LENGTH --contrast_metrics $METRICS --gradient_accumulation_steps $GRAD_ACCUM --dataset $DATASET --hf_model $HF_MODEL --validate_every_n_steps $STEPS_PER_VALIDATION --max_train_steps $MAX_STEPS"

REST="--experiment ${EXPERIMENT} --contrast_intra_sample_strategy ${SAMPLE_STRATEGY}"
echo $PROGRAM_ARGS $REST
python run.py $PROGRAM_ARGS $REST