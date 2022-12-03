#!/bin/bash
set -e

# Training Arguments
DEVICE=$1
PORT=$2
GRAD_ACCUM=8
STAGE=2

# Data arguments
DATASET=$3
SAMPLE_STRATEGY=$4
OBJECTIVE=$5  # unlikelihood, contrast, margin_rank
EXPERIMENT=$6

# Model Arguments
METRICS="faithful"
POS_METHODS="all"
NEG_METHODS="all"
HF_MODEL="primera"
CONTRAST_CKPT="primera_ft_$DATASET"
MAX_TARGET_LENGTH=256
STEPS_PER_VALIDATION=1000
MAX_STEPS=10000

LAUNCH_CMD="accelerate launch --main_process_port=$PORT --mixed_precision=fp16 --use_deepspeed --gpu_ids=$DEVICE --num_machines=1 --gradient_accumulation_steps=$GRAD_ACCUM --zero_stage=$STAGE --offload_optimizer=cpu run.py"
# Can't use mixed precision for unlikelihood
UNLIKELIHOOD_LAUNCH="accelerate launch --main_process_port=$PORT --mixed_precision=no --use_deepspeed --gpu_ids=$DEVICE --num_machines=1 --gradient_accumulation_steps=$GRAD_ACCUM --zero_stage=$STAGE --offload_optimizer=cpu run.py"

NUM_CAND=4
if [[ $OBJECTIVE == "unlikelihood" ]]
then
  ACCELERATE_CMD=$UNLIKELIHOOD_LAUNCH
  NUM_POS=1
  NUM_NEG=1
else
  ACCELERATE_CMD=$LAUNCH_CMD
  NUM_POS=2
  NUM_NEG=2
fi

if [[ $DATASET == "clinical" ]]
then
  MLE_WEIGHT=1.0
  CONTRAST_WEIGHT=1.0
else
  MLE_WEIGHT=1.0
  CONTRAST_WEIGHT=10.0
fi

PROGRAM_ARGS="--mle_weight $MLE_WEIGHT --contrast_weight $CONTRAST_WEIGHT --reference_status positive -contrast --contrast_ckpt $CONTRAST_CKPT --max_num_rank $NUM_CAND --max_num_positive $NUM_POS --max_num_negative $NUM_NEG --positive_methods $POS_METHODS --negative_methods $NEG_METHODS --contrast_objective $OBJECTIVE --max_target_length $MAX_TARGET_LENGTH --contrast_metrics $METRICS --gradient_accumulation_steps $GRAD_ACCUM --dataset $DATASET --hf_model $HF_MODEL --validate_every_n_steps $STEPS_PER_VALIDATION --max_train_steps $MAX_STEPS"

echo $ACCELERATE_CMD
echo $PROGRAM_ARGS
$ACCELERATE_CMD $PROGRAM_ARGS --experiment $EXPERIMENT --contrast_intra_sample_strategy $SAMPLE_STRATEGY -save_every_time
