#!/bin/bash
set -e

# Training Arguments
DEVICE=$1
PORT=$2
GRAD_ACCUM=8
STAGE=2

# Data arguments
DATASET=$3
FROM_CKPT=$4  # ft or fft
OBJECTIVE=$5  # unlikelihood, contrast, margin_rank
EXPERIMENT=$6

# Model Arguments
METRICS="relevance"
POS_METHODS="reference"
NEG_METHODS="none"
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

PROGRAM_ARGS="-contrast -use_mixed_methods --max_num_rank $NUM_CAND --max_num_positive $NUM_POS --max_num_negative $NUM_NEG --reference_status remove --positive_methods $POS_METHODS --negative_methods $NEG_METHODS --contrast_objective $OBJECTIVE --max_target_length $MAX_TARGET_LENGTH --contrast_metrics $METRICS --gradient_accumulation_steps $GRAD_ACCUM --dataset $DATASET --hf_model $HF_MODEL --validate_every_n_steps $STEPS_PER_VALIDATION --max_train_steps $MAX_STEPS"

if [[ $FROM_CKPT == "fft" ]]
then
  PROGRAM_ARGS+=" --contrast_ckpt $CONTRAST_CKPT"
fi

echo $ACCELERATE_CMD
echo $PROGRAM_ARGS
$ACCELERATE_CMD $PROGRAM_ARGS --experiment $EXPERIMENT -save_every_time
