#!/bin/bash
set -e

# Training Arguments
DEVICE=$1
PORT=$2
GRAD_ACCUM=8
STAGE=2

# Data arguments
DATASET=$3
OBJECTIVE="contrast"

# Model Arguments
METRICS="faithful"
POS_METHODS="all"
NEG_METHODS="all"
HF_MODEL="primera"
CONTRAST_CKPT="primera_ft_$DATASET"
MAX_TARGET_LENGTH=256
STEPS_PER_VALIDATION=1000
MAX_STEPS=10000
TUNE_STEPS=500

ACCELERATE_CMD="accelerate launch --main_process_port=$PORT --mixed_precision=fp16 --use_deepspeed --gpu_ids=$DEVICE --num_machines=1 --gradient_accumulation_steps=$GRAD_ACCUM --zero_stage=$STAGE --offload_optimizer=cpu run.py"
NUM_POS=2
NUM_NEG=2

PROGRAM_ARGS="--exit_after_n_steps $TUNE_STEPS -save_every_time --validate_every_n_steps $TUNE_STEPS --contrast_intra_sample_strategy random --reference_status positive -contrast --contrast_ckpt $CONTRAST_CKPT --max_num_positive $NUM_POS --max_num_negative $NUM_NEG --positive_methods $POS_METHODS --negative_methods $NEG_METHODS --contrast_objective $OBJECTIVE --max_target_length $MAX_TARGET_LENGTH --contrast_metrics $METRICS --gradient_accumulation_steps $GRAD_ACCUM --dataset $DATASET --hf_model $HF_MODEL --validate_every_n_steps $STEPS_PER_VALIDATION --max_train_steps $MAX_STEPS"

echo $ACCELERATE_CMD
echo $PROGRAM_ARGS
$ACCELERATE_CMD $PROGRAM_ARGS --mle_weight 1.0 --experiment tune_faith_medium_mle
$ACCELERATE_CMD $PROGRAM_ARGS --mle_weight 2.0 --experiment tune_faith_high_mle
$ACCELERATE_CMD $PROGRAM_ARGS --mle_weight 10.0 --experiment tune_faith_highest_mle
$ACCELERATE_CMD $PROGRAM_ARGS --mle_weight 0.1 --experiment faith_low_mle
