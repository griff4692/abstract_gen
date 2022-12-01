#!/bin/bash
#set -e

# Training Arguments
DEVICE=$1
PORT=$2
GRAD_ACCUM=8
STAGE=2

# Data arguments
DATASET="chemistry"
TUNE_STEPS=500

# Fixed args
OBJECTIVE="margin_rank"
# Model Arguments
METRICS="relevance"
POS_METHODS="reference"
NEG_METHODS="none"
HF_MODEL="primera"
CONTRAST_CKPT="primera_ft_$DATASET"
MAX_TARGET_LENGTH=1024
MAX_STEPS=10000
NUM_CAND=4

DEFAULT_MLE=0.5
DEFAULT_SCALE=0.01
DEFAULT_LP=1.0
DEFAULT_CW=1.0

DEFAULT_LAUNCH_CMD="accelerate launch --main_process_port=$PORT --mixed_precision=fp16 --use_deepspeed --gpu_ids=$DEVICE --num_machines=1 --gradient_accumulation_steps=$GRAD_ACCUM --zero_stage=$STAGE --offload_optimizer=cpu run.py"
PROGRAM_ARGS="-contrast --exit_after_n_steps $TUNE_STEPS --validate_every_n_steps $TUNE_STEPS --contrast_ckpt $CONTRAST_CKPT -use_mixed_methods --max_num_rank $NUM_CAND --reference_status remove --positive_methods $POS_METHODS --negative_methods $NEG_METHODS --contrast_objective $OBJECTIVE --max_target_length $MAX_TARGET_LENGTH --contrast_metrics $METRICS --dataset $DATASET --hf_model $HF_MODEL --max_train_steps $MAX_STEPS --contrast_intra_sample_strategy random -save_every_time --mle_weight $DEFAULT_MLE --contrast_weight $DEFAULT_CW --margin_scale $DEFAULT_SCALE --length_penalty $DEFAULT_LP"

echo "Bigger Batch Size"
LAUNCH_CMD_16_BATCH="accelerate launch --main_process_port=$PORT --mixed_precision=fp16 --use_deepspeed --gpu_ids=$DEVICE --num_machines=1 --gradient_accumulation_steps=16 --zero_stage=$STAGE --offload_optimizer=cpu run.py"
$LAUNCH_CMD_16_BATCH $PROGRAM_ARGS --gradient_accumulation_steps 16 --experiment tune_chemistry_16batch

echo "Lower Learning Rate"
$DEFAULT_LAUNCH_CMD $PROGRAM_ARGS --learning_rate 5e-6 --experiment tune_chemistry_low_lr --gradient_accumulation_steps $GRAD_ACCUM

echo "Weight Decay"
$DEFAULT_LAUNCH_CMD $PROGRAM_ARGS --weight_decay 5e-5 --experiment tune_chemistry_weight_decay --gradient_accumulation_steps $GRAD_ACCUM

echo "Longer Warmup"
$DEFAULT_LAUNCH_CMD $PROGRAM_ARGS --num_warmup_steps 10000 --experiment tune_chemistry_longer_warmup --gradient_accumulation_steps $GRAD_ACCUM

echo "Baseline"
$DEFAULT_LAUNCH_CMD $PROGRAM_ARGS --experiment tune_chemistry_baseline  --gradient_accumulation_steps $GRAD_ACCUM
