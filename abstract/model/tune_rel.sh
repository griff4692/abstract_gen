#!/bin/bash
#set -e

# Training Arguments
DEVICE=$1
PORT=$2
GRAD_ACCUM=8
STAGE=2

# Data arguments
DATASET=$3
TUNE_STEPS=500

# Fixed args
OBJECTIVE="margin_rank"
# Model Arguments
METRICS="relevance"
POS_METHODS="reference"
NEG_METHODS="none"
HF_MODEL="primera"
CONTRAST_CKPT="primera_ft_$DATASET"
MAX_TARGET_LENGTH=256
MAX_STEPS=10000
NUM_CAND=4

LAUNCH_CMD="accelerate launch --main_process_port=$PORT --mixed_precision=fp16 --use_deepspeed --gpu_ids=$DEVICE --num_machines=1 --gradient_accumulation_steps=$GRAD_ACCUM --zero_stage=$STAGE --offload_optimizer=cpu run.py"
PROGRAM_ARGS="-contrast --exit_after_n_steps $TUNE_STEPS --validate_every_n_steps $TUNE_STEPS --contrast_ckpt $CONTRAST_CKPT -use_mixed_methods --max_num_rank $NUM_CAND --reference_status remove --positive_methods $POS_METHODS --negative_methods $NEG_METHODS --contrast_objective $OBJECTIVE --max_target_length $MAX_TARGET_LENGTH --contrast_metrics $METRICS --gradient_accumulation_steps $GRAD_ACCUM --dataset $DATASET --hf_model $HF_MODEL --max_train_steps $MAX_STEPS --contrast_intra_sample_strategy random -save_every_time"

DEFAULT_MLE=0.1
DEFAULT_SCALE=0.01
DEFAULT_LP=1.0
DEFAULT_CW=1.0

echo "No contrast baseline"
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight 1.0 --contrast_weight 0.0 --margin_scale $DEFAULT_SCALE --length_penalty $DEFAULT_LP --experiment tune_no_contrast_baseline

echo "Running Default Hyperparameters"
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight $DEFAULT_CW --margin_scale $DEFAULT_SCALE --length_penalty $DEFAULT_LP --experiment tune_baseline

echo "Contrast Weight Tuning.."
# contrast weight 0.01, 0.1, 1.0 (default), 10.0
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight 0.01 --margin_scale $DEFAULT_SCALE --length_penalty $DEFAULT_LP --experiment tune_lowest_cw
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight 0.1 --margin_scale $DEFAULT_SCALE --length_penalty $DEFAULT_LP --experiment tune_low_cw
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight 10.0 --margin_scale $DEFAULT_SCALE --length_penalty $DEFAULT_LP --experiment tune_high_cw

echo "Length Penalty Tuning..."
 length penalty 0.1, 0.5, 1.0 (default), 2.0
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight $DEFAULT_CW --margin_scale $DEFAULT_SCALE --length_penalty 0.1 --experiment tune_lowest_lp
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight $DEFAULT_CW --margin_scale $DEFAULT_SCALE --length_penalty 0.5 --experiment tune_low_lp
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight $DEFAULT_CW --margin_scale $DEFAULT_SCALE --length_penalty 2.0 --experiment tune_high_lp

echo "MLE Tuning..."
# mle_weight 0.01, 0.1 (default), 0.5
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight 0.01 --contrast_weight $DEFAULT_CW --margin_scale $DEFAULT_SCALE --length_penalty $DEFAULT_LP --experiment tune_low_mle
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight 0.5 --contrast_weight $DEFAULT_CW --margin_scale $DEFAULT_SCALE --length_penalty $DEFAULT_LP --experiment tune_high_mle
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight 0.5 --contrast_weight $DEFAULT_CW --margin_scale $DEFAULT_SCALE --length_penalty $DEFAULT_LP --experiment tune_highest_mle

echo "Scale Tuning..."
# scale 0.001, 0.01 (default), 0.1, 1.0
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight $DEFAULT_CW --margin_scale 0.001 --length_penalty $DEFAULT_LP --experiment tune_low_scale
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight $DEFAULT_CW --margin_scale 0.1 --length_penalty $DEFAULT_LP --experiment tune_higher_scale
$LAUNCH_CMD $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight $DEFAULT_CW --margin_scale 1.0 --length_penalty $DEFAULT_LP --experiment tune_highest_scale
