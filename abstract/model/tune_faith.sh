#!/bin/bash
set -e

# Training Arguments
GRAD_ACCUM=8
MAX_STEPS=10000
TUNE_STEPS=1000
MAX_VAL_EXAMPLES=128  # Since it's faithfulness we'll need to generate and run faithfulness evaluations

# Data arguments
DATASET=$1
OBJECTIVE="contrast"

# Model Arguments
METRICS="faithful"
POS_METHODS="all"
NEG_METHODS="all"
HF_MODEL="primera"
CONTRAST_CKPT="primera_ft_$DATASET"
MAX_TARGET_LENGTH=512

NUM_POS=2
NUM_NEG=2
DEFAULT_CW=1.0
DEFAULT_MLE=1.0

PROGRAM_ARGS="--exit_after_n_steps $TUNE_STEPS -save_every_time --validate_every_n_steps $TUNE_STEPS --max_val_examples $MAX_VAL_EXAMPLES --contrast_intra_sample_strategy random --reference_status positive -contrast --contrast_ckpt $CONTRAST_CKPT --max_num_positive $NUM_POS --max_num_negative $NUM_NEG --positive_methods $POS_METHODS --negative_methods $NEG_METHODS --contrast_objective $OBJECTIVE --max_target_length $MAX_TARGET_LENGTH --contrast_metrics $METRICS --gradient_accumulation_steps $GRAD_ACCUM --dataset $DATASET --hf_model $HF_MODEL --max_train_steps $MAX_STEPS"

echo $PROGRAM_ARGS

python run.py $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight $DEFAULT_CW --experiment tune_${DATASET}_faith_baseline
python run.py $PROGRAM_ARGS --mle_weight $DEFAULT_MLE --contrast_weight 0.1 --experiment tune_${DATASET}_faith_low_cw
python run.py $PROGRAM_ARGS --mle_weight 0.1 --contrast_weight $DEFAULT_CW --experiment tune_${DATASET}_faith_low_mle
python run.py $PROGRAM_ARGS --mle_weight 0.1 --contrast_weight 0.1 --experiment tune_${DATASET}_faith_low_both
