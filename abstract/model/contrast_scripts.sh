#!/bin/bash
set -e

DEVICE=0
GRAD_ACCUM=16
ZERO_STAGE=3
POS=2
NEG=2
LAUNCH_CMD="accelerate launch --mixed_precision=fp16 --use_deepspeed --gpu_ids=$DEVICE --gradient_accumulation_steps=$GRAD_ACCUM --zero_stage=$ZERO_STAGE --offload_optimizer=cpu run.py"
# # Can't use mixed precision for unlikelihood
UNLIKELIHOOD_LAUNCH="accelerate launch --mixed_precision=no --use_deepspeed --num_machines=1 --gradient_accumulation_steps=$GRAD_ACCUM --zero_stage=$ZERO_STAGE run.py"
SHARED_ARGS="-contrast --contrast_methods all --contrast_metrics bs_src_f1 --max_num_positive 3 --max_num_negative 3 --gradient_accumulation_steps ${GRAD_ACCUM} -build_contrast_exp_name"

# # Experiment Naming Convention
# # Metric, Contrast Objective, Number Negative, Number Positive, Methods, Sample Strategy

$UNLIKELIHOOD_LAUNCH $SHARED_ARGS --contrast_sample_strategy random \
  --contrast_objective unlikelihood
