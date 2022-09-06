#!/bin/bash

python run.py -contrast --contrast_metrics rouge1,rouge2,rougeL --max_num_positive 2 --max_num_negative 2 --contrast_methods diverse_decoding_primera --contrast_objective margin_rank --contrast_weight 0.1 --experiment brio_rouge_primera
python run.py -contrast --contrast_metrics rouge1,rouge2,rougeL --max_num_positive 2 --max_num_negative 2 --contrast_methods diverse_decoding_primera --contrast_objective margin_rank --contrast_weight 0.05 --experiment brio_super_low
python run.py -contrast --contrast_metrics rouge1,rouge2,rougeL --max_num_positive 2 --max_num_negative 2 --contrast_methods diverse_decoding_primera,diverse_decoding_long_t5 --contrast_objective margin_rank --contrast_weight 0.1 --experiment brio_rouge_primera_t5_max_margin --contrast_intra_sample_strategy max_margin
python run.py -contrast --contrast_metrics rouge1,rouge2,rougeL --max_num_positive 1 --max_num_negative 3 --contrast_methods diverse_decoding_primera --contrast_objective unlikelihood --experiment rouge_unlikelihood
python run.py -contrast --contrast_metrics rouge1,rouge2,rougeL --max_num_positive 1 --max_num_negative 1 --contrast_methods diverse_decoding_primera --contrast_objective positive_teacher --max_num_positive 1 --max_num_negative 1 --contrast_intra_sample_strategy max_margin --experiment rouge_positive_teacher
python run.py -contrast --contrast_metrics rouge1,rouge2,rougeL --max_num_positive 2 --max_negative 2 --contrast_methods diverse_decoding_primera --contrast_objective contrast --experiment rouge_contrast

# GRAD_ACCUM=16
# ZERO_STAGE=3
# LAUNCH_CMD="accelerate launch --mixed_precision=fp16 --use_deepspeed --num_machines=1 --gradient_accumulation_steps=$GRAD_ACCUM --zero_stage=$ZERO_STAGE run.py"
# # Can't use mixed precision for unlikelihood
# UNLIKELIHOOD_LAUNCH="accelerate launch --mixed_precision=no --use_deepspeed --num_machines=1 --gradient_accumulation_steps=$GRAD_ACCUM --zero_stage=$ZERO_STAGE run.py"
# SHARED_ARGS="-contrast --contrast_methods all --contrast_metrics bs_src_f1 --max_num_positive 3 --max_num_negative 3 --gradient_accumulation_steps ${GRAD_ACCUM}"

# # Experiment Naming Convention
# # Metric, Contrast Objective, Number Negative, Number Positive, Methods, Sample Strategy

# $UNLIKELIHOOD_LAUNCH $SHARED_ARGS --contrast_sample_strategy random \
#     --contrast_objective unlikelihood \
#     --experiment bs_unlikelihood_3_3_all_random

# $UNLIKELIHOOD_LAUNCH $SHARED_ARGS --contrast_sample_strategy max_margin \
#     --contrast_objective unlikelihood \
#     --experiment bs_unlikelihood_3_3_all_max_margin

# $UNLIKELIHOOD_LAUNCH $SHARED_ARGS --contrast_sample_strategy min_margin \
#     --contrast_objective unlikelihood \
#     --experiment bs_unlikelihood_3_3_all_min_margin

# $LAUNCH_CMD $SHARED_ARGS --contrast_sample_strategy random \
#     --contrast_objective margin_rank \
#     --experiment bs_margin_rank_3_3_all_random

# LAUNCH_CMD $SHARED_ARGS --contrast_sample_strategy random \
#     --contrast_objective contrast \
#     --experiment bs_contrast_3_3_all_random

# LAUNCH_CMD $SHARED_ARGS --contrast_sample_strategy random \
#     --contrast_objective positive_teacher \
#     --experiment bs_positive_teacher_all_random
