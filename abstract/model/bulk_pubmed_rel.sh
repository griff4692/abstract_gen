#!/bin/bash
set -e

MAX_VAL_EXAMPLES=512
SHARED="-contrast --max_val_examples ${MAX_VAL_EXAMPLES} --contrast_ckpt primera_ft_pubmed -use_mixed_methods --max_num_rank 4 --max_num_positive 2 --max_num_negative 2 --reference_status remove --positive_methods reference --negative_methods none --contrast_objective margin_rank --max_target_length 512 --contrast_metrics relevance --gradient_accumulation_steps 8 --dataset pubmed --hf_model primera --validate_every_n_steps 1000 --exit_after_n_steps 1000 --max_train_steps 10000 --mle_weight 0.1 --contrast_weight 1.0 --margin_scale 0.1 --length_penalty 2.0 -save_every_time"


echo "Random"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_random_rel --contrast_intra_sample_strategy random

echo "Extreme Metric"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_extreme_rel --contrast_intra_sample_strategy max_margin

echo "Average Metric"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_avg_metric_rel --contrast_intra_sample_strategy min_margin

echo "Min Metric"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_min_metric_rel --contrast_intra_sample_strategy min_metric

echo "Max Metric"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_max_metric_rel --contrast_intra_sample_strategy max_metric

echo "Max Margin"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_max_margin_rel --contrast_intra_sample_strategy max_gap

echo "Min Margin"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_min_margin_rel --contrast_intra_sample_strategy min_gap

echo "Max Diversity"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_max_diversity_rel --contrast_intra_sample_strategy max_diversity

echo "Min Diversity"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_min_diversity_rel --contrast_intra_sample_strategy min_diversity

echo "Top Beam"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_top_beam --contrast_intra_sample_strategy top_beam

echo "Wide Beam"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_wide_beam --contrast_intra_sample_strategy wide_beam

echo "Bottom Beam"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_bottom_beam --contrast_intra_sample_strategy bottom_beam

echo "Max Length"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_max_length_rel --contrast_intra_sample_strategy max_length

echo "Min Length"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_min_length_rel --contrast_intra_sample_strategy min_length

echo "Max Surprise"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_max_surprise_rel --contrast_intra_sample_strategy max_surprise

echo "Min Surprise"
CUDA_VISIBLE_DEVICES=$1 python run.py $SHARED --experiment pubmed_min_surprise --contrast_intra_sample_strategy min_surprise
