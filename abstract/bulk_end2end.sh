#!/bin/bash

BATCH_SIZE=64

bash end2end.sh bs_src_f1_margin_rank_3_3_random_random_all $BATCH_SIZE
bash end2end.sh bs_src_f1_contrast_3_3_random_random_all $BATCH_SIZE
bash_end2end.sh bs_src_f1_unlikelihood_3_3_max_value_random_all $BATCH_SIZE
bash end2end.sh bs_src_f1_unlikelihood_1_1_random_random_all $BATCH_SIZE
bash end2end.sh bs_src_f1_unlikelihood_5_1_random_random_all $BATCH_SIZE
bash end2end.sh bs_src_f1_unlikelihood_1_5_random_random_all $BATCH_SIZE
bash end2end.sh bs_src_f1_unlikelihood_2_2_random_random_all $BATCH_SIZE

# bash end2end.sh bs_src_f1_unlikelihood_3_3_max_margin_max_margin_all $BATCH_SIZE
# bash end2end.sh bs_src_f1_unlikelihood_3_3_min_value_random_all $BATCH_SIZE
# bash end2end.sh primera_final $BATCH_SIZE
# bash end2end.sh bs_unlikelihood_3_3_all_random $BATCH_SIZE
# bash end2end.sh bs_unlikelihood_3_3_all_max_margin $BATCH_SIZE
# bash end2end.sh bs_unlikelihood_3_3_all_min_margin $BATCH_SIZE
# bash end2end.sh bs_src_f1_unlikelihood_3_3_max_margin_max_margin_all $BATCH_SIZE
# bash end2end.sh bs_src_f1_unlikelihood_3_3_max_margin_min_margin_all $BATCH_SIZE
# bash end2end.sh bs_src_f1_unlikelihood_3_3_max_margin_max_value_all $BATCH_SIZE
# bash end2end.sh bs_src_f1_unlikelihood_3_3_max_margin_min_value_all $BATCH_SIZE
# bash end2end.sh bs_src_f1_unlikelihood_3_3_max_margin_most_diverse_all $BATCH_SIZE
# bash end2end.sh bs_positive_teacher_3_3_all_random $BATCH_SIZE
# bash end2end.sh bs_src_f1_unlikelihood_1_1_max_margin_random_all $BATCH_SIZE
# bash end2end.sh bs_src_f1_positive_teacher_3_3_max_margin_max_value_all $BATCH_SIZE
