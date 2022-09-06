EXPERIMENT="long_t5_final"
HF_MODEL='t5'
# echo "Starting Training"

# python model/run.py \
#     --experiment $EXPERIMENT \
#     --hf_model $HF_MODEL \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \

echo "Starting Inference"
python model/inference.py \
    --experiment $EXPERIMENT \
    --hf_model $HF_MODEL \
    --batch_size 16 \
    --num_beams 1 \

# echo "Generating Diverse Abstracts for Training Set"
# python corruptions/diverse_decoding.py \
#     --experiment $EXPERIMENT \
#     --hf_model $HF_MODEL \
#     --num_candidates 5 \
#     --batch_size 8 \
