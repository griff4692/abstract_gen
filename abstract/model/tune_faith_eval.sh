#!/bin/bash
set -e

DEVICE=$1
DATASET=$2
EXPS="tune_faith_baseline_cw tune_faith_baseline_mle tune_faith_high_cw tune_faith_low_cw tune_faith_lower_mle tune_faith_medium_mle"

DIR="${HOME}/data_tmp/weights"
INFERENCE_BATCH_SIZE=8
MAX_EXAMPLES=99999999
METRIC="faithful"


SPLIT="validation"
for EXP in $EXPS
do
  CKPT_NAME="last_ckpt"
  STEP_DIR="${DIR}/${EXP}/${CKPT_NAME}"

  if [ ! -d $STEP_DIR ]
  then
    echo "$STEP_DIR does not exist. Skipping."
    continue
  fi

  OUT_MODEL_FN="${STEP_DIR}/pytorch_model.bin"
  if [ ! -f $OUT_MODEL_FN ]
  then
    python "$STEP_DIR/zero_to_fp32.py" $STEP_DIR $OUT_MODEL_FN
  fi
  python inference.py --hf_model primera --device $DEVICE --experiment $EXP --dataset $DATASET --batch_size $INFERENCE_BATCH_SIZE --ckpt_name $CKPT_NAME --results_name $CKPT_NAME --max_examples $MAX_EXAMPLES --split $SPLIT
  cd ../eval
  OUT_FN="${STEP_DIR}/${SPLIT}_predictions.csv"
  CUDA_VISIBLE_DEVICES=$DEVICE bash run_all.sh $DATASET $OUT_FN $METRIC
  cd ../model
done
