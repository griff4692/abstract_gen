#!/bin/bash
set -e

DEVICE=$1
DATASET=$2
EXPERIMENT=$3
DIR="/home/ga2530/data_tmp/weights/$EXPERIMENT"
BATCH_SIZE=32
MAX_EXAMPLES=99999999

for STEP in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
do
  CKPT_NAME="ckpt_${STEP}_steps"
  STEP_DIR="${DIR}/${CKPT_NAME}"
  OUT_MODEL_FN="${STEP_DIR}/pytorch_model.bin"
  if [ ! -f $OUT_MODEL_FN ]
  then
    python "$STEP_DIR/zero_to_fp32.py" $STEP_DIR $OUT_MODEL_FN
  fi
  python inference.py --hf_model primera --device $DEVICE --experiment $EXPERIMENT --dataset $DATASET --batch_size $BATCH_SIZE --ckpt_name $CKPT_NAME --results_name $CKPT_NAME --max_test_examples $MAX_EXAMPLES
  cd ../eval
  OUT_FN="${STEP_DIR}/predictions.csv"
  CUDA_VISIBLE_DEVICES=$DEVICE bash run_all.sh $DATASET $OUT_FN
  cd ../model
done

echo "Fini!"
