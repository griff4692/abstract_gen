#!/bin/bash
set -e

DEVICE=$1
DATASET=$2
EXPERIMENT=$3
METRIC=$4

DIR="${HOME}/data_tmp/weights/$EXPERIMENT"
BEST_STEP_FN="${DIR}/the_chosen_one.txt"
INFERENCE_BATCH_SIZE=8
MAX_VAL_EXAMPLES=256
MAX_TEST_EXAMPLES=999999999

if [ ! -f $BEST_STEP_FN ]
then
  SPLIT="validation"
  for STEP in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
  do
    CKPT_NAME="ckpt_${STEP}_steps"
    STEP_DIR="${DIR}/${CKPT_NAME}"

    if [ ! -d $STEP_DIR ]
    then
      echo "$STEP_DIR does not exist. Skipping."
      continue
    fi

    python inference.py --hf_model primera --device $DEVICE --experiment $EXPERIMENT --dataset $DATASET --batch_size $INFERENCE_BATCH_SIZE --ckpt_name $CKPT_NAME --results_name $CKPT_NAME --max_examples $MAX_VAL_EXAMPLES --split $SPLIT
    cd ../eval
    OUT_FN="${STEP_DIR}/${SPLIT}_predictions.csv"
    CUDA_VISIBLE_DEVICES=$DEVICE bash run_all.sh $DATASET $OUT_FN $METRIC
    cd ../model
  done

  echo "Fini generating for validation set! Finding best validation set"

  cd ../eval
  OUT_FN="${STEP_DIR}/${SPLIT}_predictions.csv"
  python find_best_val_ckpt.py --dataset $DATASET --experiment $EXPERIMENT --metric $METRIC
  cd ../model
fi

CHOSEN_STEP=$(head -n 1 $BEST_STEP_FN |  grep -oE "[^=]+$")
echo "${CHOSEN_STEP} is the best step on the validation set."

CKPT_NAME="ckpt_${CHOSEN_STEP}_steps"
STEP_DIR="${DIR}/${CKPT_NAME}"
SPLIT="test"

python inference.py --hf_model primera --device $DEVICE --experiment $EXPERIMENT --dataset $DATASET --batch_size $INFERENCE_BATCH_SIZE --ckpt_name $CKPT_NAME --results_name $CKPT_NAME --max_examples $MAX_TEST_EXAMPLES --split $SPLIT
cd ../eval
OUT_FN="${STEP_DIR}/${SPLIT}_predictions.csv"

CUDA_VISIBLE_DEVICES=$DEVICE bash run_all.sh $DATASET $OUT_FN all
echo "Fini Fini! Please paste results below into le google sheets by experiment ${EXPERIMENT}..."
python run.py --dataset $DATASET --fp $OUT_FN --mode to_table
