#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "No experiment"
    exit 1
fi

for step in "${@:2}"
do
  STEP_DIR=weights/${1}/ckpt_${step}_steps
  TO_DIR=$HOME/data_tmp/$STEP_DIR
  echo "Download test predictions from ${STEP_DIR} to ${TO_DIR}"
  ~/azcopy copy --from-to=BlobLocal "https://sustainsysdata.blob.core.windows.net/recipe-gen/${STEP_DIR}/test_predictions.csv?${SAS_TOKEN}" $TO_DIR
done
echo "Fini!"
