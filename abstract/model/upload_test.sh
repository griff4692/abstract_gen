#!/usr/bin/env bash

EXP=$1
for step in "${@:2}"
do
  FROM_FN="$HOME/data_tmp/weights/${EXP}/ckpt_${step}_steps/test_predictions.csv"
  TO_PATH="${EXP}/ckpt_${step}_steps/test_predictions.csv"
  echo $FROM_FN
  echo $TO_PATH
  ~/azcopy_linux_amd64_10.16.2/azcopy copy --from-to=LocalBlob $FROM_FN "https://sustainsysdata.blob.core.windows.net/recipe-gen/weights/${TO_PATH}?${SAS_TOKEN}"
done
