#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "No experiment"
    exit 1
fi

for step in "${@:2}"
do
  FROM_FN="weights/${1}/ckpt_${step}_steps/test_predictions.csv"
  TO_FN="$HOME/data_tmp/weights/${1}/ckpt_${step}_steps/test_predictions.csv"
  echo "Download test predictions from ${FROM_FN} to ${TO_FN}"
  ~/azcopy copy --from-to=BlobLocal "https://sustainsysdata.blob.core.windows.net/recipe-gen/${FROM_FN}?${SAS_TOKEN}" $TO_DIR
done
echo "Fini!"
