#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "No experiment"
    exit 1
fi

TO_DIR=$HOME/data_tmp/weights
echo "Download experiment $1 to $TO_DIR"
~/azcopy copy --recursive=true --from-to=BlobLocal "https://sustainsysdata.blob.core.windows.net/recipe-gen/weights/${1}?${SAS_TOKEN}" $TO_DIR
echo "Fini!"
