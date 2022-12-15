#!/bin/bash
set -e

TO_DIR=$HOME/data_tmp/weights
echo "Download experiment $1 to $TO_DIR"
~/azcopy_linux_amd64_10.16.2/azcopy copy --recursive=true --from-to=BlobLocal "https://sustainsysdata.blob.core.windows.net/recipe-gen/weights/${1}?${SAS_TOKEN}" $TO_DIR
echo "Fini!"
