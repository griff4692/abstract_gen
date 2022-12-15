#!/usr/bin/env bash

for d in $HOME/data_tmp/weights/*; do
	echo "Uploading $d"
	~/azcopy_linux_amd64_10.16.2/azcopy copy --recursive=true --from-to=LocalBlob $d "https://sustainsysdata.blob.core.windows.net/recipe-gen/weights/?${SAS_TOKEN}"
done
