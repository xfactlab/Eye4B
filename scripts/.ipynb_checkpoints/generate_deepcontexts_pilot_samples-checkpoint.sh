#!/bin/bash

for data_type in mobility_pilot_study
do
    echo Data type: "$data_type"
    python generate_deepcontexts.py \
    --data_type "$data_type" \
    --pilot_sample \
    --with_image \
    --project_dir /userhomes/namin/BarrierFree
done