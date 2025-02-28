#!/bin/bash

for data_type in boundingbox segmentations outdoor indoor
do
    echo Data type: "$data_type"
    python generate_scenarios.py \
    --data_type "$data_type" \
    --final_sample \
    --few_shot_k 3 \
    --with_image \
    --project_dir /userhomes/namin/BarrierFree
done