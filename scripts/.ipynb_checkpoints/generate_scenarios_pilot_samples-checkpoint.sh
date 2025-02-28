#!/bin/bash

for data_type in outdoor indoor # sideguide obstacle wotr sidewalk outdoor indoor
do
    echo Data type: "$data_type"
    python generate_scenarios.py \
    --data_type "$data_type" \
    --pilot_sample \
    --few_shot_k 3 \
    --project_dir /userhomes/namin/BarrierFree
done