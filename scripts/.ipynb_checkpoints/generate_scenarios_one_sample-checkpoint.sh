#!/bin/bash

echo "Your API key: "
read api_key
for data_type in sideguide obstacle wotr sidewalk #firstper_outdoor firstper_indoor
do
    echo Data type: "$data_type"
    python generate_scenarios.py \
    --data_type "$data_type" \
    --one_sample \
    --few_shot_k 3 \
    --api_key "$api_key" \
    --project_dir /userhomes/namin/BarrierFree
done