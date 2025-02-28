#!/bin/bash

for dataset in filtered_polaris #  pascal50s #imgreward_test
do
    echo Data type: "$dataset"
    for context_dataset in yuwd #OID-rated-image-captions.v2.dev.alignment.tsv #VOCdevkit/VOC2010 #ImageReward/data 
    do
        echo Context data type: "$context_dataset"
        for model_type in qwen_rlaif-s_inst
        do
            echo Model type: "$model_type"
            python evaluate.py \
            --dataset "$dataset" \
            --context_dataset "$context_dataset" \
            --model_type "$model_type" \
            --checkpoint_dir "/projects/brl/mobility/ckpt/polairs/Qwen2-VL-2B-Instruct-mse-lr-2e7-equal-re/final_checkpoint" # "/projects/brl/mobility/ckpt/imgreward/Qwen2-VL-2B-Instruct-mse-lr-5e7-equal-imgreward-all/final_checkpoint" # "/projects/brl/mobility/ckpt/polairs/Qwen2-VL-2B-Instruct-mse-lr-2e7-equal-re/final_checkpoint" \
            
        done
    done
done