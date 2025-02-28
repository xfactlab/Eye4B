#!/bin/bash

for dataset in brl_final #filtered_oid #pascal50s #imgreward_test #brl_new #filtered_polaris #filtered_oid #flickr8k_cf #flickr8k_expert # polaris # brl
do
    echo Data type: "$dataset"
    for context_dataset in gp_avg.csv gp_overall.csv # OID-rated-image-captions.v2.dev.alignment.tsv #yuwd #ImageReward/data  #export_116070_project-116070-at-2024-11-26-14-30-faad6f04.csv #OID-rated-image-captions.v2.dev.alignment.tsv #crowdflower_flickr8k.json #flickr8k.json #polaris_test.csv # qwen-vl_4-shot_mobility_pilot_study.json #internlm-x2_4-shot_mobility_pilot_study.json llava16-7b_4-shot_mobility_pilot_study.json openflamingo_4-shot_mobility_pilot_study.json qwen-vl-chat_4-shot_mobility_pilot_study.json otter-llama_4-shot_mobility_pilot_study.json # qwen-vl_3-shot_mobility_pilot_study.json internlm-x2_3-shot_mobility_pilot_study.json llava16-7b_3-shot_mobility_pilot_study.json openflamingo_3-shot_mobility_pilot_study.json qwen-vl-chat_3-shot_mobility_pilot_study.json otter-llama_3-shot_mobility_pilot_study.json
    do
        echo Context data type: "$context_dataset"
        for model_type in blip-s #polos #pac-s #polos # qwen_rlaif-s_inst #longclip-s # qwen_rlaif-s_inst #llava_rlaif-s_inst #llava_rlaif-s #clip-s #llava_rlaif-s # longclip-s #polos #blip # contextclip # longclip # clip
        do
            echo Model type: "$model_type"
            python evaluate.py \
            --dataset "$dataset" \
            --context_dataset "$context_dataset" \
            --model_type "$model_type" #\
            --reference_free True
        done
    done
done