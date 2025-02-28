#!/bin/bash

for dataset in flickr8k_expert # brl_new # imgreward_test #filtered_polaris #filtered_oid #flickr8k_cf #flickr8k_expert # polaris # brl
do
    echo Data type: "$dataset"
    for context_dataset in flickr8k.json #gp_avg.csv gp_overall.csv #ImageReward/data #export_116070_project-116070-at-2024-11-26-14-30-faad6f04.csv #ImageReward/data  #OID-rated-image-captions.v2.dev.alignment.tsv #crowdflower_flickr8k.json #flickr8k.json #polaris_test.csv # qwen-vl_4-shot_mobility_pilot_study.json #internlm-x2_4-shot_mobility_pilot_study.json llava16-7b_4-shot_mobility_pilot_study.json openflamingo_4-shot_mobility_pilot_study.json qwen-vl-chat_4-shot_mobility_pilot_study.json otter-llama_4-shot_mobility_pilot_study.json # qwen-vl_3-shot_mobility_pilot_study.json internlm-x2_3-shot_mobility_pilot_study.json llava16-7b_3-shot_mobility_pilot_study.json openflamingo_3-shot_mobility_pilot_study.json qwen-vl-chat_3 -shot_mobility_pilot_study.json otter-llama_3-shot_mobility_pilot_study.json
    do
        echo Context data type: "$context_dataset"
        for model_type in imagereward #blip-s #imagereward #longclip-s # qwen_rlaif-s_inst #llava_rlaif-s_inst #llava_rlaif-s #clip-s #llava_rlaif-s # longclip-s #polos #blip # contextclip # longclip # clip
        do
            echo Model type: "$model_type"
            python evaluate.py \
            --dataset "$dataset" \
            --context_dataset "$context_dataset" \
            --model_type "$model_type"
            # --model_name_or_path /userhomes/namin/BarrierFree/Github/ImageReward/train/checkpoint/blip_uni_cross_mul_bs128_fix=0.7_lr=1e-05cosine/best_lr=1e-05.pt # filename = '/userhomes/namin/BarrierFree/Github/ImageReward/train/checkpoint/blip_uni_cross_mul_bs16_fix=0.7_lr=1e-05cosine/best_lr=1e-05.pt' # sample brl dataset # ImageReward-v1.0
        done
    done
done