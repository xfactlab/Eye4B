import os
import argparse
from tqdm import tqdm
import torch
import json
import urllib

from src.utils import set_seed, SCORES, download, ImageReward_download
from src.load import load_eval_data
from src.evaluation import EvalModel


parser = argparse.ArgumentParser(description='Barrier Free BRL Project for Evaluating Generated Deep Contexts')
parser.add_argument("--model_type", type=str, default="clip",
                    choices=['clip', 'longclip', 'polos',
                             'clip-s', 'contextclip-s', 'longclip-s', 'pac-s',
                             'llava_rlaif-s', 'llava_rlaif-s_inst',
                             'qwen_rlaif-s', 'qwen_rlaif-s_inst',
                             'imagereward', 'blip-s'], help="Which model to use for evaluation")
parser.add_argument('--dataset', default='brl', type=str,
                    choices=['brl', 'polaris',  'pascal50s', 'foil',
                             'flickr8k_expert', 'flickr8k_cf',
                             'filtered_oid', 'filtered_polaris',
                             'brl_new', 'imgreward_test', 'brl_final'])
parser.add_argument("--context_dataset", type=str, default="qwen-vl_3-shot_mobility_pilot_study.json",
                    help="The files including the generated deep contexts",
                    choices=['VOCdevkit/VOC2010',
                             'gp_overall.csv',
                             'gp_avg.csv',
                             'ImageReward/data',
                             'export_116070_project-116070-at-2024-11-26-14-30-faad6f04.csv',
                             'export_116070_project-116070-at-2024-11-25-07-34-81134997.csv',
                             'yuwd',
                             'OID-rated-image-captions.v2.dev.alignment.tsv',
                             'flickr8k.json',
                             'crowdflower_flickr8k.json',
                             'foilv1.0_test_2017.json',
                             'polaris_test.csv',
                             'qwen-vl_3-shot_mobility_pilot_study.json',
                             'internlm-x2_3-shot_mobility_pilot_study.json',
                             'llava16-7b_3-shot_mobility_pilot_study.json',
                             'openflamingo_3-shot_mobility_pilot_study.json',
                             'qwen-vl-chat_3-shot_mobility_pilot_study.json',
                             'otter-llama_3-shot_mobility_pilot_study.json',
                             'qwen-vl_4-shot_mobility_pilot_study.json',
                             'internlm-x2_4-shot_mobility_pilot_study.json',
                             'llava16-7b_4-shot_mobility_pilot_study.json',
                             'openflamingo_4-shot_mobility_pilot_study.json',
                             'qwen-vl-chat_4-shot_mobility_pilot_study.json',
                             'otter-llama_4-shot_mobility_pilot_study.json'])
parser.add_argument("--use_peft", type=bool, default=False, help="Use LoRA model")
parser.add_argument("--model_name_or_path", type=str, default="/userhomes/namin/BarrierFree/Github/ImageReward/train/checkpoint/blip_uni_cross_mul_bs128_fix=0.7_lr=1e-05cosine/best_lr=1e-05.pt",
                    choices=["Qwen/Qwen2-VL-2B-Instruct", "SNUMPR/vlm_sft_video_llava_13b", "ImageReward-v1.0",
                             "/userhomes/namin/BarrierFree/Github/ImageReward/train/checkpoint/blip_uni_cross_mul_bs128_fix=0.7_lr=1e-05cosine/best_lr=1e-05.pt", "/userhomes/namin/BarrierFree/Github/ImageReward/train/checkpoint/blip_uni_cross_mul_bs16_fix=0.7_lr=1e-05cosine/best_lr=1e-05.pt"],
                    help="Backbone model name or path")
parser.add_argument("--peft_checkpoint_dir", type=str, default=None, help="Checkpoint path")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint path for reward head")
parser.add_argument("--reference_free", type=bool, default=False, help="Whether to implement reference-free Polos")
parser.add_argument("--result_dir", type=str, default="/projects/brl/mobility/score_results", help="Where to save results")
parser.add_argument("--seed", type=int, default=42, help="Reproducibility purpose")

args = parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Prepare deep context data
    eval_data = load_eval_data(args.dataset, args.context_dataset)

    # Prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")  
    gen = EvalModel(args.model_type, args.dataset, eval_data, device)

    # Load model and evaluate the generated deep contexts
    if args.dataset == 'brl':
        gen.load_eval_model()
        file_name = args.context_dataset.split('.json')[0]
        image_paths = gen.scenario_dic.keys()
        print(f'Evaluation image data size: {len(image_paths)}')
        for image_path in tqdm(image_paths):
            gen.evaluate(image_path)
            
    elif args.dataset in ['polaris', 'pascal50s', 'foil',
                          'flickr8k_expert', 'flickr8k_cf',
                          'filtered_oid', 'filtered_polaris',
                          'brl_new', 'imgreward_test', 'brl_final']:
        if 'csv' in args.context_dataset:
            file_name = args.context_dataset.split('.csv')[0]
        elif 'json' in args.context_dataset:
            file_name = args.context_dataset.split('.json')[0]
        elif 'tsv' in args.context_dataset:
            file_name = args.context_dataset.split('.tsv')[0]
        elif '/' in args.context_dataset:
            file_name = args.context_dataset.split('/')[-1]
        else:
            file_name = args.context_dataset
            
        if args.model_type in ['polos']:
            gen.load_eval_model()
            mod = gen.model
        elif args.model_type in ['clip-s', 'longclip-s', 'pac-s', 'contextclip-s']:
            from models.clip_score import CLIPScore
            mod = CLIPScore(args.model_type, args.dataset, device) 
        elif 'rlaif' in args.model_type.lower():
            from models.vlm_rlaif_score import RewardMetricScore
            mod = RewardMetricScore(
                model_type = args.model_type, 
                dataset_type = args.dataset, 
                model_name_or_path = args.model_name_or_path, 
                use_peft = args.use_peft, 
                peft_checkpoint_dir = args.peft_checkpoint_dir, 
                checkpoint_dir = args.checkpoint_dir, 
                device = device)
        elif 'imagereward' in args.model_type.lower():
            import ImageReward as ImageReward
            mod = ImageReward.load("ImageReward-v1.0").to(device)

            model_path = '/userhomes/namin/BarrierFree/Github/ImageReward/train/checkpoint/blip_uni_cross_mul_bs256_fix=0.0_lr=1e-07cosine/best_lr=1e-07.pt' #blip_uni_cross_mul_bs128_fix=0.7_lr=5e-06cosine/best_lr=5e-06.pt' # blip_uni_cross_mul_bs256_fix=0.8_lr=1e-06cosine/best_lr=1e-06.pt' # blip_uni_cross_mul_bs128_fix=0.7_lr=1e-05cosine/best_lr=1e-05.pt' #blip_uni_cross_mul_bs256_fix=0.7_lr=1e-05cosine/best_lr=1e-05_final.pt'
            print('load checkpoint from %s'%model_path)
            checkpoint = torch.load(model_path, map_location='cpu') 
            state_dict = checkpoint
            for key in list(state_dict.keys()):
                state_dict[key.replace('module.', '')] = state_dict[key]
                del state_dict[key]

            msg = mod.load_state_dict(state_dict, strict=False)
            print("missing keys:", msg.missing_keys)
            mod.eval()
                
        elif args.model_type in ['blip-s']:
            from models.blip_score import BLIPScore
            name = 'BLIP'
            model_download_root = "/projects/namin/"
            model_path = download(SCORES[name], model_download_root)
            state_dict = torch.load(model_path, map_location=device)
            med_config = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json", model_download_root)
            mod = BLIPScore(med_config=med_config, dataset=args.dataset, device=device).to(device)
            # mod.blip.load_state_dict(state_dict['model'], strict=False)

            model_path = '/userhomes/namin/BarrierFree/Github/ImageReward/train/checkpoint/blip_uni_cross_mul_bs256_fix=0.7_lr=5e-07cosine/best_lr=5e-07.pt'
            print('load checkpoint from %s'%model_path)
            checkpoint = torch.load(model_path, map_location='cpu') 
            state_dict = checkpoint
            for key in list(state_dict.keys()):
                state_dict[key.replace('module.', '')] = state_dict[key]
                del state_dict[key]
            msg = mod.load_state_dict(state_dict, strict=False)
            print("missing keys:", msg.missing_keys)
            mod.eval()
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for the dataset {args.dataset}")
        gen.evaluate_whole(mod, args.dataset, args)

    scores = gen.score_lst
    with open(f'{args.result_dir}/{args.dataset}_{file_name}_{args.model_type}_scores.json','w') as fw:
        json.dump(scores, fw)
        fw.write("\n")
    print(f'Evaluated "{args.dataset}_{file_name}_{args.model_type}_scores.json" in {args.result_dir}...')