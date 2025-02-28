import os
import sys
import json
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import torch
import pandas as pd
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import yprint


def load_eval_data(dataset, context_dataset):
    yprint(f"Processing {dataset} ... (file name: {context_dataset})")
    
    if dataset == 'brl':
        scenario_dic = defaultdict(list)
        deep_context_dic = defaultdict(list)
        with open(f'/userhomes/namin/BarrierFree/VL-ICL/results/{dataset}/{context_dataset}', 'r') as file:
            data = json.load(file)

        for i,line in enumerate(data):
            image_path = line['image']
            scenario = line['question']
            deep_context = line['prediction']
        
            scenario_dic[image_path].append(scenario)
            deep_context_dic[image_path].append(deep_context)
            
        test_dic = {'scenario_dic': scenario_dic, 'deep_context_dic': deep_context_dic}
        return test_dic
            
    elif dataset == 'polaris':
        from src.datasets import CVPRDataset
        df = pd.read_csv(f'/projects/brl/mobility/polaris/{context_dataset}')
        df = df[["mt","refs","score", "imgid"]]
        refs_list = []
        for refs in df["refs"]:
            refs = eval(refs)
            refs_list.append(refs)
    
        df["refs"] = refs_list
        df["mt"] = df["mt"].astype(str)
        df["score"] = df["score"].astype(float)
        df["imgid"] = df["imgid"].astype(str)
        test_dataset = df.to_dict("records")
        test_dataset = CVPRDataset(test_dataset, "/projects/brl/mobility/polaris/images")
        return test_dataset

    elif dataset == 'pascal50s':
        from src.datasets import Pascal50sDataset
        test_dataset = Pascal50sDataset(root="/projects/brl/mobility/pascal50s",
                                        voc_path=f"/projects/brl/mobility/pascal50s/{context_dataset}")
        return test_dataset

    elif dataset == 'foil':
        from src.datasets import FoilDataset
        test_dataset = FoilDataset(coco_root_path="/projects/HBoP/PnP-VQA/lavis/coco/annotations",
                                   foil_path=f"/projects/brl/mobility/foil/{context_dataset}",
                                   coco_image_path="/projects/HBoP/PnP-VQA/lavis/coco/images/val2014")
        return test_dataset

    elif 'flickr8k' in dataset:
        from src.datasets import Flickr8k
        test_dataset = Flickr8k(json_file=context_dataset, 
                                root='/projects/namin/pacscore_datasets/flickr8k/')
        return test_dataset

    elif dataset == 'filtered_oid':
        from src.datasets import AlignmentDataset
        result_df = pd.read_csv(f'/projects/eunki/OID/{context_dataset}', sep='\t')
        test_dataset = AlignmentDataset(result_df)
        return test_dataset

    elif dataset == 'filtered_polaris':
        from datasets import load_dataset
        alignment_dataset = load_dataset(f"{context_dataset}/Polaris")['validation']
        # filter out the samples that human_score is equal or more than 0.5
        test_dataset = alignment_dataset.filter(lambda x: x['human_score'] <= 0.5)  
        return test_dataset

    elif dataset == 'brl_new':
        from src.datasets import BRLDataset
        result_df = pd.read_csv(f'/userhomes/eunki/BarrierFree/data/{context_dataset}')
        test_dataset = BRLDataset(result_df)
        return test_dataset

    elif dataset == 'imgreward_test':
        from src. datasets import IRDataset
        data_dir = f'/userhomes/namin/BarrierFree/Github/{context_dataset}'
        images_dir = '/projects/brl/mobility/imagereward/test_images'
        with open(os.path.join(data_dir, f"test.json"), "r") as f:
            data = json.load(f)
        test_dataset = IRDataset(data, images_dir)
        return test_dataset

    elif dataset == 'brl_final':
        from src.datasets import BRLFinalDataset
        result_df = pd.read_csv(f'/projects/brl/mobility/irb/blv/{context_dataset}')
        test_dataset = BRLFinalDataset(result_df)
        return test_dataset


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)
        
def load_tokenizer_and_model(model_args, evaluation_args, data_args):
    from llava.model import LlavaLlamaForCausalLM
    from models.reward_model import RewardModel, RewardConfig, load_4bit_reward_model_for_inference
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    with DisableLogger():
        reward_base_model = LlavaLlamaForCausalLM.from_pretrained(model_args.model_name_or_path)
    
        reward_vision_tower = reward_base_model.get_vision_tower()
    
        if not reward_vision_tower.is_loaded:
            reward_vision_tower.load_model()
        reward_image_processor = reward_vision_tower.image_processor
    
    config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)

    args = argparse.Namespace(
        **vars(model_args), **vars(evaluation_args), **vars(data_args)
    )

    with DisableLogger():
        reward_model = RewardModel(args=args, config=config, checkpoint_dir=model_args.checkpoint_dir, tokenizer=tokenizer, qlora=True)

    return tokenizer, reward_image_processor, reward_model