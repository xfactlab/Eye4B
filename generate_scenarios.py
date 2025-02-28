import os
import argparse
import datetime
import json
from tqdm import tqdm
import pandas as pd
import torch
import openai

from src.utils import set_seed
from src.datasets import load_data
from src.prompts import generate_inputs
from src.generation import Generation


parser = argparse.ArgumentParser(description='Barrier Free BRL Project for Generating Mobility Scenarios')

parser.add_argument("--data_type", type=str, default="sideguide", 
                    choices=['sideguide', 'obstacle', 'outdoor', 'indoor', 'wotr', 'sidewalk', 'boundingbox', 'segmentations'], help="Which dataset to sample")
parser.add_argument("--one_sample", action='store_true', help="Whether to generate scenarios for the first sample")
parser.add_argument("--pilot_sample", action='store_true', help="Whether to generate scenarios for the pilot study")
parser.add_argument("--final_sample", action='store_true', help="Whether to generate scenarios for the final human experiment")
parser.add_argument("--with_image", action='store_true', help="Whether to use image in the prompt")
parser.add_argument("--few_shot_k", type=int, default=0,
                    choices=[0, 1, 3], help="How many context examples in the prompt")
parser.add_argument("--max_tokens", type=int, default=300)
parser.add_argument("--model_type", type=str, default="gpt-4o", 
                    choices=['gpt-4o-mini', 'gpt-4o'], help="Which model to use for scenario generation")

# parser.add_argument("--api_key", type=str, help="Your OpenAI api key")
api_key = os.environ.get('OPENAI_API_KEY')
parser.add_argument("--seed", type=int, default=42, help="Reproducibility purpose")
parser.add_argument("--project_dir", type=str, default="~/BarrierFree", help="Where your project directory is")
parser.add_argument("--data_dir", type=str, default="/projects/brl/mobility/chosen_final",
                    choices=["/projects/brl/mobility", "/projects/brl/mobility/chosen_final"], help="Where to locate images")
parser.add_argument("--result_dir", type=str, default="/projects/brl/mobility/results", help="Where to save results")

parser.parse_args()


if __name__ == '__main__':
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Prepare dataset
    sampled_data = load_data(args, args.data_type, args.data_dir)

    # Prepare inputs for model
    system_prompt, user_prompt = generate_inputs(args)

    # Prepare model
    mod = Generation(args.model_type, system_prompt, user_prompt, args.max_tokens, api_key)

    # Generate scenarios
    for (idx,dict) in tqdm(sampled_data.items()):

        try:
            response = mod.generate_response(dict['image'], args.few_shot_k, args.with_image)
            response_lst = [s.origscenario for s in response.origscenarios]
            sampled_data[idx]['response'] = response_lst
            sampled_data[idx]['image'] = sampled_data[idx]['image_dir']
        except:
            print(idx)

    # Save results
    os.makedirs(args.result_dir, exist_ok=True)
    if args.one_sample:
        tot_data = 'ones'
    elif args.pilot_sample:
        tot_data = 'pilot'
    else:
        tot_data = 'total'
    json_file = os.path.join(args.result_dir, f'{args.data_type}_{tot_data}_{args.few_shot_k}_{args.model_type}_withimage.json')

    with open(json_file, 'w') as f:
        json.dump(sampled_data, f, indent=2)

    print(f'JSON File dumped as "{os.path.basename(json_file)}" in "{args.result_dir}"')