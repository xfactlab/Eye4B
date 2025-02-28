from tqdm import tqdm
import torch
import os
import json
import argparse
import gc
from utils import model_inference, utils, ICL_utils, load_models


def parse_args():
    parser = argparse.ArgumentParser(description='I2T ICL Inference')

    parser.add_argument('--dataDir', default='/', type=str, help='Data directory.')
    parser.add_argument('--jsonDir', default='example', type=str, help='Json directory.')
    parser.add_argument('--dataset', default='brl', type=str, choices=['brl', 'operator_induction', 'textocr', 'open_mi', 
                                                                       'clevr','operator_induction_interleaved', 'matching_mi',])
    parser.add_argument('--query_dataset', default='valid_scenarios_total.json', type=str, choices=['valid_scenarios_total.json', 'valid_scenarios.json', 'query.json', 'mobility_pilot_study.json', 'mobility_pilot_study_extra.json'])
    parser.add_argument('--support_dataset', default='support.json', type=str, choices=['support_outdoor.json', 'support_indoor.json'])
    parser.add_argument("--engine", "-e", choices=["openflamingo", "otter-llama", "llava16-7b", "qwen-vl", "qwen-vl-chat", 'internlm-x2', 
                                                   'emu2-chat', 'idefics-9b-instruct', 'idefics-80b-instruct', 'gpt4v'],
                        default=["qwen-vl"], nargs="+")
    # parser.add_argument('--n_shot', default=[0,1,2,3,4], nargs="+", help='Number of support images.')
    parser.add_argument('--n_shot', default=3, type=int, help='Number of support images.')
    parser.add_argument('--max-new-tokens', default=256, type=int, help='Max new tokens for generation.')
    parser.add_argument('--task_description', default='detailed', type=str, choices=['nothing', 'concise', 'detailed'], help='Detailed level of task description.')

    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--sampling', default=True, type=bool, help='Sampling for generation.')
    parser.add_argument('--top_k', default=50, type=int, help='Top k for sampling.')
    parser.add_argument('--top_p', default=0.95, type=float, help='Top p for sampling.')
    parser.add_argument('--temperature', default=0.8, type=float, help='Temperature for sampling.')
    parser.add_argument('--num_return_sequences', default=1, type=int, help='Number of return sequences.')
    parser.add_argument('--multiple_support_dataset', default='t', type=str, help='Whether to use two different types of support datasets')
    return parser.parse_args()


def eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, n_shot):
    data_path = args.dataDir
    results = []
    max_new_tokens = args.max_new_tokens
    print(len(query_meta))

    for query in tqdm(query_meta):
        
        n_shot_support = ICL_utils.select_demonstration(args, support_meta, n_shot, args.dataset, query=query)

    #     try:
        predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                             n_shot_support, data_path, processor, max_new_tokens)
        query['prediction'] = predicted_answer
        results.append(query)
        print(predicted_answer)
    #     except Exception as e:
    #         print(e)
    #         print(f"Couldn't generate deep context for the following query: {query}")

    return results
    

if __name__ == "__main__":
    args = parse_args()

    query_meta, support_meta = utils.load_data(args)
    query_dataset_name = args.query_dataset.split('.')[0]
    for engine in args.engine:

        model, tokenizer, processor = load_models.load_i2t_model(engine, args)
        print("Loaded model: {}\n".format(engine))
        
        utils.set_random_seed(args.seed)
        # for shot in args.n_shot:
        shot = args.n_shot
        results_dict = eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, int(shot))
        os.makedirs(f"results/{args.dataset}", exist_ok=True)
        with open(f"results/{args.dataset}/{engine}_{shot}-shot_{query_dataset_name}.json", "w") as f:
            json.dump(results_dict, f, indent=4)

        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()