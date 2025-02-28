import random
import json
from typing import List


def generate_inputs(args, task_type='generation') -> List:
    
    if 'generation' in task_type:
        if task_type == 'generation':
            start_file_name = 'dict'
        elif task_type == 'generation_deepcontexts':
            start_file_name = 'deepcontexts_dict'
            
        few_shot_k, data_type, project_dir = args.few_shot_k, args.data_type, args.project_dir
        
        with open(f'{project_dir}/inputs/{start_file_name}_system.json', 'r') as fw:
            system_prompt = json.load(fw)['0']

        if few_shot_k == 0:
            user_prompt = None
        else:
            if task_type == 'generation':
                if data_type in ['sideguide', 'obstacle', 'outdoor', 'wotr', 'sidewalk', 'boundingbox', 'segmentations']:
                    if args.with_image:
                        with open(f'{project_dir}/inputs/{start_file_name}_user_outdoor_image.json', 'r') as fw:
                            user_prompt = list(json.load(fw).values())
                    else:
                        with open(f'{project_dir}/inputs/{start_file_name}_user_outdoor.json', 'r') as fw:
                            user_prompt = list(json.load(fw).values())
                    
                elif args.data_type in ['indoor']:
                    if args.with_image:
                        with open(f'{project_dir}/inputs/{start_file_name}_user_indoor_image.json', 'r') as fw:
                            user_prompt = list(json.load(fw).values())                
                    else:
                        with open(f'{project_dir}/inputs/{start_file_name}_user_indoor.json', 'r') as fw:
                            user_prompt = list(json.load(fw).values())
                            
                if few_shot_k < 3:
                    user_prompt = random.sample(user_prompt, few_shot_k)

            elif task_type == 'generation_deepcontexts':
                if args.with_image:
                    with open(f'{project_dir}/inputs/{start_file_name}_user_outdoor_image.json', 'r') as fw:
                        user_prompt1 = list(json.load(fw).values())
                    user_prompt1 = random.sample(user_prompt1, 2) 
                    
                    with open(f'{project_dir}/inputs/{start_file_name}_user_indoor_image.json', 'r') as fw:
                        user_prompt2 = list(json.load(fw).values())     
                    user_prompt2 = random.sample(user_prompt2, 2) 
                    user_prompt = user_prompt1 + user_prompt2
                
    elif task_type == 'translation':
        system_prompt = "You are an expert translator proficient in both English and Korean. Your task is to translate English text into Korean with a focus on accuracy, natural expression, and cultural context. Use fluent, natural Korean that is appropriate for the target audience, considering nuances, idioms, and tone. Ensure the translation preserves the original meaning while sounding natural and culturally appropriate in Korean. Avoid overly literal translations unless necessary for clarity."
        # system_prompt = "You are an English to Korean Translator. I will provide a list containing 5 sentences. You are required to accurately translate these five sentences into Korean. The output should be properly formatted as a list, containing 5 sentences translated into Koreans.'"
    
        user_prompt = None

    return system_prompt, user_prompt