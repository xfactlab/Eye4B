# https://github.com/keio-smilab24/Polos/blob/12613990ae7cf336a39ee98fae034ba6bd2a6aaa/validate/clip_score.py#L30
import os
import sys
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2 as cv
from transformers import AutoProcessor
from src.args import SNUModelArguments, SNUEvaluationArguments, SNUDataArguments
from src.load import load_tokenizer_and_model
from modules import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    LLAVA_CHAT_TEMPLATE,
    QWEN_CHAT_TEMPLATE,
    INTERNLM_CHAT_TEMPLATE,
    load_model_with_index
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_image(imgid):
    from pathlib import Path
    vanilla = Path(imgid)
    fixed = Path(f"/projects/brl/mobility/polaris/images/{imgid}")
    # assert not (vanilla.exists() == fixed.exists()) 

    path = vanilla if vanilla.exists() else fixed
    return Image.open(path).convert("RGB")

def _convert_image(image):
    image =  np.array(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image = cv.resize(image, (336, 336))
    return image
    
class RewardMetricScore():
    def __init__(self, model_type, dataset_type, model_name_or_path, use_peft, peft_checkpoint_dir=None, checkpoint_dir = None, device="cuda"):
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.device = device
        if model_type in ['llava_rlaif-s', 'llava_rlaif-s_inst']:
            from models.reward_model import RewardConfig, RewardModel
            self.is_snu = True
            self.model_args = SNUModelArguments(model_name_or_path=model_name_or_path,
                                        checkpoint_dir=checkpoint_dir)
            self.evaluation_args = SNUEvaluationArguments(output_dir="./output")
            self.data_args = SNUDataArguments(eval_dataset_path="/projects/eunki/OID/OID-rated-image-captions.v2.dev.alignment.tsv",                                              
                                      eval_type="Polaris")
            
            if self.model_type == 'llava_rlaif-s_inst':
                from llava.conversation import conv_templates
                self.conv_templates = conv_templates
                self.conv_mode = 'llava_v1'
                reward_prompt_file = "/userhomes/namin/BarrierFree/prompts/brl_rlaif_reward_prompt.txt"
                with open(reward_prompt_file, "r") as f:
                    self.reward_model_prompt_untokenized = " " + f.read().strip() 
            
            tokenizer, self.vis_preprocess, self.model = load_tokenizer_and_model(self.model_args, self.evaluation_args, self.data_args)

            self.model.eval()
            if tokenizer.chat_template == None:
                if 'llava' in self.model_type.lower():
                    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
                elif 'qwen' in self.model_type.lower():
                    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
                elif 'internlm' in self.model_type.lower():
                    tokenizer.chat_template = INTERNLM_CHAT_TEMPLATE
            self.conv_mode = 'llava_v1'
            self.tokenizer = tokenizer         
            
        else:  
            self.is_snu = False
            from models.reward_model_tuned import VLMRewardModel, VLMRewardConfig
            self.model_args = ModelArguments(model_name_or_path=model_name_or_path,
                                            use_peft=use_peft,
                                            peft_checkpoint_dir=peft_checkpoint_dir,
                                            checkpoint_dir=checkpoint_dir)
            self.config = VLMRewardConfig()
            self.model = VLMRewardModel(config=self.config, args=self.model_args)
            load_model_with_index(self.model, checkpoint_dir)
            print(f"Loaded model from {checkpoint_dir}")
            self.model = self.model.to(self.device)

        
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            if self.processor == None:
                if 'llava' in self.model_type.lower():
                    self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
                elif 'qwen' in self.model_type.lower():
                    self.processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL")
                elif 'internlm' in self.model_type.lower():
                    self.processor = AutoProcessor.from_pretrained("internlm/internlm-xcomposer2-7b")
            if self.processor.chat_template == None:
                if 'llava' in self.model_type.lower():
                    self.processor.chat_template = LLAVA_CHAT_TEMPLATE
                elif 'qwen' in self.model_type.lower():
                    self.processor.chat_template = QWEN_CHAT_TEMPLATE
                elif 'internlm' in self.model_type.lower():
                    self.processor.chat_template = INTERNLM_CHAT_TEMPLATE

        if 'inst' in self.model_type:
            reward_prompt_file = "/userhomes/namin/BarrierFree/prompts/brl_rlaif_reward_prompt.txt"
            with open(reward_prompt_file, "r") as f:
                self.reward_model_prompt_untokenized = " " + f.read().strip() 
            self.factual_prompt = "Specifically, the AI's response should be fully supported by the combination of the following caption:\n"

    def batchify(self, targets, batch_size):
        return [targets[i:i+batch_size] for i in range(0,len(targets),batch_size)]

    def select_preference(self, chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst, B=1):
        scores = []
        if self.dataset_type == 'filtered_oid':
            for chosen_img_key, chosen_cap, rejected_cap in zip(chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst):
                image_path = os.path.join("/projects/eunki/OID/dev/", chosen_img_key+".jpg")
                chosen_img = Image.open(image_path).convert("RGB")
                
                factual_prompt = self.factual_prompt + f" - {chosen_cap}\n"
                chosen_prompt = self.processor.apply_chat_template([
                    {"role": "user", "content": [
                        {"type": "text", "text": self.reward_model_prompt_untokenized.format(factual_prompt=factual_prompt)},
                         {"type": "image"}],
                    }
                ], tokenize=False)
                chosen_input = self.processor(text=[chosen_prompt], images=[chosen_img], return_tensors="pt", padding=True)
       
                factual_prompt = self.factual_prompt + f" - {rejected_cap}\n"
                rejected_prompt = self.processor.apply_chat_template([
                    {"role": "user", "content": [
                        {"type": "text", "text": self.reward_model_prompt_untokenized.format(factual_prompt=factual_prompt)},
                         {"type": "image"}],
                    }
                ], tokenize=False)
                rejected_input = self.processor(text=[rejected_prompt], images=[chosen_img], return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    reward_chosen = self.model(**chosen_input.to(self.device)).rewards
                    reward_rejected = self.model(**rejected_input.to(self.device)).rewards
    
                bool_scores = (reward_chosen > reward_rejected).tolist()
                s = list(map(int, bool_scores))
                scores.extend(s)
                
        elif self.dataset_type == 'filtered_polaris':
            chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst = [self.batchify(x,B) for x in [chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst]]
    
            for chosen_imgs, chosen_caps, rejected_caps in (pbar:= tqdm(zip(chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst), total=len(chosen_image_key_lst))):
                chosen_imgs = [_convert_image(img) for img in chosen_imgs]
                    
                new_chosen_caps = []
                for cap in chosen_caps:
                    factual_prompt = self.factual_prompt + f"  - {cap}\n"
                    chosen_prompt = self.processor.apply_chat_template([
                        {"role": "user", "content": [
                            {"type": "text", "text": self.reward_model_prompt_untokenized.format(factual_prompt=factual_prompt)},
                            {"type": "image"}],
                        }
                    ], tokenize=False)
                    new_chosen_caps.append(chosen_prompt)
    
                new_rejected_caps = []
                for cap in rejected_caps:
                    factual_prompt = self.factual_prompt + f"  - {cap}\n"
                    rejected_prompt = self.processor.apply_chat_template([
                        {"role": "user", "content": [
                            {"type": "text", "text": self.reward_model_prompt_untokenized.format(factual_prompt=factual_prompt)},
                            {"type": "image"}],
                        }
                    ], tokenize=False)
                    new_rejected_caps.append(chosen_prompt)
                    
                chosen_input = self.processor(text=new_chosen_caps, images=chosen_imgs, return_tensors="pt", padding=True) #'max_length')
                rejected_input = self.processor(text=new_rejected_caps, images=chosen_imgs, return_tensors="pt", padding=True) #'max_length')

                with torch.no_grad():
                    reward_chosen = self.model(**chosen_input.to(self.device)).rewards
                    reward_rejected = self.model(**rejected_input.to(self.device)).rewards
    
                bool_scores = (reward_chosen > reward_rejected).tolist()
                s = list(map(int, bool_scores))
                scores.extend(s)
                    
        return scores

    def calculate_corr_acc(self, batch_img_dir, batch_request, batch_gpt_text, batch_label):
        if self.model_type == 'llava_rlaif-s_inst':
            chosen_imgs = [_convert_image(Image.open(img_dir)) for img_dir in batch_img_dir]
            chosen_imgs = torch.cat([torch.tensor(img, dtype=torch.bfloat16).permute(2, 0, 1).unsqueeze(0) for img in chosen_imgs], dim=0).to(self.device)
            
            new_chosen_caps = []
            for cap in batch_gpt_text:
                conv = self.conv_templates[self.conv_mode].copy()
                factual_prompt = self.factual_prompt + f"  - {cap}\n"
                conv.append_message(conv.roles[0], self.reward_model_prompt_untokenized.format(factual_prompt=factual_prompt))
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                new_chosen_caps.append(prompt)

            chosen_data = self.tokenizer(new_chosen_caps, return_tensors="pt", padding='max_length')
            chosen_input_ids = chosen_data.data['input_ids'].to(self.device)
            chosen_attention_mask = chosen_data.data['attention_mask'].to(self.device)

            with torch.no_grad():
                sims = self.model(input_ids = chosen_input_ids,
                                  attention_mask = chosen_attention_mask,
                                  images = chosen_imgs).rewards

        else:
            chosen_img = [Image.open(img_dir).convert("RGB") for img_dir in batch_img_dir][0]
            
            factual_prompt = self.factual_prompt + f" - {batch_gpt_text}\n"
            chosen_prompt = self.processor.apply_chat_template([
                {"role": "user", "content": [
                    {"type": "text", "text": self.reward_model_prompt_untokenized.format(factual_prompt=factual_prompt)},
                     {"type": "image"}],
                }
            ], tokenize=False)
            chosen_input = self.processor(text=[chosen_prompt], images=[chosen_img], return_tensors="pt", padding=True)
    
            with torch.no_grad():
                sims = self.model(input_ids = chosen_input['input_ids'].to(self.device),
                                  attention_mask = chosen_input['attention_mask'].to(self.device),
                                  pixel_values = chosen_input['pixel_values'].to(self.device),
                                  image_grid_thw = chosen_input["image_grid_thw"].to(self.device)).rewards

        sims = sims.cpu().numpy().tolist()
        batch_label = batch_label.tolist()
        new_sims = [int(val * 4) / 4 for val in sims]
        acc_lst = [1 if s == t else 0 for (s,t) in zip(new_sims, batch_label)]
        acc = sum(acc_lst) / len(acc_lst)
                    
        return sims, batch_label, acc 

    def inference_rank(self, prompt, img_lst):
        scores = []
        for img in img_lst:
            imgs = [_read_image(img).resize((224,224))]
            mt_input = self.processor.apply_chat_template([
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."}
                ]}], tokenize=False) + self.processor.apply_chat_template([{"role": "assistant", "content": [
                    {"type": "text", "text": prompt},],
                }], tokenize=False)           
            processed = self.processor(text=[mt_input], images=imgs, return_tensors="pt")
            with torch.no_grad():
                reward_score = self.model(**processed.to(self.device)).rewards
                scores.extend(reward_score.tolist())
        return scores

    def __call__(self, mt_list, img_list, B=32, no_ref=False):
        # if self.dataset_type in ['polaris', 'foil']:
        scores = []
        if self.is_snu:
            mt_list, img_list = [self.batchify(x,B) for x in [mt_list, img_list]]
            assert len(mt_list) == len(img_list)
            for mt, imgs in (pbar:= tqdm(zip(mt_list, img_list), total=len(mt_list))):
                imgs = [_read_image(imgid) for imgid in imgs]
                imgs = torch.cat([torch.tensor(self.vis_preprocess(img).data['pixel_values'][0], dtype=torch.bfloat16).unsqueeze(0) for img in imgs], dim=0).to(self.device)

                texts_input_ids = self.tokenizer(mt, return_tensors="pt", padding='max_length').data['input_ids'].to(self.device)
                texts_attention_mask = self.tokenizer(mt, return_tensors="pt", padding='max_length').data['attention_mask'].to(self.device)

                reward_score = self.model(input_ids = texts_input_ids,
                                        attention_mask = texts_attention_mask,
                                        images = imgs).rewards
                scores.extend(reward_score.tolist())
        else:
            for mt, img in (pbar:= tqdm(zip(mt_list, img_list), total=len(mt_list))):
                imgs = [_read_image(img).resize((224,224))]
                mt_input = self.processor.apply_chat_template([
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image."}
                    ]}], tokenize=False) + self.processor.apply_chat_template([{"role": "assistant", "content": [
                        {"type": "text", "text": mt},],
                    }], tokenize=False)                
                processed = self.processor(text=[mt_input], images=imgs, return_tensors="pt")
                with torch.no_grad():
                    reward_score = self.model(**processed.to(self.device)).rewards
                    scores.extend(reward_score.tolist())
        return scores