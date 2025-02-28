'''
@File       :   BLIPScore.py
@Time       :   2023/02/19 20:48:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   BLIPScore.
* Based on BLIP code base
* https://github.com/salesforce/BLIP
'''

import os
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Github.ImageReward.ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain


def load_model(model, ckpt_path = None):
    model_name = ckpt_path
    print('load checkpoint from %s'%model_name)
    checkpoint = torch.load(model_name, map_location='cpu') 
    state_dict = checkpoint
    msg = model.load_state_dict(state_dict,strict=False)
    print("missing keys:", msg.missing_keys)

    return model 


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _open_image(image_key):
    image_path = os.path.join("/projects/eunki/OID/dev/", image_key+".jpg")
    image = Image.open(image_path).convert("RGB").resize((336, 336))
    return image

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _read_convert_image(path):
    return Image.open(path).convert("RGB")

class BLIPScore(nn.Module):
    def __init__(self, med_config, dataset='filtered_oid', device='cpu'):
        super().__init__()
        self.dataset = dataset
        self.device = device
        
        self.preprocess = _transform(224)
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)

        self.blip.eval()


    def score(self, prompt, image_path):
        
        if (type(image_path).__name__=='list'):
            _, rewards = self.inference_rank(prompt, image_path)
            return rewards
            
        # text encode
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        text_output = self.blip.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:,0,:]))
        
        # image encode
        pil_image = Image.open(image_path)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)
        image_features = F.normalize(self.blip.vision_proj(image_embeds[:,0,:]), dim=-1)    
        
        # score
        rewards = torch.sum(torch.mul(txt_feature, image_features), dim=1, keepdim=True)
        
        return rewards.detach().cpu().numpy().item()


    def inference_rank(self, prompt, generations_list):
    
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        text_output = self.blip.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:,0,:]))
        
        txt_set = []
        img_set = []
        for generations in generations_list:
            # image encode
            img_path = generations
            pil_image = Image.open(img_path)
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)
            image_features = F.normalize(self.blip.vision_proj(image_embeds[:,0,:]), dim=-1)    
            img_set.append(image_features)
            txt_set.append(txt_feature)
            
        txt_features = torch.cat(txt_set, 0).float() # [image_num, feature_dim]
        img_features = torch.cat(img_set, 0).float() # [image_num, feature_dim]
        rewards = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()

    def batchify(self, targets, batch_size):
        return [targets[i:i+batch_size] for i in range(0, len(targets), batch_size)]

    def select_preference(self, chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst, B=1):
        chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst = [self.batchify(x, B) for x in [chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst]]

        scores = []
        for chosen_imgs, chosen_caps, rejected_caps in (pbar:= tqdm(zip(chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst), total=len(chosen_image_key_lst))):

            if self.dataset == 'filtered_oid':
                chosen_imgs = [_open_image(img) for img in chosen_imgs]
            chosen_imgs = torch.cat([self.preprocess(img).unsqueeze(0) for img in chosen_imgs], dim=0).to(self.device)
            input_ids_chosen = self.blip.tokenizer([cap for cap in chosen_caps], padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
            input_ids_rejected = self.blip.tokenizer([cap for cap in rejected_caps], padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)

            with torch.no_grad():
                chosen_imgs = self.blip.visual_encoder(chosen_imgs)
                chosen_imgs = F.normalize(self.blip.vision_proj(chosen_imgs[:,0,:]), dim=-1)    
                
                chosen_mts = self.blip.text_encoder(input_ids_chosen.input_ids,
                                                    attention_mask = input_ids_chosen.attention_mask,
                                                    mode='text') 
                chosen_mts = self.blip.text_proj(chosen_mts.last_hidden_state[:,0,:])
                
                rejected_mts = self.blip.text_encoder(input_ids_rejected.input_ids,
                                                      attention_mask = input_ids_rejected.attention_mask,
                                                      mode='text') 
                rejected_mts = self.blip.text_proj(rejected_mts.last_hidden_state[:,0,:])
                
                reward_chosen = F.cosine_similarity(chosen_imgs, chosen_mts, eps=0)
                reward_rejected = F.cosine_similarity(chosen_imgs, rejected_mts, eps=0)

            bool_scores = (reward_chosen > reward_rejected).tolist()
            s = list(map(int, bool_scores))
            scores.extend(s)
                    
        return scores

    def calculate_corr_acc(self, batch_img_dir, batch_request, batch_gpt_text, batch_label):

        # image encode
        imgs = [Image.open(image_path) for image_path in batch_img_dir]
        image = torch.cat([self.preprocess(img).unsqueeze(0) for img in imgs], dim=0).to(self.device)

        # text encode
        text_input = self.blip.tokenizer(['Request: ' + batch_request[i] + '\nResponse: ' + batch_gpt_text[i] for i in range(len(batch_gpt_text))],
                                   padding='max_length', truncate=True, max_length=35, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embeds = self.blip.visual_encoder(image)
            image_features = F.normalize(self.blip.vision_proj(image_embeds[:,0,:]), dim=-1) 

            text_output = self.blip.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
            txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:,0,:]))
                    
            sims = F.cosine_similarity(image_features, txt_feature, eps=0)

        sims = sims.tolist()
        batch_label = batch_label.tolist()
        new_sims = [int(val * 4) / 4 for val in sims]
        acc_lst = [1 if s == t else 0 for (s,t) in zip(new_sims, batch_label)]
        acc = sum(acc_lst) / len(acc_lst)
                    
        return sims, batch_label, acc 

    def __call__(self, mt_list, img_list, B=32, no_ref=False):
        mt_list, img_list = [self.batchify(x,B) for x in [mt_list, img_list]]
        scores = []
        assert len(mt_list) == len(img_list)
        for prompt, img_paths in (pbar:= tqdm(zip(mt_list, img_list), total=len(mt_list))):
            # text encode
            text_input = self.blip.tokenizer([cap for cap in prompt], padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
            text_output = self.blip.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
            txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:,0,:]))
            
            # image encode
            imgs = [Image.open(image_path) for image_path in img_paths]
            image = torch.cat([self.preprocess(img).unsqueeze(0) for img in imgs], dim=0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)
            image_features = F.normalize(self.blip.vision_proj(image_embeds[:,0,:]), dim=-1)    
            
            # score
            rewards = torch.sum(torch.mul(txt_feature, image_features), dim=1, keepdim=True)
            scores.extend(rewards.detach().cpu().numpy())
        return scores
        
        