# https://github.com/keio-smilab24/Polos/blob/12613990ae7cf336a39ee98fae034ba6bd2a6aaa/validate/clip_score.py#L30
import os
import sys
import torch
import clip
import cv2 as cv
import numpy as np
import scipy
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Github.LongCLIP.model import longclip

def _read_image(imgid):
    from pathlib import Path
    vanilla = Path(imgid)
    fixed = Path(f"/projects/brl/mobility/polaris/images/{imgid}")
    assert not (vanilla.exists() == fixed.exists()) 

    path = vanilla if vanilla.exists() else fixed
    return Image.open(path).convert("RGB")

def _open_image(image_key):
    image_path = os.path.join("/projects/eunki/OID/dev/", image_key+".jpg")
    image = Image.open(image_path).convert("RGB").resize((336, 336))
    return image

def _convert_image(image):
    image = image.convert("RGB").resize((336, 336))
    return image

def _read_convert_image(path):
    return Image.open(path).convert("RGB")

class CapDataset(torch.utils.data.Dataset):
    def __init__(self, data, tok, prefix='A photo depicts'):
        self.data = data
        self.tok = tok
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = self.tok.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        if transform:
            self.preprocess = transform
        else:
            self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)
        
def extract_all_captions(captions, model, tok, device, batch_size=256, num_workers=6):
    data = torch.utils.data.DataLoader(CapDataset(captions, tok),
                                       batch_size=batch_size, num_workers=num_workers,
                                       shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).detach().cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features

def extract_all_images(images, model, transform, device, batch_size=64, num_workers=6):
    data = torch.utils.data.DataLoader(ImageDataset(images, transform), batch_size=batch_size, num_workers=num_workers,
                                       shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['image'].to(device)
            all_image_features.append(model.encode_image(b).detach().cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features

class CLIPScore():
    def __init__(self, model_type, dataset_type, device="cuda"):
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.device = device
        if self.model_type in ['clip-s', 'contextclip-s', 'pac-s']:
            import clip
            self.clip, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.tok = clip
            if self.model_type == 'pac-s':
                self.clip = self.clip.float()
                checkpoint = torch.load("/projects/namin/pacscore_checkpoints/clip_ViT-B-32.pth")
                self.clip.load_state_dict(checkpoint['state_dict'])
        elif self.model_type == 'longclip-s':
            self.clip, self.clip_preprocess = longclip.load("/projects/namin/longclip-B.pt", device=self.device)
            self.tok = longclip
        self.clip.eval()

    def batchify(self, targets, batch_size):
        return [targets[i:i+batch_size] for i in range(0,len(targets),batch_size)]

    def select_preference(self, chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst, B=1):
        chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst = [self.batchify(x,B) for x in [chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst]]

        scores = []
        for chosen_imgs, chosen_caps, rejected_caps in (pbar:= tqdm(zip(chosen_image_key_lst, chosen_caption_lst, rejected_caption_lst), total=len(chosen_image_key_lst))):
            if self.dataset_type == 'filtered_polaris':
                chosen_imgs = [_convert_image(img) for img in chosen_imgs]
            elif self.dataset_type == 'filtered_oid':
                chosen_imgs = [_open_image(img) for img in chosen_imgs]
            chosen_imgs = torch.cat([self.clip_preprocess(img).unsqueeze(0) for img in chosen_imgs], dim=0).to(self.device)
            
            input_ids_chosen = self.tok.tokenize([cap for cap in chosen_caps], truncate=True).to(self.device)
            input_ids_rejected = self.tok.tokenize([cap for cap in rejected_caps], truncate=True).to(self.device)

            with torch.no_grad():
                chosen_imgs = self.clip.encode_image(chosen_imgs)
                chosen_mts = self.clip.encode_text(input_ids_chosen)
                rejected_mts = self.clip.encode_text(input_ids_rejected)
                
                reward_chosen = F.cosine_similarity(chosen_imgs, chosen_mts, eps=0)
                reward_rejected = F.cosine_similarity(chosen_imgs, rejected_mts, eps=0)

            bool_scores = (reward_chosen > reward_rejected).tolist()
            s = list(map(int, bool_scores))
            scores.extend(s)
                    
        return scores

    def calculate_corr_acc(self, batch_img_dir, batch_request, batch_gpt_text, batch_label):
        if self.model_type  == 'clip-s': 
            imgs = [_read_convert_image(img) for img in batch_img_dir]
            imgs = torch.cat([self.clip_preprocess(img).unsqueeze(0) for img in imgs], dim=0).to(self.device)
            
            txts = self.tok.tokenize(['Request: ' + batch_request[i] + '\nResponse: ' + batch_gpt_text[i] for i in range(len(batch_gpt_text))], truncate=True).to(self.device)
    
            with torch.no_grad():
                imgs = self.clip.encode_image(imgs)
                txts = self.clip.encode_text(txts)
                
                sims = F.cosine_similarity(imgs, txts, eps=0)
                
        elif self.model_type  == 'contextclip-s':
            imgs = [_read_convert_image(img) for img in batch_img_dir]
            imgs = torch.cat([self.clip_preprocess(img).unsqueeze(0) for img in imgs], dim=0).to(self.device)
            
            txts = self.tok.tokenize([batch_gpt_text[i] for i in range(len(batch_gpt_text))], truncate=True).to(self.device)
            context_txts = self.tok.tokenize([batch_request[i] for i in range(len(batch_gpt_text))], truncate=True).to(self.device)
    
            with torch.no_grad():
                imgs = self.clip.encode_image(imgs)
                txts = self.clip.encode_text(txts)
                context_txts = self.clip.encode_text(context_txts)

                p1 = F.cosine_similarity(txts, context_txts, eps=0)
                p2 = F.cosine_similarity(txts, F.normalize(imgs, dim=-1) - F.normalize(context_txts, dim=-1), eps=0)
                sims = p1 + torch.max(p2, dim=-1)[0]

        sims = sims.tolist()
        batch_label = batch_label.tolist()
        new_sims = [int(val * 4) / 4 for val in sims]
        acc_lst = [1 if s == t else 0 for (s,t) in zip(new_sims, batch_label)]
        acc = sum(acc_lst) / len(acc_lst)
                    
        return sims, batch_label, acc 

    def inference_rank(self, prompt, img_lst):
        imgs = [_read_convert_image(imgid) for imgid in img_lst]
        imgs = torch.cat([self.clip_preprocess(img).unsqueeze(0) for img in imgs], dim=0).to(self.device)
        imgs = self.clip.encode_image(imgs)
        mts = self.tok.tokenize(["A photo depicts " + prompt], truncate=True).to(self.device)
        mts = self.clip.encode_text(mts)
        cos = F.cosine_similarity(imgs, mts, eps=0)
        cos[cos < 0.] = 0.
        clip_score = 2.5 * cos
   
        return clip_score.tolist()


    def __call__(self, mt_list, img_list, B=32, no_ref=False):
        if self.dataset_type in ['polaris', 'foil', 'pascal50s']:
            mt_list, img_list = [self.batchify(x,B) for x in [mt_list, img_list]]
            scores = []
            assert len(mt_list) == len(img_list)
            for mt, imgs in (pbar:= tqdm(zip(mt_list, img_list), total=len(mt_list))):
                imgs = [_read_image(imgid) for imgid in imgs]
    
                mts = self.tok.tokenize(["A photo depicts " + x for x in mt], truncate=True).to(self.device)
                imgs = torch.cat([self.clip_preprocess(img).unsqueeze(0) for img in imgs], dim=0).to(self.device)
    
                imgs = self.clip.encode_image(imgs)
                mts = self.clip.encode_text(mts)
                cos = F.cosine_similarity(imgs, mts, eps=0)
                cos[cos < 0.] = 0.
                clip_score = 2.5 * cos
    
                if no_ref:
                    scores.extend(clip_score.tolist())
                    continue
    
                cos = F.cosine_similarity(imgs, mts, eps=0)
                cos[cos < 0.] = 0.
                clip_score2 = cos
    
                assert clip_score.shape == clip_score2.shape
                clip_score = 2.0 * clip_score * clip_score2 / (clip_score + clip_score2)
    
                if not no_ref:
                    scores.extend(clip_score.tolist())

        elif 'flickr8k' in self.dataset_type:
            scores = []
            images = img_list
            candidates = mt_list
            len_candidates = [len(c.split()) for c in candidates] 
            if isinstance(images, list):
                # extracting image features
                images = extract_all_images(images, self.clip, self.clip_preprocess, self.device)
        
            candidates = extract_all_captions(candidates, self.clip, self.tok, self.device)
        
            images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
            candidates = candidates / np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))
        
            per = 2.5 * np.clip(np.sum(images * candidates, axis=1), 0, None)
            scores.extend(per.tolist())
        
        return scores