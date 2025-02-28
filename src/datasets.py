import os
from os import path
import json
import glob
from tqdm import tqdm
from typing import Dict
import random
import base64
import scipy
import pandas as pd
from collections import defaultdict
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_data(args, data_type, data_dir, task_type=None) -> Dict[int, str]:
    one_sample = args.one_sample
    pilot_sample = args.pilot_sample
    if pilot_sample:
        stop_len = 10
    final_sample = args.final_sample
    
    sampled_data = defaultdict(dict)
    idx = 0    
    if task_type == 'generation_deepcontexts':
        folder_path = Path(f'{data_dir}/deepcontexts')
        file_path = os.path.join(folder_path, f'{data_type}.json') 
        
        with open(file_path, 'r') as file:
            data = json.load(file)
        idx = 0
        print(len(data))
        for i in range(len(data)):
            sample = data[i]
            full_img_path = sample['image_path']
            base64_image = encode_image(full_img_path)
            scenario = sample['scenario']
            pos_dg = sample['pos_dg'] 

            if 'pilot' in args.data_type:
                neg_dg = sample['neg_dg']
                sampled_data[idx][f'neg_dg'] = neg_dg
            elif '4blv' in args.data_type:
                neg_keys = [k for k in sample.keys() if 'neg_' in k]
                for neg_key in neg_keys:
                    sampled_data[idx][neg_key] = sample[neg_key]
            else:
                for i in range(5):
                    neg_dg = sample[f'neg_dg{i}']
                    sampled_data[idx][f'neg_dg{i}'] = neg_dg
        
            sampled_data[idx]['image_dir'] = full_img_path
            sampled_data[idx]['image'] = base64_image
            txt = f"Here is an example. This is a sample image, request, and model response.\nRequest: {scenario}\nResponse: {pos_dg} \n Enhanced response:"
            sampled_data[idx]['text'] = txt
            sampled_data[idx]['pos_dg'] = pos_dg
            idx += 1
                
            if one_sample:
                break
            if pilot_sample and len(sampled_data) == stop_len:
                break
                
        num_data = len(sampled_data)
        print(f"Total Number of Data for Deep Context Generation: {num_data}")
    else:
        if final_sample:
            data_folder_type_dict = {'boundingbox':'jpg',
                                     'segmentations':'jpg',
                                     'outdoor':'png', 
                                     'indoor':'png'}
        else:
            data_folder_type_dict = {'sideguide':{'boundingbox':'jpg', 'segmentations':'jpg'},
                                     'sidewalk':{'바운딩박스':'jpg'},
                                     'outdoor':{'Bridge':'png',
                                                'Building_area':'png',
                                                'Market_1':'png',
                                                'Market_2':'png',
                                                'Market_3':'png',
                                                'Market_5':'png',
                                                'Park':'png',
                                                'Residential_area_1':'png',
                                                'Residential_area_2':'png',
                                                'School':'png'}, 
                                     'indoor':{'balcony':'png',
                                               'elevator':'png',
                                               'kitchen':'png',
                                               'living_room':'png',
                                               'meeting_room':'png',
                                               'office':'png',
                                               'stairs':'png',
                                               'storage':'png',
                                               'utility_room':'png',
                                               'yard':'png'}, 
                                     'obstacle':{'JPEGImages':'jpg'},
                                     'wotr':{'JPEGImages':'jpg'}}
    
        if final_sample:        
            folder_path = Path(f'{data_dir}/{data_type}')
            
            for img_path in folder_path.glob(f'*.{data_folder_type_dict[data_type]}'):
                full_img_path = os.path.join(folder_path, img_path)  
                base64_image = encode_image(full_img_path)
                sampled_data[idx]['image'] = base64_image
                sampled_data[idx]['image_dir'] = full_img_path
                idx += 1
                
        elif pilot_sample or one_sample:
            folder_type_dict = data_folder_type_dict[data_type]

            if data_type in ['sideguide', 'sidewalk']:
                for folder_type in folder_type_dict.keys():
                    
                    for folder in tqdm(os.listdir(f'{data_dir}/{data_type}/{folder_type}')): 
                        folder_path = Path(f'{data_dir}/{data_type}/{folder_type}/{folder}')
                                                   
                        for img_path in folder_path.glob(f'*.{folder_type_dict[folder_type]}'):
                            full_img_path = os.path.join(folder_path, img_path)           
                            base64_image = encode_image(full_img_path)
                            sampled_data[idx]['image'] = base64_image
                            sampled_data[idx]['image_dir'] = full_img_path
                            idx += 1
        
                            if one_sample:
                                break
                            if pilot_sample and len(sampled_data) == stop_len:
                                break
                        if one_sample:
                            break
                        if pilot_sample and len(sampled_data) == stop_len:
                            break
                    if one_sample:
                        break
                    if pilot_sample and len(sampled_data) == stop_len:
                        break
    
            elif data_type in ['outdoor', 'indoor']:
                for folder_type in folder_type_dict.keys():
                    folder_path = Path(f'{data_dir}/{data_type}/{folder_type}')
                    
                    for img_path in folder_path.glob(f'*.{folder_type_dict[folder_type]}'):
                        full_img_path = os.path.join(folder_path, img_path)           
                        base64_image = encode_image(full_img_path)
                        sampled_data[idx]['image'] = base64_image
                        sampled_data[idx]['image_dir'] = full_img_path
                        idx += 1
                        
                        if one_sample:
                            break
                        if pilot_sample and len(sampled_data) == stop_len:
                            break
                    if one_sample:
                        break
                    if pilot_sample and len(sampled_data) == stop_len:
                        break
                
            elif data_type in ['obstacle', 'wotr']:
                for folder_type in folder_type_dict.keys():
                    folder_path = Path(f'{data_dir}/{data_type}/{folder_type}')
                    
                    for img_path in folder_path.glob(f'*.{folder_type_dict[folder_type]}'):
                        full_img_path = os.path.join(folder_path, img_path)           
                        base64_image = encode_image(full_img_path)
                        sampled_data[idx]['image'] = base64_image
                        sampled_data[idx]['image_dir'] = full_img_path
                        idx += 1
                        
                        if one_sample:
                            break
                        if pilot_sample and len(sampled_data) == stop_len:
                            break
                    if one_sample:
                        break
                    if pilot_sample and len(sampled_data) == stop_len:
                        break

        num_data = len(sampled_data)
        print(f"Total Number of Data for Scenario Generation: {num_data}")
        
    return sampled_data

class CVPRDataset(Dataset):
    def __init__(self, dataset, img_dir_path):
        self.dataset = dataset
        self.img_dir_path = img_dir_path

        # Filter out entries with broken image files
        for data in self.dataset:
            assert self.is_image_ok(path.join(img_dir_path, f"{data['imgid']}"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from copy import deepcopy
        # Get image from df
        imgid = self.dataset[idx]["imgid"]
        img_name = path.join(self.img_dir_path, f"{imgid}")

        # Get label from  df
        labels = deepcopy(self.dataset[idx])

        # print(labels)

        # Open image file
        # from detectron2.data.detection_utils import read_image
        # img = read_image(img_name, format="RGB")
        img = Image.open(img_name).convert("RGB")

        labels["img"] = img

        return labels

    @staticmethod
    def is_image_ok(img_path):
        # Try to open the image file
        try:
            img = Image.open(img_path)
            img.verify()
            return True
        except (IOError, SyntaxError) as e:
            return False

class Pascal50sDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root: str = "/projects/brl/mobility/pascal50s",
                 media_size: int = 224,
                 voc_path: str = "data/VOC2010/"):
        super().__init__()
        self.voc_path = voc_path
        self.fix_seed()
        self.read_data(root)
        self.read_score(root)
        self.idx2cat = {1: 'HC', 2: 'HI', 3: 'HM', 4: 'MM'}

    @staticmethod
    def loadmat(path):
        return scipy.io.loadmat(path)

    def fix_seed(self, seed=42):
        torch.manual_seed(seed)
        random.seed(seed)

    def read_data(self, root):
        mat = self.loadmat(
            os.path.join(root, "pair_pascal.mat")) #"pyCIDErConsensus/pair_pascal.mat"))
        self.data = mat["new_input"][0]
        self.categories = mat["category"][0]
        # sanity check
        c = torch.Tensor(mat["new_data"])
        hc = (c.sum(dim=-1) == 12).int()
        hi = (c.sum(dim=-1) == 13).int()
        hm = ((c < 6).sum(dim=-1) == 1).int()
        mm = ((c < 6).sum(dim=-1) == 2).int()
        assert 1000 == hc.sum()
        assert 1000 == hi.sum()
        assert 1000 == hm.sum()
        assert 1000 == mm.sum()
        assert (hc + hi + hm + mm).sum() == self.categories.shape[0]
        chk = (torch.Tensor(self.categories) - hc - hi * 2 - hm * 3 - mm * 4)
        assert 0 == chk.abs().sum(), chk

    def read_score(self, root):
        mat = self.loadmat(
            os.path.join(root, "consensus_pascal.mat")) #"pyCIDErConsensus/consensus_pascal.mat"))
        data = mat["triplets"][0]
        self.labels = []
        self.references = []
        for i in range(len(self)):
            votes = {}
            refs = []
            for j in range(i * 48, (i + 1) * 48):
                a,b,c,d = [x[0][0] for x in data[j]]
                key = b[0].strip() if 1 == d else c[0].strip()
                refs.append(a[0].strip())
                votes[key] = votes.get(key, 0) + 1
            assert 2 >= len(votes.keys()), votes
            assert len(votes.keys()) > 0
            try:
                vote_a = votes.get(self.data[i][1][0].strip(), 0)
                vote_b = votes.get(self.data[i][2][0].strip(), 0)
            except KeyError:
                print("warning: data mismatch!")
                print(f"a: {self.data[i][1][0].strip()}")
                print(f"b: {self.data[i][2][0].strip()}")
                print(votes)
                exit()
            # Ties are broken randomly.
            label = 0 if vote_a > vote_b + random.random() - .5 else 1 # a == bの場合は0.5の確率で0か1を選ぶ
            self.labels.append(label)
            self.references.append(refs)

    def __len__(self):
        return len(self.data)

    def get_image_path(self, filename: str):
        path = os.path.join(self.voc_path, "JPEGImages")
        return os.path.join(path, filename)

    def __getitem__(self, idx: int):
        vid, a, b = [x[0] for x in self.data[idx]]
        label = self.labels[idx]
        img_path = self.get_image_path(vid)
        a = a.strip()
        b = b.strip()
        references = self.references[idx]
        category = self.categories[idx]
        category_str = self.idx2cat[category]
        return img_path, a, b, references, category_str, label

class FoilDataset:
    def __init__(self, coco_root_path="data_en/coco", foil_path="data_en/foil/foilv1.0_test_2017.json", coco_image_path=None):
        coco_root_path = Path(coco_root_path)
        coco_path = coco_root_path / Path("captions_val2014.json")
        coco_refs = self._read_coco(coco_path)
        self.data = self._build_foil(foil_path, coco_refs) # data[anno_id][foil or orig] = [anno1, anno2, ...]
        self.coco_root_path = coco_root_path
        self.dataset = {"one_ref" : None, "four_ref" : None}
        self.coco_image_path = coco_image_path

    def _read_coco(self, coco_annos):
        refs = {}
        with open(coco_annos) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            refs.setdefault(ann['image_id'],[]).append(ann['caption'])
        return refs
    
    def _build_foil(self, path, coco_refs):
        with open(path) as f:
            self.data = json.load(f)
        images = self.data["images"]
        annos = self.data["annotations"]

        data = {}
        imgid_to_img = {img["id"] : img for img in images}
        for anno in annos:
            anno_id = anno["id"]
            data.setdefault(anno_id, {"foil" : [], "orig" : []})
            key = "foil" if anno["foil"] else "orig"
            anno["image"] = imgid_to_img[anno["image_id"]]
            anno["refs"] = coco_refs[anno["image_id"]]
            data[anno_id][key].append(anno)
        
        return data

    def get_data(self,one_ref):
        key = "one_ref" if one_ref else "four_ref"
        if self.dataset[key] is not None:
            return self.dataset[key]
        
        dataset = []
        for _, data in (pbar := tqdm(self.data.items())):  # data[anno_id][foil or orig] = [anno1, anno2, ...]
            pbar.set_description("Prepare dataset ...")
            foiles, origs = data["foil"], data["orig"]

            assert len(origs) == 1
            N = len(foiles)
            for foil, orig in zip(foiles, [origs[0]]*N):
                refs = foil["refs"]
                refs = [r for r in refs if r != orig["caption"]]
                if one_ref:
                    refs = [refs[0]]
                
                filename = Path(foil["image"]["file_name"])
                img_path = Path(f"{self.coco_image_path}") / filename

                dataset.append({
                    "imgid" : img_path,
                    "refs": refs,
                    "mt": foil["caption"],
                    "type": "foil"
                })
                dataset.append({
                    "imgid" : img_path,
                    "refs": refs,
                    "mt": orig["caption"],
                    "type": "orig"
                })
        
        self.dataset[key] = dataset
        return self.dataset[key]

class Flickr8k(torch.utils.data.Dataset):
    def __init__(self, json_file, root='datasets/flickr8k/',
                 transform=None, load_images=False):
        self.im_folder = os.path.join(root, 'images')
        self.transform = transform
        self.load_images = load_images

        with open(os.path.join(root, json_file)) as fp:
            data = json.load(fp)

        self.data = list()
        for i in data:
            for human_judgement in data[i]['human_judgement']:
                if np.isnan(human_judgement['rating']):
                    print('NaN')
                    continue
                d = {
                    'image': data[i]['image_path'].split('/')[-1],
                    'references': [' '.join(gt.split()) for gt in data[i]['ground_truth']],
                    'candidate': ' '.join(human_judgement['caption'].split()),
                    'human_score': human_judgement['rating']
                }
                self.data.append(d)

    def get_image(self, filename):
        img = Image.open(os.path.join(self.im_folder, filename)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im_idx = self.data[idx]['image']
        candidate = self.data[idx]['candidate']
        references = self.data[idx]['references']
        score = self.data[idx]['human_score']

        if self.load_images:
            im = self.get_image(im_idx)
        else:
            im = os.path.join(self.im_folder, im_idx)

        return im, candidate, references, score

class AlignmentDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.sort_values(by=["IMAGE_KEY", "ROUNDED_AVG_USER_RATING"], ascending=[True, False]).reset_index(drop=True)
        self.pairs = []
        for key, group in self.data.groupby("IMAGE_KEY"):
            if len(group) == 2:
                chosen = group.iloc[0]
                rejected = group.iloc[1]
                self.pairs.append((chosen, rejected))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        chosen_sample, rejected_sample = self.pairs[idx]
        return chosen_sample["IMAGE_KEY"], chosen_sample["CAPTION_PRED"], rejected_sample["CAPTION_PRED"]

class BRLDataset(Dataset):
    def __init__(self, df):
        category2label = {'Strongly Agree': 1.0,
                          'Agree': 0.75,
                          'Neutral': 0.5,
                          'Disagree': 0.25,
                          'Strongly Disagree': 0.0}
        df = df[~df['accuracy'].isna()].reset_index()
        self.data = []
        for i in range(df.shape[0]):
            img_dir = '/projects/brl/mobility/chosen_final/' + df['img_dir'][i].split('gs://human_experiment/')[-1]
            request = df['request'][i]
            gpt_text = df['gpt_txt'][i]
            accuracy = df['accuracy'][i]
            label = category2label[accuracy]
            self.data.append((img_dir, request, gpt_text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dir, request, gpt_text, label = self.data[idx]
        # img = Image.open(img_dir).convert('RGB')
        return img_dir, request, gpt_text, label

class BRLFinalDataset(Dataset):
    def __init__(self, df):
        self.data = []
        for i in range(df.shape[0]):
            img_dir = '/projects/brl/mobility/chosen_final/' + df['img_dir'][i].split('gs://human_experiment/')[-1]
            request = df['request'][i]
            text = df['txt'][i]
            label = df['score'][i] / 5
            self.data.append((img_dir, request, text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dir, request, text, label = self.data[idx]
        # img = Image.open(img_dir).convert('RGB')
        return img_dir, request, text, label

class IRDataset(Dataset):
    def __init__(self, data, data_dir):
        self.data = data
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_id = self.data[idx]['id']
        prompt = self.data[idx]['prompt']
        gens = [os.path.join(self.data_dir, self.data[idx]['generations'][j]) for j in range(len(self.data[idx]['generations']))]
        ranking = self.data[idx]['ranking']
        return data_id, prompt, gens, ranking

class IRTrainingDataset(Dataset):
    def __init__(self, data_type, data_dir, score_key='overall_rating', seed=42):  
        self.data_dir = data_dir
        if data_type == 'train':
            meta_path = f'{self.data_dir}/metadata-train.parquet'            
        if data_type == 'valid':
            meta_path = f'{self.data_dir}/metadata-validation.parquet'            
        if data_type == 'test':
            meta_path = f'{self.data_dir}/metadata-test.parquet'
            
        df = pd.read_parquet(meta_path, engine='pyarrow')
        # if data_type == 'train':
        #     grouped = df.groupby(score_key)
        #     min_count = grouped.size().min()
        
        #     df = grouped.apply(lambda x: x.sample(n=min_count, random_state=seed)).reset_index(drop=True)
            
        full_img_lst = []
        prompt_lst =  []
        human_score_lst = []
        new_df = pd.DataFrame()
        for i in range(df.shape[0]):
            image_path_split = df['image_path'][i].split('/')
            full_img_path = os.path.join(self.data_dir, '/'.join(image_path_split[:2]) + '/' + image_path_split[-1])
            try:
                image = Image.open(full_img_path).convert("RGB")
                full_img_lst.append(full_img_path) 
                prompt_lst.append(df['prompt'][i])
                human_score_lst.append(df['overall_rating'][i])
            except:
                # print(i, df.shape[0], full_img_path)
                pass
        new_df['img'] = full_img_lst
        new_df['cand'] = prompt_lst
        new_df['human_score'] = human_score_lst
        df = new_df[['img', 'cand', 'human_score']]
        self.data = HFDataset.from_pandas(df)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]['img']
        prompt = self.data[idx]['cand']
        human_score = self.data[idx]['human_score']
        return img, cand, human_score