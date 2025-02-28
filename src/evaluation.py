import os
import sys
import json
import scipy
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import yprint, rprint, collate_fn, cal_acc


class EvalModel:
    def __init__(self, model_type, dataset, eval_data, device):
        self.model_type = model_type
        self.dataset = dataset
        if self.dataset == 'brl':
            self.deep_context_dic = eval_data['deep_context_dic']
            self.scenario_dic = eval_data['scenario_dic']
        elif self.dataset in ['polaris', 'foil',
                              'flickr8k_expert', 'flickr8k_cf',
                              'filtered_oid', 'filtered_polaris',
                              'brl_new', 'imgreward_test', 'brl_final', 'pascal50s']:
            self.test_dataset = eval_data
        
        self.device = device
        self.score_lst = []
    
    def load_eval_model(self):
        if self.model_type in ['clip', 'contextclip']:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
        elif self.model_type == 'longclip':
            from Github.LongCLIP.model import longclip
            self.model, self.preprocess = longclip.load("/projects/namin/longclip-B.pt", device=self.device)
            
        elif self.model_type == 'polos':
            from Github.Polos.polos.models import load_checkpoint
            self.model = load_checkpoint('/projects/brl/mobility/polaris/reprod/reprod.ckpt')
            
    def evaluate(self, image_path):
        temp_dic = {}

        # Load images
        img = Image.open(image_path)
        if self.model_type in ['clip', 'longclip', 'contextclip']:
            image = self.preprocess(img).unsqueeze(0).to(self.device)
    
        # Load texts
        text_lst = self.deep_context_dic[image_path]
        if self.model_type  in ['clip', 'contextclip']:
            text = clip.tokenize(text_lst, truncate=True).to(self.device)
            if self.model_type == 'contextclip':
                context_lst = self.scenario_dic[image_path]
                context = clip.tokenize(context_lst, truncate=True).to(self.device)
        elif self.model_type == 'longclip':
            text = longclip.tokenize(text_lst, truncate=True).to(self.device)

        # Compute scores
        with torch.no_grad():
            if self.model_type == 'clip':
                logits_per_image, _ = self.model(image, text)
                
            elif self.model_type == 'longclip':
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)
                logits_per_image = image_features @ text_features.T
                
            elif self.model_type == 'contextclip':
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)
                context_features = self.model.encode_text(context)
                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                context_features_norm = context_features / context_features.norm(dim=-1, keepdim=True)
                logits_per_image = context_features @ text_features_norm.T + text_features @ (image_features_norm - context_features_norm).T
                
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            temp_dic['image'] = image_path
            temp_dic['text'] = text_lst
            temp_dic['score_prob'] = probs[0].tolist()
            self.score_lst.append(temp_dic)

    def evaluate_whole(self, mod, dataset_type, args):
        from Github.Polos.polos.metrics.regression_metrics import RegressionReport
        yprint(f"Compute {dataset_type}...")
        
        if dataset_type == 'polaris':
            # Load images and texts
            data = []
            img_lst = []
            gt_scores = []
            mt_lst = []
            for data_ in (pbar := tqdm(self.test_dataset)):
                pbar.set_description("Prepare dataset ...")
                data.append(data_)
                img_lst.append(data_["imgid"])
                gt_scores.append(data_["score"])
                mt_lst.append(data_["mt"])
    
            rep = RegressionReport(kendall_type='c')
            # Compute scores
            if self.model_type == 'polos':
                _, sys_scores = mod.predict(data, cuda=True, batch_size=128, reference_free=args.reference_free)
            elif self.model_type in ['clip-s', 'longclip-s', 'llava_rlaif-s', 'qwen_rlaif-s_inst', 'llava_rlaif-s_inst', 'qwen_rlaif-s']:
                sys_scores = mod(mt_lst, img_lst, B=48)
            elif self.model_type in ['imagereward', 'blip-s']:
                sys_scores = []
                img_prefix = '/projects/brl/mobility/polaris/images/'
                for i, img in enumerate(img_lst):
                    img_lst[i] = img_prefix + img
                    assert os.path.exists(img_lst[i])
                for img, mt in tqdm(zip(img_lst, mt_lst)):
                    sys_scores.append(mod.score(mt, img))
            else:
                raise ValueError(f"Model type {self.model_type} is not supported for the dataset {dataset_type}")

            coef_tensor = rep.compute(sys_scores, gt_scores)
            coefs = {k : round(float(v.numpy() if not isinstance(v,float) else v),4) for k, v in coef_tensor.items()}
            self.score_lst.append(coefs)

        elif dataset_type == 'pascal50s':
            data = {}
            data_new = {}
            for (img_path, a, b, references, category_str, label) in (pbar := tqdm(self.test_dataset)):
                pbar.set_description("Prepare dataset ...")
                data.setdefault(category_str, {"A" : [], "B" : [], "gt": []})
                data_new.setdefault(category_str, {"img_path" : [], "A" : [], "B" : [], "gt": []})
                data[category_str]["A"].append({
                    "img" : Image.open(img_path).convert("RGB"),
                    "imgid" : img_path,
                    "refs": references,
                    "mt": a,
                })
                data[category_str]["B"].append({
                    "img" : Image.open(img_path).convert("RGB"),
                    "imgid" : img_path,
                    "refs": references,
                    "mt": b,
                })
                data[category_str]["gt"].append(label) # 0 if A > B else 1
                
                data_new[category_str]["img_path"].append(img_path)
                data_new[category_str]["A"].append(a)
                data_new[category_str]["B"].append(b)
                data_new[category_str]["gt"].append(label)
                
            accs = {}
            if self.model_type == 'polos':
                for category_str, data_ in (pbar := tqdm(data.items())):
                    pbar.set_description(f"Compute {category_str}")
                    _, sys_scoreA = mod.predict(data_["A"], cuda=True, batch_size=512, reference_free=args.reference_free)
                    _, sys_scoreB = mod.predict(data_["B"], cuda=True, batch_size=512, reference_free=args.reference_free)
        
                    print("Compute accuracy ...")
                    assert len(sys_scoreA) == len(sys_scoreB) == len(data_["gt"])
                    acc, N = 0, len(sys_scoreA)
                    for a, b, gt in zip(sys_scoreA,sys_scoreB,data_["gt"]):
                        score = 0 if a > b else 1
                        acc += 1 if score == gt else 0
                    acc /= N
                    accs[category_str] = acc
                    rprint(f"acc({category_str}) : {acc}")
                    
            elif self.model_type in ['clip-s', 'longclip-s', 'llava_rlaif-s', 'qwen_rlaif-s_inst', 'llava_rlaif-s_inst', 'qwen_rlaif-s']:
                for category_str, data_ in (pbar := tqdm(data_new.items())):
                    sys_scoreA = mod(data_["A"], data_["img_path"], B=48)
                    sys_scoreB = mod(data_["B"], data_["img_path"], B=48)

                    print("Compute accuracy ...")
                    assert len(sys_scoreA) == len(sys_scoreB) == len(data_["gt"])
                    acc, N = 0, len(sys_scoreA)
                    for a, b, gt in zip(sys_scoreA,sys_scoreB,data_["gt"]):
                        score = 0 if a > b else 1
                        acc += 1 if score == gt else 0
                    acc /= N
                    accs[category_str] = acc
                    rprint(f"acc({category_str}) : {acc}")
                    
            elif self.model_type in ['imagereward', 'blip-s']:
                for category_str, data_ in (pbar := tqdm(data_new.items())):
                    sys_scoreA, sys_scoreB = [], []
                    
                    for img, mt in tqdm(zip(data_["img_path"], data_["A"])):
                        sys_scoreA.append(mod.score(mt, img))
                    for img, mt in tqdm(zip(data_["img_path"], data_["B"])):
                        sys_scoreB.append(mod.score(mt, img))

                    print("Compute accuracy ...")
                    assert len(sys_scoreA) == len(sys_scoreB) == len(data_["gt"])
                    acc, N = 0, len(sys_scoreA)
                    for a, b, gt in zip(sys_scoreA,sys_scoreB,data_["gt"]):
                        score = 0 if a > b else 1
                        acc += 1 if score == gt else 0
                    acc /= N
                    accs[category_str] = acc
                    rprint(f"acc({category_str}) : {acc}")
            else:
                raise ValueError(f"Model type {self.model_type} is not supported for the dataset {dataset_type}")
    
            self.score_lst.append(accs)

        elif dataset_type == 'foil':
            accs = {}
            for one_ref in [True, False]:
                suffix = "(one_ref)" if one_ref else "(four-ref)"
                dataset_type += suffix
                bucket_count = 5
                data = self.test_dataset.get_data(one_ref)

                print("Compute ...")
                sys_score = []
                for i in range(bucket_count):
                    bucket_size = len(data) // bucket_count
                    subset = deepcopy(data[i*bucket_size:(i+1)*bucket_size])
                    for j, sub in enumerate(pbar := tqdm(subset)):
                        pbar.set_description(f"Processing {i+1}/{bucket_count}")
                        subset[j].update({"img" : Image.open(sub["imgid"]).convert("RGB")})

                    subset_data = []
                    subset_img_lst = []
                    subset_mt_lst = []
                    for data_ in (pbar := tqdm(subset)):
                        pbar.set_description("Preparing dataset ...")
                        subset_data.append(data_)
                        subset_img_lst.append(data_["imgid"])
                        subset_mt_lst.append(data_["mt"])

                    if self.model_type == 'polos':
                        _, sys_scores = mod.predict(subset_data, cuda=True, batch_size=512, reference_free=args.reference_free)
                    elif self.model_type in ['clip-s', 'longclip-s', 'llava_rlaif-s', 'qwen_rlaif-s_inst', 'llava_rlaif-s_inst', 'qwen_rlaif-s', 'blip-s']: 
                        sys_scores = mod(subset_mt_lst, subset_img_lst, B=32)
                    elif self.model_type == 'imagereward':
                        sys_scores = []
                        for i in tqdm(range(len(subset_img_lst))):
                            assert os.path.exists(subset_img_lst[i])
                            sco = mod.score(subset_mt_lst[i], Image.open(subset_img_lst[i]))
                            sys_scores.append(sco)
                        
                    sys_score.extend(sys_scores)
                    del subset
                
                assert len(sys_score) == len(data)
                assert len(sys_score) % 2 == 0
            
                acc = 0.
                N = len(sys_score) // 2
                for i in range(0,2*N,2):
                    s1 = sys_score[i] # foil
                    s2 = sys_score[i+1] # orig
                    
                    # sanity check
                    assert data[i]["type"] == "foil" and data[i+1]["type"] == "orig"
            
                    if s2 > s1:
                        acc += 1.
            
                acc /= N
                rprint(f"acc: {acc}")
                accs[suffix] = acc
    
            self.score_lst.append(accs)

        elif 'flickr8k' in dataset_type:
            dataloader = DataLoader(self.test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
            if dataset_type == 'flickr8k_expert':
                kendall_type = "c"
            elif dataset_type == 'flickr8k_cf':
                kendall_type = "b"

            gen = {}
            gts = {}
        
            human_scores = list()
            ims_cs = list()
            gen_cs = list()
            gts_cs = list()

            rew_lst = []
            
            for it, (images, candidates, references, scores) in enumerate(iter(dataloader)):
                for i, (im_i, gts_i, gen_i, score_i) in enumerate(zip(images, references, candidates, scores)):
                    gen['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
        
                    ims_cs.append(im_i)
                    gen_cs.append(gen_i)
                    gts_cs.append(gts_i)
                    human_scores.append(score_i)

            if self.model_type == 'polos':
                mod.eval()
                data = [{
                    "mt" : gen,
                    "refs": refs,
                    "img": Image.open(image).convert("RGB")
                    } for image, refs, gen in zip(ims_cs, gts_cs, gen_cs)
                ]
                _, sys_scores = mod.predict(data, cuda=True, batch_size=256, reference_free=args.reference_free)
            elif self.model_type in ['clip-s', 'longclip-s', 'blip-s', 'llava_rlaif-s', 'qwen_rlaif-s_inst']:
                sys_scores = mod(gen_cs, ims_cs, B=32) #48)
            elif self.model_type == 'imagereward':
                sys_scores = []
                for i in tqdm(range(len(ims_cs))):
                    sys_score = mod.score(gen_cs[i], ims_cs[i])
                    sys_scores.append(sys_score)
                rew_lst.extend(sys_scores)

            if self.model_type in ['imagereward']:
                print(f'Max: {max(rew_lst)}, Min: {min(rew_lst)}')
                print(f'Human Max: {max(human_scores)}, Min: {min(human_scores)}')

            kendalltau_b = 100 * scipy.stats.kendalltau(sys_scores, human_scores, variant='b')[0]
            kendalltau_c = 100 * scipy.stats.kendalltau(sys_scores, human_scores, variant='c')[0]
            coefs = {'kendalltau_b' : kendalltau_b, 'kendalltau_c' : kendalltau_c}
            self.score_lst.append(coefs)
            
        elif dataset_type == 'filtered_oid': # please note that there are several images missing in this dataset.
            dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn) # must set batch_size=1 to skip only non-existent images
            acc_lst = []
            if self.model_type == 'polos':
                mod.eval()
 
            for batch_chosen_image_key, batch_chosen_caption, batch_rejected_caption in tqdm(iter(dataloader)):
                if self.model_type in ['clip-s', 'longclip-s', 'pac-s', 'blip-s',
                                       'llava_rlaif-s', 'llava_rlaif-s_inst', 'qwen_rlaif-s', 'qwen_rlaif-s_inst']:
                    try:
                        sys_scores = mod.select_preference(batch_chosen_image_key, batch_chosen_caption, batch_rejected_caption)
                        acc_lst.extend(sys_scores)
                    except OSError as e:
                            print(e)
                elif self.model_type == 'imagereward':
                    for i in range(len(batch_chosen_image_key)):
                        try:
                            batch_img_dir = os.path.join("/projects/eunki/OID/dev/", batch_chosen_image_key[i]+".jpg")
                            pred_score1 = mod.score(batch_chosen_caption[i], batch_img_dir)
                            pred_score2 = mod.score(batch_rejected_caption[i], batch_img_dir)
                            if pred_score1 > pred_score2:
                                acc_lst.append(1)
                            else:
                                acc_lst.append(0)
                        except OSError as e:
                            print(e)
                elif self.model_type == 'polos':
                    try:
                        data = [{
                            "mt" : batch_rejected_caption[0],
                            "refs": batch_chosen_caption[0],
                            "img": Image.open(os.path.join("/projects/eunki/OID/dev/", batch_chosen_image_key[0]+".jpg")).convert("RGB")
                            }
                        ]
                        _, sys_scores = mod.predict(data, cuda=True, batch_size=256, reference_free=True)
                        acc_lst.extend(sys_scores)
                    except OSError as e:
                        print(e)

            if self.model_type in ['imagereward']:
                print(f'Mean: {sum(rew_lst)/len(rew_lst)}, STD: {np.std(rew_lst)}')
                print(f'Max: {max(rew_lst)}, Min: {min(rew_lst)}')
                print(f'Human Max: {max(human_scores)}, Min: {min(human_scores)}')
                
            acc = {'acc': sum(acc_lst) / len(acc_lst), 'len': len(acc_lst)}
            self.score_lst.append(acc)

        elif dataset_type == 'filtered_polaris':
            batch_imgs, batch_refs, batch_cands = [], [], []
            for batch in tqdm(self.test_dataset):
                imgs = batch["img"]
                refs = batch["refs"][0]
                cands = batch["cand"]
    
                batch_imgs.append(imgs)
                batch_refs.append(refs)
                batch_cands.append(cands)

            if self.model_type in ['clip-s', 'longclip-s', 'pac-s', 'blip-s',
                                   'llava_rlaif-s', 'llava_rlaif-s_inst', 'qwen_rlaif-s', 'qwen_rlaif-s_inst']:
                acc_scores = mod.select_preference(batch_imgs, batch_refs, batch_cands, B=8)
            elif self.model_type == 'imagereward':
                acc_scores = []
                for i in tqdm(range(len(batch_imgs))):
                    pred_score1 = mod.score(batch_refs[i], batch_imgs[i])
                    pred_score2 = mod.score(batch_cands[i], batch_imgs[i])
                    if pred_score1 > pred_score2:
                        acc_scores.append(1)
                    else:
                        acc_scores.append(0)
            elif self.model_type == 'polos':
                acc_scores = []
                data = [{
                    "mt" : gen,
                    "refs": refs,
                    "img": image
                    } for image, refs, gen in zip(batch_imgs, batch_refs, batch_cands)
                ]
                _, sys_scores = mod.predict(data, cuda=True, batch_size=256, reference_free=True)
                acc_scores.extend(sys_scores)
         
            acc = {'acc': sum(acc_scores) / len(acc_scores), 'len': len(acc_scores)}
            self.score_lst.append(acc)

        elif dataset_type in ['brl_new', 'brl_final']:
            dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn) # must set batch_size=1 to skip only non-existent images
            pred_lst, gt_lst, acc_lst = [], [], []
            for batch_img_dir, batch_request, batch_gpt_text, batch_label in tqdm(iter(dataloader)):
                if self.model_type in ['clip-s', 'longclip-s', 'pac-s', 'blip-s', 'contextclip-s',
                                       'llava_rlaif-s', 'llava_rlaif-s_inst', 'qwen_rlaif-s', 'qwen_rlaif-s_inst']:
                    pred_scores, gt_scores, acc_score = mod.calculate_corr_acc(batch_img_dir, batch_request, batch_gpt_text, batch_label)
                    pred_lst.extend(pred_scores)
                    gt_lst.extend(gt_scores)
                    acc_lst.append(acc_score)
                    
                elif self.model_type in ['imagereward']:
                    for i in range(len(batch_img_dir)):
                        text = 'Request: ' + batch_request[i] + '\nResponse: ' + batch_gpt_text[i]
                        pred_score = mod.score(text, [batch_img_dir[i]])
                        gt_score = batch_label[i]
                        pred_lst.append(pred_score)
                        gt_lst.append(gt_score)
                        acc_lst.append(pred_score)

                elif self.model_type == 'polos':
                    for i in range(len(batch_img_dir)):
                        gt_score = batch_label[i]
                        gt_lst.append(gt_score.item())

                        text = 'Request: ' + batch_request[i] + '\nResponse: ' + batch_gpt_text[i]
                        image = Image.open(batch_img_dir[i]).convert("RGB")
                        data = [{
                        "mt" : text,
                        "refs": text,
                        "img": image
                        }]
                        _, sys_scores = mod.predict(data, cuda=True, batch_size=1, reference_free=True) 
                        pred_lst.extend(sys_scores)                        

                        new_sims = [int(val * 4) / 4 for val in pred_lst]
                        acc_lst = [1 if s == t else 0 for (s,t) in zip(new_sims, gt_lst)]

            if self.model_type in ['imagereward']:
                print(f'Mean: {sum(pred_lst)/len(pred_lst)}, STD: {np.std(pred_lst)}')
                
                min_reward = min(acc_lst)
                max_reward = max(acc_lst)
                try:
                    new_sims = [int((val - min_reward) / (max_reward - min_reward)) for val in acc_lst]
                    acc_lst = [1 if s == t else 0 for (s,t) in zip(new_sims, gt_lst)]
                except:
                    acc_lst = range(len(pred_lst))

            print(acc_lst[0], pred_lst[0], gt_lst[0])
            assert len(pred_lst) == len(gt_lst) == len(acc_lst)
            kendalltau_b = 100 * scipy.stats.kendalltau(pred_lst, gt_lst, variant='b')[0]
            kendalltau_c = 100 * scipy.stats.kendalltau(pred_lst, gt_lst, variant='c')[0]
            score_dic = {'kendalltau_b' : kendalltau_b, 'kendalltau_c' : kendalltau_c, 'acc': sum(acc_lst) / len(acc_lst), 'len': len(acc_lst)}
            self.score_lst.append(score_dic)

        elif dataset_type == 'imgreward_test':
            target_sample = []
            if self.model_type == 'polos':
                mod.eval()

            if self.model_type in ['imagereward']:
                rew_lst = []
            for data_id, prompt, im_lst, ranking in tqdm(self.test_dataset):
                if self.model_type in ['imagereward', 'blip-s']:
                    _, rewards = mod.inference_rank(prompt, im_lst)
                    print(rewards)
                    rew_lst.extend(rewards)
                elif self.model_type in ['clip-s', 'longclip-s', 'pac-s', 'llava_rlaif-s', 'qwen_rlaif-s_inst', 'llava_rlaif-s_inst', 'qwen_rlaif-s']:
                    rewards = mod.inference_rank(prompt, im_lst)   
                elif self.model_type == 'polos':
                    rewards = []
                    for img_dir in im_lst:
                        data = [{
                            "mt" : prompt,
                            "refs": [prompt],
                            "img": Image.open(img_dir).convert("RGB")
                            }
                        ]
                        _, reward = mod.predict(data, cuda=True, batch_size=1, reference_free=args.reference_free)  
                        rewards.append(reward)
                        
                target_item = {
                    "id": data_id,
                    "prompt": prompt,
                    "rewards": rewards
                }
                target_sample.append(target_item)

            if self.model_type in ['imagereward']:
                print(f'Mean: {sum(rew_lst)/len(rew_lst)}, STD: {np.std(rew_lst)}')
                print(f'Max: {max(rew_lst)}, Min: {min(rew_lst)}')
                    
            test_acc = cal_acc(self.test_dataset.data, target_sample)
            res = {'test_acc' : test_acc}    
            self.score_lst.append(res)
        