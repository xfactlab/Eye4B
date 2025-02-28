import os
import urllib
import random
import numpy as np
import torch
from termcolor import colored
import torch
from huggingface_hub import hf_hub_download


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def rprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'white', 'on_red', attrs=["bold"]))

def yprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'yellow',attrs=["bold"]))

def gprint(*args):
    combined_text = ' '.join(map(str, args))
    print(colored(combined_text, 'green',attrs=["bold"]))

def collate_fn(batch):
    if isinstance(batch, tuple) and isinstance(batch[0], list):
        return batch
    elif isinstance(batch, list):
        transposed = list(zip(*batch))
        return [collate_fn(samples) for samples in transposed]
    return torch.utils.data.default_collate(batch)

def cal_acc(score_sample, target_sample):
    
    tol_cnt = 0.
    true_cnt = 0.
    for idx in range(len(score_sample)):
        item_base = score_sample[idx]["ranking"]
        item = target_sample[idx]["rewards"]
        for i in range(len(item_base)):
            for j in range(i+1, len(item_base)):
                if item_base[i] > item_base[j]:
                    if item[i] >= item[j]:
                        tol_cnt += 1
                    elif item[i] < item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                elif item_base[i] < item_base[j]:
                    if item[i] > item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                    elif item[i] <= item[j]:
                        tol_cnt += 1
    
    return true_cnt / tol_cnt


SCORES = {
    "CLIP": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "BLIP": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth",
    "Aesthetic": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
}

def download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target
    
def ImageReward_download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)
    hf_hub_download(repo_id="THUDM/ImageReward", filename=filename, local_dir=root)
    return download_target


