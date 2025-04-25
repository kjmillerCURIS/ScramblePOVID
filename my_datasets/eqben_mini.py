import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import subprocess
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from my_datasets.winoground import get_winoground_scores, get_winoground_acc


def image_loader(image_path):
    if image_path.split('.')[-1] == 'npy':
        return Image.fromarray(np.load(image_path)[:, :, [2, 1, 0]], 'RGB')
    else:
        return Image.open(image_path).convert("RGB")

class EqBen_Mini(Dataset):
    def __init__(self, image_preprocess=None, root_dir='./', return_image_paths=True):
        self.preprocess = image_preprocess
        
        self.root_dir = os.path.join(root_dir, "eqben_vllm")
        if not os.path.exists(self.root_dir):
            # https://drive.google.com/file/d/11YUTf06uzRHtFV8rYi96z4vTPi8_GNEM/view?usp=sharing
            os.makedirs(self.root_dir, exist_ok=True)
            subprocess.call(
                ["gdown", "--no-cookies", "11YUTf06uzRHtFV8rYi96z4vTPi8_GNEM", "--output", 
                 os.path.join(self.root_dir, "eqben_vllm.zip")]
            )
            subprocess.call(["unzip", "-q", "eqben_vllm.zip"], cwd=self.root_dir)
            
        self.root_dir = os.path.join(root_dir, "eqben_vllm", "images")
        self.subset_types = {
            'eqbensd': ['eqbensd'],
            'eqbenk': ['eqbenkubric_cnt', 'eqbenkubric_loc', 'eqbenkubric_attr'],
            'eqbeng': ['eqbengebc'],
            'eqbenag': ['eqbenag'],
            'eqbeny': ['eqbenyoucook2'],
        }
        json_file = os.path.join(root_dir, "eqben_vllm", "all_select.json")
        self.metadata = json.load(open(json_file, 'r'))
        self.subset_indices = {subset_type: [] for subset_type in self.subset_types}
        for item_idx, item in enumerate(self.metadata):
            image_path = item['image0']
            for subset_type in self.subset_types:
                if image_path.split('/')[0] in self.subset_types[subset_type]:
                    self.subset_indices[subset_type].append(item_idx)
                    break
        
        self.return_image_paths = return_image_paths
        self.transform = image_preprocess
        if self.return_image_paths:
            assert self.transform is None, "Cannot return image paths and apply transforms"
        self.image_loader = image_loader
     
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        image_0_path = os.path.join(self.root_dir, self.metadata[index]['image0'])
        image_1_path = os.path.join(self.root_dir, self.metadata[index]['image1'])
        if self.return_image_paths:
            image_0 = image_0_path
            image_1 = image_1_path
        else:
            image_0 = self.transform(self.image_loader(image_0_path))
            image_1 = self.transform(self.image_loader(image_1_path))
        
        caption_0 = self.metadata[index]['caption0']
        caption_1 = self.metadata[index]['caption1']
        item = {"images": [image_0, image_1], "texts": [caption_0, caption_1]}
        return item
    
    def evaluate_scores(self, scores):
        winoground_scores = get_winoground_scores(scores)
        acc = get_winoground_acc(winoground_scores)
        print("EQBen_Mini performance (overall)")
        print(f"{'Dataset': <70} {'Text': <10} {'Image': <10} {'Group': <10}")
        print(f"{'EQBen_Mini': <70} {acc['text']: <10.2%} {acc['image']: <10.2%} {acc['group']: <10.2%}")
        results = {}
        results['all'] = acc
        for subset_type in self.subset_types:
            subset_indices = self.subset_indices[subset_type]
            subset_scores = [winoground_scores[idx] for idx in subset_indices]
            subset_acc = get_winoground_acc(subset_scores)
            print(f"{'EQBen_Mini ' + subset_type: <70} {subset_acc['text']: <10.2%} {subset_acc['image']: <10.2%} {subset_acc['group']: <10.2%}")
            results[subset_type] = subset_acc
        return results, acc['group']