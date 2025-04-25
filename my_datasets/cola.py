import os
import json
import subprocess
import pandas as pd

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from my_datasets.winoground import get_winoground_scores, get_winoground_acc

class COLA(Dataset):
    def __init__(self, image_preprocess=None, root_dir='./', return_image_paths=True):
        # TODO : add auto download
        self.root_dir = os.path.join(root_dir, "cola")
        self.data = json.load(open(os.path.join(self.root_dir, "multi_object.json"), 'r'))
        self.return_image_paths = return_image_paths
        self.transform = image_preprocess
        if self.return_image_paths:
            assert self.transform is None, "Cannot return image paths and apply transforms"
        self.image_loader = default_loader
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        image_0_path = os.path.join(self.root_dir, item['image_0'])
        image_1_path = os.path.join(self.root_dir, item['image_1'])
        if self.return_image_paths:
            image_0 = image_0_path
            image_1 = image_1_path
        else:
            image_0 = self.transform(self.image_loader(image_0_path))
            image_1 = self.transform(self.image_loader(image_1_path))
        
        return {"images": [image_0, image_1], "texts": [item['caption_0'], item['caption_1']]}
    
    def evaluate_scores(self, scores):
        winoground_scores = get_winoground_scores(scores)
        acc = get_winoground_acc(winoground_scores)
        results = {'all': acc}
        
        print("COLA performance (overall)")
        print(f"{'Dataset': <70} {'Text': <10} {'Image': <10} {'Group': <10}")
        print(f"{'COLA': <70} {acc['text']: <10.2%} {acc['image']: <10.2%} {acc['group']: <10.2%}")
        
        return results, acc['group']