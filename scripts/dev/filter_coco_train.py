import os
import sys
sys.path.append(os.path.abspath('.'))

import torch
from typing import Any, Tuple
from torchvision.datasets import CocoCaptions
from pathlib import Path
import json


class CocoDataset(CocoCaptions):
    def __init__(self, img_dir, ann_path, transform=None):
        super().__init__(img_dir, ann_path, transform=transform)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return index, image, target

# coco_root = Path('/projectnb/ivc-ml/array/data/COCO')
coco_root = Path('/projectnb/ivc-ml/samarth/datasets/COCO')
coco_img_dir = coco_root / 'images' / 'val2017'
dset = CocoDataset(coco_img_dir, coco_root / 'annotations/captions_val2017.json')

clip_sims = torch.load('playground/dev/coco_clip_sims_val.pt')


img_cap_sims = [[] for _ in range(len(dset))]
# rearrange into image x captions. keep only the best caption and then take top 10k
for i, (sim, (img_idx, cap_idx)) in enumerate(zip(clip_sims['sims'], clip_sims['idxs'])):
    img_cap_sims[img_idx].append((sim.item(), img_idx, cap_idx))

for i in range(len(dset)):
    img_cap_sims[i] = max(img_cap_sims[i], key=lambda x: x[0])

# img_cap_sims = sorted(img_cap_sims, key=lambda x: x[0], reverse=True)
# img_cap_sims = img_cap_sims[:10000]

# get these imgs and captions
img_cap_set = [{
    'img_path' : dset.coco.loadImgs(dset.ids[i])[0]["file_name"], 
    'caption' : dset.coco.loadAnns(dset.coco.getAnnIds(dset.ids[i]))[ci]['caption'],
    'img_cap_clip_sim' : sim} for sim, i, ci in img_cap_sims]

with open('playground/dev/coco_val_caps.json', 'w') as f:
    json.dump(img_cap_set, f, indent=4)