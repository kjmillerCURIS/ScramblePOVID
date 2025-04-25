import os
import sys
sys.path.append(os.path.abspath('.'))

import torch
from typing import Any, Tuple
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
from pathlib import Path
import json
import open_clip
from tqdm import tqdm


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

coco_root = Path('/projectnb/ivc-ml/array/data/COCO')
coco_img_dir = coco_root / 'images' / 'train2017'
dset = CocoDataset(coco_img_dir, coco_root / 'annotations/captions_train2017.json')
loader = DataLoader(dset, batch_size=256, shuffle=False, num_workers=3, collate_fn=lambda x: tuple(zip(*x)))

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-CLIPA', pretrained='datacomp1b')
tokenizer = open_clip.get_tokenizer('ViT-L-14-CLIPA')
model.to('cuda')
model.eval()

winoground_eg_captions = [
    'I had cleaned my car'
    'a bottle is in water',
    'there are three bananas and two apples',
    'the happy person is on the right and the sad person is on the left',
    'that person dusting off their hands',
    'the red car is behind the blue car'
]

winoground_eg_tokens = tokenizer(winoground_eg_captions).to('cuda')
winoground_eg_embeddings = model.encode_text(winoground_eg_tokens)


all_winoground_sims = []
with torch.inference_mode():
    for (indices, _, captions) in tqdm(loader, total=len(loader)):
        idxs = [(idx, cap_idx) for i, idx in enumerate(indices) for cap_idx in range(len(captions[i]))]
        captions = [caption for sublist in captions for caption in sublist]
        caption_embeddings = model.encode_text(tokenizer(captions).to('cuda'))
        
        caption_winoground_sims = caption_embeddings @ winoground_eg_embeddings.t()
        caption_winoground_sims = caption_winoground_sims.max(dim=1).values 
        
        # all_dset_idxs.extend(idxs)
        all_winoground_sims.append(caption_winoground_sims)

all_winoground_sims = torch.cat(all_winoground_sims)

clip_sims = torch.load('playground/dev/coco_clip_sims.pt')
all_sims = [[] for _ in range(len(dset))]
# rearrange into image x captions. keep only the best caption and then take top 10k
for i, (img_cap_sim, winoground_sim, (img_idx, cap_idx)) in enumerate(zip(clip_sims['sims'], all_winoground_sims, clip_sims['idxs'])):
    all_sims[img_idx].append((img_cap_sim.item(), winoground_sim.item(), img_idx, cap_idx))

for i in range(len(dset)):
    all_sims[i] = max(all_sims[i], key=lambda x: x[0]) # NOTE : Keeping the caption most similar to the image still

# Get best winoground sims
all_sims = sorted(all_sims, key=lambda x: x[1], reverse=True)
all_sims = all_sims[:10000]

# get these imgs and captions
img_cap_set = [{
    'img_path' : dset.coco.loadImgs(dset.ids[i])[0]["file_name"], 
    'caption' : dset.coco.loadAnns(dset.coco.getAnnIds(dset.ids[i]))[ci]['caption'],
    'img_cap_clip_sim' : sim,
    'winoground_sim' : winoground_sim} for sim, winoground_sim, i, ci in all_sims]

with open('playground/dev/coco_10k_caps_train2.json', 'w') as f:
    json.dump(img_cap_set, f, indent=4)