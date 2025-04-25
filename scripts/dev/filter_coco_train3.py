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
import numpy as np
from fast_pytorch_kmeans import KMeans


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

# all_idxs = []
# all_cap_embeds = []
# with torch.inference_mode():
#     for (indices, _, captions) in tqdm(loader, total=len(loader)):
#         idxs = [(idx, cap_idx) for i, idx in enumerate(indices) for cap_idx in range(len(captions[i]))]
#         captions = [caption for sublist in captions for caption in sublist]
#         caption_embeddings = model.encode_text(tokenizer(captions).to('cuda'))
#         all_cap_embeds.append(caption_embeddings)
#         all_idxs.extend(idxs)

# all_cap_embeds = torch.cat(all_cap_embeds)

# torch.save({
#     'all_idxs' : all_idxs,
#     'all_cap_embeds' : all_cap_embeds.cpu()
# }, 'playground/dev/all_cap_embeds.pt')

ckpt = torch.load('playground/dev/all_cap_embeds.pt')
all_idxs = ckpt['all_idxs']
all_cap_embeds = ckpt['all_cap_embeds'].to('cuda')

print('Starting k-means')
# Perform k-means clustering
num_clusters = 1000  # You can adjust the number of clusters as needed
cluster_assignments = KMeans(n_clusters=num_clusters, max_iter=100, verbose=2, init_method='kmeans++').fit_predict(all_cap_embeds)
cluster_assignments = cluster_assignments.cpu().numpy()
print('k-means done')

# TODO : apply some quality filtering as well
RNG = np.random.RandomState(44)
print('Selecting examples')
selected_indices = []
for cluster_id in tqdm(range(num_clusters), total=num_clusters):
    cluster_indices = np.where(cluster_assignments == cluster_id)[0]
    if len(cluster_indices) > 25:
        selected_indices.extend(RNG.choice(cluster_indices, 25, replace=False))
    else:
        selected_indices.extend(cluster_indices)
    
cap_idxs = [all_idxs[idx] for idx in selected_indices]

# get these imgs and captions
img_cap_set = [{
    'img_path' : dset.coco.loadImgs(dset.ids[i])[0]["file_name"], 
    'caption' : dset.coco.loadAnns(dset.coco.getAnnIds(dset.ids[i]))[ci]['caption']} for i, ci in cap_idxs]
img_cap_set = RNG.permutation(img_cap_set).tolist() # so that viewing doesn't seem like all captions are similar

with open('playground/dev/coco_25k_caps_train3.json', 'w') as f:
    json.dump(img_cap_set, f, indent=4)