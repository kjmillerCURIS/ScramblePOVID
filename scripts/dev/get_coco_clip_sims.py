import os
import sys
from typing import Any, Tuple
sys.path.append(os.path.abspath('.'))
import open_clip
from torchvision.datasets import CocoCaptions
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
import json

# class CocoDataset(CocoCaptions):
#     def __init__(self, img_dir, ann_path, transform=None):
#         super().__init__(img_dir, ann_path)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.image_loader = default_loader
#         self.annotations = json.load(open(ann_path, 'r'))
        
        
#     def __len__(self):
#         return len(self.ann_path)
    
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.annotations[idx]['file_name'])
#         img = self.image_loader(img_path)
#         if self.transform:
#             img = self.transform(img)
#         return img

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
        
def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    # coco_root = '/projectnb/ivc-ml/array/data/COCO'
    coco_root = '/projectnb/ivc-ml/samarth/datasets/COCO'
    coco_root = Path(coco_root)
    coco_img_dir = coco_root / 'images' / 'val2017'
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-CLIPA', pretrained='datacomp1b')
    tokenizer = open_clip.get_tokenizer('ViT-L-14-CLIPA')
    model.to('cuda')
    model.eval()
    
    dataset = CocoDataset(coco_img_dir, coco_root / 'annotations/captions_val2017.json', transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=3)
    
    all_sims = []
    all_dset_idxs = []
    
    with torch.inference_mode():
        for (indices, imgs, captions) in tqdm(dataloader, total=len(dataloader)):
            imgs = torch.stack(imgs).to('cuda')
            img_embeddings = model.encode_image(imgs)
            img_embeddings = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
            
            img_embeddings = torch.cat([img_embedding.repeat(len(sublist), 1) for img_embedding, sublist in zip(img_embeddings, captions)], dim=0)
            idxs = [(idx, cap_idx) for i, idx in enumerate(indices) for cap_idx in range(len(captions[i]))]
            captions = [caption for sublist in captions for caption in sublist]
            
            text_tokens = tokenizer(captions).to('cuda') # (5xB) x T
            text_embeddings = model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            sims = torch.einsum('ij,ij->i', img_embeddings, text_embeddings)
            all_sims.append(sims.cpu())
            all_dset_idxs.append(torch.tensor(idxs))
            
    all_sims = torch.cat(all_sims, dim=0)
    all_dset_idxs = torch.cat(all_dset_idxs, dim=0)
    torch.save(
        {'sims': all_sims, 'idxs': all_dset_idxs}, 
        'playground/dev/coco_clip_sims_val.pt')