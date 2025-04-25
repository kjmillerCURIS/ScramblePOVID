import os
import json
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets.folder import default_loader

class SugarCrepe(Dataset):

    def __init__(self, root_dir, data_split='add_obj', image_preprocess=None):
        self.image_dir = str(Path(root_dir) / 'val2017')
        self.ann = json.load(open(Path(root_dir) / f'{data_split}.json', 'r'))
        self.image_preprocess = image_preprocess
        self.idx_strings = list(self.ann.keys()) # NOTE : indices may be non-contiguous
        self.image_loader = default_loader

    def get_item_path(self, idx):
        idx_str = self.idx_strings[idx]
        caption = self.ann[idx_str]['caption']
        negative_caption = self.ann[idx_str]['negative_caption']
        return os.path.join(self.image_dir, self.ann[idx_str]['filename']), [caption, negative_caption]
    
    def __getitem__(self, idx):
        idx_str = self.idx_strings[idx]
        data = self.ann[idx_str]
        img = self.image_loader(os.path.join(self.image_dir, data['filename']))
        if self.image_preprocess is not None:
            img = self.image_preprocess(img)
        caption = data['caption']
        negative_caption = data['negative_caption']
        return {
            'images' : [img],
            'texts' : [caption, negative_caption]
        }

    def __len__(self):
        return len(self.ann)
    
    def evaluate_scores(self, scores):
        scores = scores.squeeze()
        preds = scores.argmax(dim=1)
        acc = (preds == 0).sum().div(len(preds))
        return {'top1_acc' : acc.item()}