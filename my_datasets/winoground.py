import os
import json
import subprocess
import pandas as pd

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

def get_winoground_scores(scores_i2t):
    ids = list(range(scores_i2t.shape[0]))
    winoground_scores = []
    for id, score_i2t in zip(ids, scores_i2t):
        winoground_scores.append({
            "id" : id,
            "c0_i0": score_i2t[0][0],
            "c0_i1": score_i2t[1][0],
            "c1_i0": score_i2t[0][1],
            "c1_i1": score_i2t[1][1]}
        )
    return winoground_scores

def get_winoground_acc(scores):
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)
    
    for result in scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(scores)
    result = {
        'text': text_correct_count/denominator,
        'image': image_correct_count/denominator,
        'group': group_correct_count/denominator,
    }
    return result

class Winoground(Dataset):
    def __init__(self, image_preprocess=None, root_dir='./', return_image_paths=True):
        self.root_dir = os.path.join(root_dir, "winoground")
        if not os.path.exists(self.root_dir):
            subprocess.call(
                ["gdown", "--no-cookies", "1Lril_90vjsbL_2qOaxMu3I-aPpckCDiF", "--output",
                 os.path.join(root_dir, "winoground.zip")]
            )
            subprocess.call(
                ["unzip", "-q", "winoground.zip"],
                cwd=root_dir
            )
        csv_file = os.path.join(self.root_dir, 'metadata.csv')
        self.metadata = pd.read_csv(csv_file).to_dict(orient='records')
        json_file = os.path.join(self.root_dir, 'examples.jsonl')
        self.winoground = [json.loads(line) for line in open(json_file, 'r')]
        self.return_image_paths = return_image_paths
        if return_image_paths:
            assert image_preprocess is None
            self.preprocess = None
            
        self.preprocess = image_preprocess
        self.original_tags = self.get_original_tags()
        self.new_tags = self.get_new_tags(path=os.path.join(self.root_dir, "why_winoground_hard.json"))
        self.image_loader = default_loader
    
    def __len__(self):
        return len(self.winoground)

    def __getitem__(self, idx):
        assert self.metadata[idx]['id'] == idx
        image_0_path = os.path.join(self.root_dir, self.metadata[idx]['image_0'])
        image_1_path = os.path.join(self.root_dir, self.metadata[idx]['image_1'])
        if self.return_image_paths:
            image_0 = image_0_path
            image_1 = image_1_path
        else:
            image_0 = self.preprocess(self.image_loader(image_0_path))
            image_1 = self.preprocess(self.image_loader(image_1_path))
        
        caption_0 = self.metadata[idx]['caption_0']
        caption_1 = self.metadata[idx]['caption_1']
        item = {
            "images": [image_0, image_1],
            "texts": [caption_0, caption_1]
        }
        return item
    
    def get_original_tags(self):
        tags = {}
        for example in self.winoground:
            if example['num_main_preds'] == 1:
                if '1 Main Pred' not in tags:
                    tags["1 Main Pred"] = []
                tags['1 Main Pred'].append(example["id"])
            elif example['num_main_preds'] == 2:
                if '2 Main Pred' not in tags:
                    tags["2 Main Pred"] = []
                tags['2 Main Pred'].append(example["id"])
            else:
                # This won't happen
                raise ValueError(f"num_main_preds: {example['num_main_preds']}")
            if example["collapsed_tag"] not in tags:
                tags[example["collapsed_tag"]] = []
            tags[example["collapsed_tag"]].append(example["id"])
        return tags

    def get_new_tags(self, path="./why_winoground_hard.json"):
        new_tag_dict = json.load(open(path))
        tags = {}
        for idx in new_tag_dict:
            curr_tags = new_tag_dict[idx]
            if len(curr_tags) == 0:
                if "No Tag" not in tags:
                    tags["No Tag"] = []
                tags["No Tag"].append(int(idx))
            for tag in curr_tags:
                if tag not in tags:
                    tags[tag] = []
                tags[tag].append(int(idx))
        return tags
    
    
    def evaluate_scores(self, scores):
        winoground_scores = get_winoground_scores(scores)
        acc = get_winoground_acc(winoground_scores)
        print("Winoground performance (overall)")
        print(f"{'Dataset': <70} {'Text': <10} {'Image': <10} {'Group': <10}")
        print(f"{'Winoground': <70} {acc['text']: <10.2%} {acc['image']: <10.2%} {acc['group']: <10.2%}")
        results = {}
        results['all'] = acc
        
        
        # for tag in self.original_tags:
        #     results[tag] = get_winoground_acc([winoground_scores[i] for i in self.original_tags[tag]])
        # for tag in self.new_tags:
        #     results[tag] = get_winoground_acc([winoground_scores[i] for i in self.new_tags[tag]])
            
            # print(f"Winoground {tag} text score: {results[tag]['text']}")
            # print(f"Winoground {tag} image score: {results[tag]['image']}")
            # print(f"Winoground {tag} group score: {results[tag]['group']}")
        return results, acc['group']
    
    
