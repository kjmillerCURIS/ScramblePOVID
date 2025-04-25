import os
import sys
sys.path.append(os.path.abspath('.'))
import torch
import json
from my_datasets.winoground import get_winoground_scores, Winoground
from collections import OrderedDict
from pathlib import Path

def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)

if __name__ == '__main__':
    model_names = [
        'llava-v1.5-13b',
        'train_coco_syn2_adv_ref',
        'train_coco_syn2_adv_ref_combined',
        'train_coco_syn_swap_v1',
        'train_coco_syn_cot_adv_ref',
    ]
    # ckpt_root_dir = Path('checkpoint/coco_syn/')
    
    for model_name in model_names:
        root_path = f'playground/data/eval/winoground/eval_ans_{model_name}'
        scores_file = f'{root_path}/scores.pt'
        winoground_root_dir = '/projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets'
        winoground_dataset = Winoground(root_dir=winoground_root_dir)
    
        scores = torch.load(scores_file)
        winoground_scores = get_winoground_scores(scores)
        
        incorrect_group_score = []
        incorrect_text_score = []
        incorrect_image_score = []
        for s, w in zip(winoground_scores, winoground_dataset.winoground):
            if not group_correct(s):
                incorrect_group_score.append({
                    'captionA' : w['caption_0'],
                    'captionB' : w['caption_1'],
                })
            if not text_correct(s):
                incorrect_text_score.append({
                    'captionA' : w['caption_0'],
                    'captionB' : w['caption_1'],
                })
            if not image_correct(s):
                incorrect_image_score.append({
                    'captionA' : w['caption_0'],
                    'captionB' : w['caption_1'],
                })
                
        json.dump(incorrect_group_score, open(f'{root_path}/incorrect_group_score.json', 'w'), indent=4)
        json.dump(incorrect_text_score, open(f'{root_path}/incorrect_text_score.json', 'w'), indent=4)
        json.dump(incorrect_image_score, open(f'{root_path}/incorrect_image_score.json', 'w'), indent=4)
