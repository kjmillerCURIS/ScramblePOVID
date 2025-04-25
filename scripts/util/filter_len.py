import os
import sys
sys.path.append(os.path.abspath('.'))
import json
from pathlib import Path
from transformers import AutoTokenizer

if __name__ == '__main__':
    # inp_caps = json.load(open('data/coco_syn/coco_single_rule_neg_swap_att/combined_caps_w_scores.json', 'r'))
    # inp_caps = json.load(open('data/coco_syn/coco_caps_neg_cot/combined_caps.json', 'r'))
    inp_root_dir = 'data/coco_train_syn/coco_caps_neg_cot'
    inp_caps = json.load(open(Path(inp_root_dir) / 'combined_caps_w_scores.json', 'r'))
    # inp_caps = json.load(open('data/coco_syn/coco_syn_swap_v1.json', 'r'))
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
    
    print('Length before filtering: ', len(inp_caps))
    out_caps = []
    for cap in inp_caps:
        cap_len = len(tokenizer.encode(cap['caption']))
        neg_cap_len = len(tokenizer.encode(cap['neg_caption']))
        min_len = 0.7 * cap_len
        max_len = 1.2 * cap_len
        if (neg_cap_len >= min_len) and (neg_cap_len <= max_len):
            out_caps.append(cap)
    print('Length after filtering: ', len(out_caps))
    # json.dump(out_caps, open('data/coco_syn/coco_syn_swap_v1_len_filtered.json', 'w'), indent=4)
    # json.dump(out_caps, open('data/coco_syn/coco_single_rule_neg_swap_att/combined_caps_w_scores_len_filtered.json', 'w'))
    json.dump(out_caps, open(Path(inp_root_dir) / 'combined_caps_w_scores_len_filtered.json', 'w'), indent=4)