import os
import sys
sys.path.append(os.path.abspath('.'))
import json

from transformers import AutoTokenizer


if __name__ == '__main__':
    inp_caps = json.load(open('data/coco_syn/coco_syn_swap_v1.json', 'r'))
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
    
    print('Length before filtering: ', len(inp_caps))
    out_caps = []
    for cap in inp_caps:
        if len(tokenizer.encode(cap['neg_caption'])) < 50:
            out_caps.append(cap)
    
    print('Length after filtering: ', len(out_caps))
    json.dump(out_caps, open('data/coco_syn/coco_syn_swap_v1_short.json', 'w'), indent=4)