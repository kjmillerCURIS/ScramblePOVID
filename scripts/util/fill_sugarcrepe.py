import os
import sys
sys.path.append(os.path.abspath('.'))
import json
from collections import defaultdict, deque

if __name__ == '__main__':
    my_neg = defaultdict(deque)
    # with open('data/my_sugarcrepe/neg_cot/combined_caps_len_filtered.json', 'r') as f:
    #     my_neg_caps = json.load(f)
    with open('data/my_sugarcrepe/single_rule_neg_swap_att/adversarial_refine.json', 'r') as f:
        my_neg_caps = json.load(f)
    for cap in my_neg_caps:
        my_neg[cap['img_path']].append(cap)
    with open('data/my_sugarcrepe/single_rule_neg_swap_obj/adversarial_refine.json', 'r') as f:
        my_neg_caps = json.load(f)
    for cap in my_neg_caps:
        my_neg[cap['img_path']].append(cap)
    
    
    with open('playground/dev/sugarcrepe_swap_pos_caps.json', 'r') as f:
        sugarcrepe_swap = json.load(f)
    
    final_neg_caps = []
    num_my_neg = 0
    num_sugarcrepe_neg = 0
    for cap in sugarcrepe_swap:
        if cap['img_path'] in my_neg and len(my_neg[cap['img_path']]) > 0:
            final_neg_caps.append(my_neg[cap['img_path']].popleft())
            num_my_neg += 1
        else:
            cap['neg_caption'] = cap['sugarcrepe_neg']
            del cap['sugarcrepe_neg']
            final_neg_caps.append(cap)
            num_sugarcrepe_neg += 1
    
    print('Neg examples from our generated negatives:', num_my_neg)
    print('Neg examples from sugarcrepe:', num_sugarcrepe_neg)
    
    with open('data/my_sugarcrepe/swap_v1_adv_ref_filled.json', 'w') as f:
        json.dump(final_neg_caps, f, indent=4)
        