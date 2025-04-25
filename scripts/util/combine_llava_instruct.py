import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_llava_samples', type=int, default=3000)
    parser.add_argument('--output_file', type=str, default='data/preference_data/sugarcrepe_all_preference_combined_only_swap_mix_llava_ocr_vqa_3000.json')
    args = parser.parse_args()
    
    RNG = np.random.RandomState(44)
    
    pref_data_file = 'data/preference_data/sugarcrepe_all_preference_combined_only_swap.json'
    conv_file_llava = '/projectnb/ivc-ml/samarth/datasets/LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
    
    with open(pref_data_file, 'r') as f:
        orig_pref_data = json.load(f)
    
    with open(conv_file_llava, 'r') as f:
        convs_llava = json.load(f)

    # NOTE : additional filtering for ocr_vqa images
    convs_llava = [conv for conv in convs_llava if 'image' in conv and conv['image'].startswith('ocr_vqa')]
    
    convs_llava = RNG.choice(convs_llava, args.num_llava_samples, replace=False)
    all_convs = []
    for conv in orig_pref_data:
        conv['image'] = 'coco/val2017/' + conv['image']
        conv['wt_rejected'] = 1.
        all_convs.append(conv)
        
    for conv in convs_llava:
        conv['rejected_conversations'] = [{'from': 'human', 'value': '<image>\nignore'}, {'from': 'gpt', 'value': 'ignore'}]
        conv['wt_rejected'] = 0.
        all_convs.append(conv)
    
    with open(args.output_file, 'w') as f:
        json.dump(all_convs, f, indent=4)
        
    print(f"Saved {len(all_convs)} conversations to {args.output_file}")