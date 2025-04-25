import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import argparse
import numpy as np

LLAVA_CAP_INSTRS = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented."
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_llava_samples', type=int, default=3000)
    parser.add_argument('--caption_style', type=str, choices=['default','llava'], default='llava', help='Whether to include caption instruction in llava format')
    parser.add_argument('--output_file', type=str, default='data/sft_data/sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000.json')
    args = parser.parse_args()
    
    RNG = np.random.RandomState(44)
    
    sugarcrepe_data_file = 'data/sft_data/sugarcrepe_all_instruct_combined_only_swap.json'
    conv_file_llava = '/projectnb/ivc-ml/samarth/datasets/LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
    
    with open(sugarcrepe_data_file, 'r') as f:
        orig_sugarcrepe_data = json.load(f)
    
    with open(conv_file_llava, 'r') as f:
        convs_llava = json.load(f)

    # NOTE : additional filtering for ocr_vqa images
    convs_llava = [conv for conv in convs_llava if 'image' in conv and conv['image'].startswith('ocr_vqa')]
    
    convs_llava = RNG.choice(convs_llava, args.num_llava_samples, replace=False)
    all_convs = []
    for conv in orig_sugarcrepe_data:
        conv['image'] = 'coco/val2017/' + conv['image']
        
        if args.caption_style == 'llava' and conv['conversations'][0]['value'].endswith('caption:'):
            cap_instr = RNG.choice(LLAVA_CAP_INSTRS) # change the caption instruction
            conv['conversations'][0]['value'] = f'<image>\n{cap_instr}'
        
        all_convs.append(conv)
        
    for conv in convs_llava:
        all_convs.append(conv)
    
    with open(args.output_file, 'w') as f:
        json.dump(all_convs, f, indent=4)
        
    print(f"Saved {len(all_convs)} conversations to {args.output_file}")