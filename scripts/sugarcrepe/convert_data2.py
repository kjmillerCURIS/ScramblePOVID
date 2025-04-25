import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import pandas as pd

import argparse

import numpy as np
RNG = np.random.RandomState(44)

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


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='/projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe')
parser.add_argument('--outpath', type=str, default='data/sugarcrepe_all_preference_combined.json')
parser.add_argument('--caption', action='store_true', default=False, help='Whether to include caption preference')
parser.add_argument('--caption_style', type=str, choices=['default','llava'], default='default', help='Whether to include caption instruction in llava format')
parser.add_argument('--yesno', action='store_true', default=False, help='Whether to include yes/no preference')

args = parser.parse_args()

assert args.caption or args.yesno, 'Either caption or yesno must be true'

# ALL_SPLITS = [
#     'add_obj', 'add_att', 'replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att']

# ALL_SPLITS = [
#     'add_obj', 'add_att', 'replace_obj', 'replace_att', 'replace_rel']

ALL_SPLITS = [
    'swap_obj', 'swap_att']

root_dir = args.root_dir
outpath = args.outpath
all_data = {}

for split in ALL_SPLITS:
    with open(os.path.join(root_dir, f'{split}.json'), 'r') as file:
        all_data[split] = json.load(file)
        
output_json = []
overall_count = 0
for split in ALL_SPLITS:
    for k, v in all_data[split].items():
        if args.caption:
            if args.caption_style == 'llava':
                cap_instr = RNG.choice(LLAVA_CAP_INSTRS)
            else:
                cap_instr = 'caption:'
        
            output_json.append({
                'id' : overall_count,
                'image' : v['filename'],
                'conversations' : [
                    {
                        'from' : 'human',
                        'value' : f'<image>\n{cap_instr}'
                    },
                    {
                        'from' : 'gpt',
                        'value' : v['caption']
                    }
                ],
                'rejected_conversations' : [
                    {
                        'from' : 'human',
                        'value' : f'<image>\n{cap_instr}'
                    },
                    {
                        'from' : 'gpt',
                        'value' : v['negative_caption']
                    }
                ]
            })
            overall_count += 1
            
            
        if args.yesno:
            output_json.append({
                'id' : overall_count,
                'image' : v['filename'],
                'conversations' : [
                    {
                        'from' : 'human',
                        'value' : f'<image>\nDoes the image show \'{v["caption"]}\'? Please answer yes or no.'
                    },
                    {
                        'from' : 'gpt',
                        'value' : 'Yes'
                    }
                ],
                'rejected_conversations' : [
                    {
                        'from' : 'human',
                        'value' : f'<image>\nDoes the image show \'{v["caption"]}\'? Please answer yes or no.'
                    },
                    {
                        'from' : 'gpt',
                        'value' : 'No'
                    }
                ]
            })
            overall_count += 1
            output_json.append({
                'id' : overall_count,
                'image' : v['filename'],
                'conversations' : [
                    {
                        'from' : 'human',
                        'value' : f'<image>\nDoes the image show \'{v["negative_caption"]}\'? Please answer yes or no.'
                    },
                    {
                        'from' : 'gpt',
                        'value' : 'No'
                    }
                ],
                'rejected_conversations' : [
                    {
                        'from' : 'human',
                        'value' : f'<image>\nDoes the image show \'{v["negative_caption"]}\'? Please answer yes or no.'
                    },
                    {
                        'from' : 'gpt',
                        'value' : 'Yes'
                    }
                ]
            })
            overall_count += 1

with open(outpath, 'w') as f:
    json.dump(output_json, f, indent=4)
