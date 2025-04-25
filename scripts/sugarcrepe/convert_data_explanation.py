import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='/projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe')
parser.add_argument('--outpath', type=str, default='data/preference_data/sugarcrepe_only_swap_combined_w_explanation.json')
parser.add_argument('--caption', action='store_true', default=False, help='Whether to include caption preference')
parser.add_argument('--yesno', action='store_true', default=False, help='Whether to include yes/no preference')

args = parser.parse_args()

assert args.caption or args.yesno, 'Either caption or yesno must be true'
lower_first = lambda s: s[:1].lower() + s[1:] if s else ''

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
            output_json.append({
                'id' : overall_count,
                'image' : v['filename'],
                'conversations' : [
                    {
                        'from' : 'human',
                        'value' : '<image>\ncaption:'
                    },
                    {
                        'from' : 'gpt',
                        'value' : v['caption']
                    }
                ],
                'rejected_conversations' : [
                    {
                        'from' : 'human',
                        'value' : '<image>\ncaption:'
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
                        'value' : f'<image>\nDoes the image show \'{v["caption"]}\'?'
                    },
                    {
                        'from' : 'gpt',
                        'value' : f'Yes, I see {lower_first(v["caption"])}'
                    }
                ],
                'rejected_conversations' : [
                    {
                        'from' : 'human',
                        'value' : f'<image>\nDoes the image show \'{v["caption"]}\'?'
                    },
                    {
                        'from' : 'gpt',
                        'value' : f'No, I see {lower_first(v["negative_caption"])}'
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
                        'value' : f'<image>\nDoes the image show \'{v["negative_caption"]}\'?'
                    },
                    {
                        'from' : 'gpt',
                        'value' : f'No, I see {lower_first(v["caption"])}'
                    }
                ],
                'rejected_conversations' : [
                    {
                        'from' : 'human',
                        'value' : f'<image>\nDoes the image show \'{v["negative_caption"]}\'?'
                    },
                    {
                        'from' : 'gpt',
                        'value' : f'Yes, I see {lower_first(v["negative_caption"])}'
                    }
                ]
            })
            overall_count += 1

with open(outpath, 'w') as f:
    json.dump(output_json, f, indent=4)
