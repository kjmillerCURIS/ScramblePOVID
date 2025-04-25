import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--root_dir', type=str, default='/home/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets/sugar_crepe')
parser.add_argument('--root_dir', type=str, default='/projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets/sugar_crepe')
parser.add_argument('--outpath', type=str, default='data/sugarcrepe_all_preference_combined.json')
parser.add_argument('--caption', action='store_true', default=False, help='Whether to include caption data')
parser.add_argument('--yesno', action='store_true', default=False, help='Whether to include questions with the answer yes/no')

args = parser.parse_args()

assert args.caption or args.yes, 'Either caption or yes must be true'

# ALL_SPLITS = [
#     'add_obj', 'add_att', 'replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att']

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
                ]
            })
        overall_count += 1

with open(outpath, 'w') as f:
    json.dump(output_json, f, indent=4)
