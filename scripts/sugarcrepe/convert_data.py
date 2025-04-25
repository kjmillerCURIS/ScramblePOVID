import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import pandas as pd


ALL_SPLITS = [
    'add_obj', 'add_att', 'replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att']

root_dir = '/home/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets/sugar_crepe'
all_data = {}

for split in ALL_SPLITS:
    with open(os.path.join(root_dir, f'{split}.json'), 'r') as file:
        all_data[split] = json.load(file)
        
output_json = []
overall_count = 0
for split in ALL_SPLITS:
    for k, v in all_data[split].items():
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

outpath = 'data/sugarcrepe_all_preference.json'
with open(outpath, 'w') as f:
    json.dump(output_json, f, indent=4)
