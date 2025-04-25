import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train_pct', type=float, required=True)
args = parser.parse_args()

root_dir = '/home/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets/winoground'
info_file_path = os.path.join(root_dir, 'examples.jsonl')
metadata_path = os.path.join(root_dir, 'metadata.csv')
image_dir = os.path.join(root_dir, 'images/')

out_dir = f'data/winoground_splits/'
os.makedirs(out_dir, exist_ok=True)


winoground_data = [json.loads(line) for line in open(info_file_path).readlines()]
metadata = pd.read_csv(metadata_path).to_dict(orient='records')

RNG = np.random.RandomState(42)
idxs = RNG.permutation(len(winoground_data))

for split_type in ['train', 'val']:
    outfile_path = os.path.join(out_dir, f'winoground_preference_train_pct_{args.train_pct}_{split_type}.json')
    if split_type == 'train':
        subset_idxs = idxs[:int(args.train_pct*len(winoground_data))]
    else:
        subset_idxs = idxs[int(args.train_pct*len(winoground_data)):]
        
    output_json = []
    for i, example in enumerate(winoground_data):
        if i not in subset_idxs:
            continue
        output_json.append({
            'id' : 2*example['id'],
            'image' : metadata[i]['image_0'],
            'conversations' : [
                {
                    'from' : 'human',
                    'value' : '<image>\ncaption:'
                },
                {
                    'from' : 'gpt',
                    'value' : example['caption_0']
                }
            ],
            'rejected_conversations' : [
                {
                    'from' : 'human',
                    'value' : '<image>\ncaption:'
                },
                {
                    'from' : 'gpt',
                    'value' : example['caption_1']
                }
            ]
        })
        
        output_json.append({
            'id' : 2*example['id'] + 1,
            'image' : metadata[i]['image_1'],
            'conversations' : [
                {
                    'from' : 'human',
                    'value' : '<image>\ncaption:'
                },
                {
                    'from' : 'gpt',
                    'value' : example['caption_1']
                }
            ],
            'rejected_conversations' : [
                {
                    'from' : 'human',
                    'value' : '<image>\ncaption:'
                },
                {
                    'from' : 'gpt',
                    'value' : example['caption_0']
                }
            ]
        })

    with open(outfile_path, 'w') as f:
        json.dump(output_json, f, indent=4)
