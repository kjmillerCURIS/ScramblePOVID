import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import pandas as pd

# root_dir = '/home/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets/winoground'
root_dir = '/projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets/winoground'
info_file_path = os.path.join(root_dir, 'examples.jsonl')
metadata_path = os.path.join(root_dir, 'metadata.csv')
image_dir = os.path.join(root_dir, 'images/')

outpath = 'data/preference_data/winoground_preference_combined.json'

winoground_data = [json.loads(line) for line in open(info_file_path).readlines()]
metadata = pd.read_csv(metadata_path).to_dict(orient='records')

# TODO : Possibly generate examples like yes/no questions
output_json = []
out_id = 0
for i, example in enumerate(winoground_data):
    output_json.append({
        'id' : out_id,
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
    out_id += 1

    output_json.append({
        'id' : out_id,
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
    out_id += 1
    
    # Yes/No question
    output_json.append({
        'id' : out_id,
        'image' : metadata[i]['image_0'],
        'conversations' : [
            {
                'from' : 'human',
                'value' : f'<image>\nDoes the image show \'{example["caption_0"]}\'? Please answer yes or no.'
            },
            {
                'from' : 'gpt',
                'value' : 'Yes'
            }
        ],
        'rejected_conversations' : [
            {
                'from' : 'human',
                'value' : f'<image>\nDoes the image show \'{example["caption_1"]}\'? Please answer yes or no.'
            },
            {
                'from' : 'gpt',
                'value' : 'Yes'
            }
        ]
    })
    
    output_json.append({
        'id' : out_id,
        'image' : metadata[i]['image_1'],
        'conversations' : [
            {
                'from' : 'human',
                'value' : f'<image>\nDoes the image show \'{example["caption_1"]}\'? Please answer yes or no.'
            },
            {
                'from' : 'gpt',
                'value' : 'Yes'
            }
        ],
        'rejected_conversations' : [
            {
                'from' : 'human',
                'value' : f'<image>\nDoes the image show \'{example["caption_0"]}\'? Please answer yes or no.'
            },
            {
                'from' : 'gpt',
                'value' : 'Yes'
            }
        ]
    })

with open(outpath, 'w') as f:
    json.dump(output_json, f, indent=4)
