import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inpath', type=str, default='data/coco_syn/coco_caps_neg_cot/adversarial_refine.json')
parser.add_argument('--outpath', type=str, default='data/preference_data/coco_syn_cot_adv_ref_preference_combined_w_explanation.json')
parser.add_argument('--caption', action='store_true', default=False, help='Whether to include caption preference')
parser.add_argument('--yesno', action='store_true', default=False, help='Whether to include yes/no preference')
parser.add_argument('--explain', action='store_true', default=False, help='Whether to include yes/no preference with "explanation"')

args = parser.parse_args()

assert args.caption or args.yesno or args.explain, 'Either caption or yesno or explain must be true'
lower_first = lambda s: s[:1].lower() + s[1:] if s else ''

outpath = args.outpath
all_data = json.load(open(args.inpath, 'r'))
# all_data = [v for v in all_data if v['caption_corr'] == 1]
print('Number of caption pairs remaining:', len(all_data))

output_json = []
overall_count = 0
for v in all_data:
    if args.caption:
        output_json.append({
            'id' : overall_count,
            'image' : v['img_path'],
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
                    'value' : v['neg_caption']
                }
            ]
        })
        overall_count += 1
        
        
    if args.yesno:
        output_json.append({
            'id' : overall_count,
            'image' : v['img_path'],
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
            'image' : v['img_path'],
            'conversations' : [
                {
                    'from' : 'human',
                    'value' : f'<image>\nDoes the image show \'{v["neg_caption"]}\'? Please answer yes or no.'
                },
                {
                    'from' : 'gpt',
                    'value' : 'No'
                }
            ],
            'rejected_conversations' : [
                {
                    'from' : 'human',
                    'value' : f'<image>\nDoes the image show \'{v["neg_caption"]}\'? Please answer yes or no.'
                },
                {
                    'from' : 'gpt',
                    'value' : 'Yes'
                }
            ]
        })
        overall_count += 1
    
    if args.explain:
        output_json.append({
            'id' : overall_count,
            'image' : v['img_path'],
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
                    'value' : f'No, I see {lower_first(v["neg_caption"])}'
                }
            ]
        })
        overall_count += 1
        output_json.append({
            'id' : overall_count,
            'image' : v['img_path'],
            'conversations' : [
                {
                    'from' : 'human',
                    'value' : f'<image>\nDoes the image show \'{v["neg_caption"]}\'?'
                },
                {
                    'from' : 'gpt',
                    'value' : f'No, I see {lower_first(v["caption"])}'
                }
            ],
            'rejected_conversations' : [
                {
                    'from' : 'human',
                    'value' : f'<image>\nDoes the image show \'{v["neg_caption"]}\'?'
                },
                {
                    'from' : 'gpt',
                    'value' : f'Yes, I see {lower_first(v["neg_caption"])}'
                }
            ]
        })
        overall_count += 1
        
with open(outpath, 'w') as f:
    json.dump(output_json, f, indent=4)
