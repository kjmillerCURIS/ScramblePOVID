import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import numpy as np
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--winoground_examples_file', type=str, default='/home/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets/winoground/examples.jsonl')
    parser.add_argument('--num_examples', type=int, default=10)
    parser.add_argument('--outdir', type=str, default='data/openai/winoground_incontext_examples')
    parser.add_argument('--outfile_name', type=str, default='random.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    winoground_examples_file = args.winoground_examples_file
    winoground_examples = [json.loads(line) for line in open(winoground_examples_file, 'r')]
    
    RNG = np.random.RandomState(args.seed)
    all_idxs = RNG.choice(len(winoground_examples), args.num_examples, replace=False)
    selected_winoground_examples = [{
        'captionA' : winoground_examples[idx]['caption_0'],
        'captionB' : winoground_examples[idx]['caption_1'],
        'winoground_id' : winoground_examples[idx]['id']
    } for idx in all_idxs]
    
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.outdir, args.outfile_name), 'w') as f:
        json.dump(selected_winoground_examples, f, indent=4)
    
    

