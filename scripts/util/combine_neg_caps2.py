import os
import sys
sys.path.append(os.path.abspath('.'))
import json
from pathlib import Path

if __name__ == '__main__':
    root_dir = 'data/coco_train_syn/coco_caps_neg_cot'
    outfile = Path(root_dir) / 'combined_caps.json'
    num_jobs = 60
    
    all_caps = []
    for job_idx in range(num_jobs):
        cap_file = Path(root_dir) / f'caps_{job_idx}.jsonl'
        with open(cap_file, 'r') as f:
            for line in f.readlines():
                cap = json.loads(line)
                neg_caption = cap['neg_caption']
                if neg_caption.strip().lower() == 'na':
                    continue
                else:
                    all_caps.append({
                        'img_path': cap['img_path'],
                        'caption': cap['caption'],
                        'neg_caption': neg_caption,
                    })
    
    print(f'Number of captions: {len(all_caps)}')
    with open(outfile, 'w') as f:
        json.dump(all_caps, f, indent=4)