import os
import sys
sys.path.append(os.path.abspath('.'))
import json
from pathlib import Path

if __name__ == '__main__':
    root_dir = 'data/coco_syn/coco_caps_neg_v2'
    outfile = Path(root_dir) / 'combined_caps.json'
    num_jobs = 20
    
    all_caps = []
    for job_idx in range(num_jobs):
        cap_file = Path(root_dir) / f'caps_tested_{job_idx}.jsonl'
        with open(cap_file, 'r') as f:
            for line in f.readlines():
                cap = json.loads(line)
                caption_corr = cap['caption_corr']
                if caption_corr is None:
                    cap['caption_corr'] = None
                elif caption_corr.lower().startswith('yes'):
                    cap['caption_corr'] = 1
                elif caption_corr.lower().startswith('no'):
                    cap['caption_corr'] = 0
                else:
                    print('#' * 100)
                    print('File : ', cap_file)
                    print('Image : ', cap['img_path'])
                    print('Caption : ', cap['caption'])
                    print('Neg caption : ', cap['neg_caption'])
                    print('Invalid caption_corr : ', caption_corr)
                    cap['caption_corr'] = None
                    cap['llm_out'] = caption_corr 
                    print('#' * 100)
                all_caps.append(cap)
    
    with open(outfile, 'w') as f:
        json.dump(all_caps, f, indent=4)