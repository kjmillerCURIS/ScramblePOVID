import os
import sys
sys.path.append(os.path.abspath('.'))
import json
from pathlib import Path

if __name__ == '__main__':
    root_dir = 'data/coco_syn/coco_caps_neg_v2/adv_ref'
    num_jobs = 5
    outfile = Path(root_dir) / 'combined_caps_paraphrase_test.json'
    
    all_caps = []
    pos_num = 0
    neg_num = 0
    null_num = 0
    for job_idx in range(num_jobs):
        infile = Path(root_dir) / f'caps_paraphrase_test_{job_idx}.jsonl'
        with open(infile, 'r') as f:
            for line in f.readlines():
                curr_cap = json.loads(line)
                if curr_cap['viz_diff'] == True:
                    del curr_cap['viz_diff']
                    del curr_cap['llm_out']
                    all_caps.append(curr_cap)
                    pos_num += 1
                    continue
                elif curr_cap['viz_diff'] == False:
                    neg_num += 1
                    continue
                else:
                    all_caps.append(curr_cap)
                    null_num += 1
                    continue
    
    with open(outfile, 'w') as f:
        json.dump(all_caps, f, indent=4)
    print(f'Positive: {pos_num}, Negative: {neg_num}, Null: {null_num}')