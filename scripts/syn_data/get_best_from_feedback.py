import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='playground/dev/coco_train_caps_cleaned.json')
    parser.add_argument('--cap_start_idx', type=int, default=0)
    parser.add_argument('--num_caps', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default='data/coco_train_syn/coco_caps_neg_feedback')
    parser.add_argument('--num_jobs', type=int, default=60)
    args = parser.parse_args()
    
    
    with open(args.caption_file, 'r') as f:
        all_orig_caps = json.load(f)
    if args.num_caps is None:
        args.num_caps = len(all_orig_caps)
    all_orig_caps = all_orig_caps[args.cap_start_idx:args.cap_start_idx+args.num_caps]
    caps_per_job = int(np.ceil(len(all_orig_caps) / args.num_jobs))
    
    best_caps = []
    for i in range(args.num_jobs):
        curr_orig_caps = all_orig_caps[i * caps_per_job: (i + 1) * caps_per_job]
        
        with open(os.path.join(args.data_dir, f'caps_{i}.jsonl'), 'r') as f:
            data = [json.loads(line) for line in f.read().splitlines()]
            
        for j, sample in enumerate(data):
            if 'caption' not in sample[0]:
                assert sample[0]['neg_caption'] == 'NA'
                continue
            
            pos_cap = sample[0]['caption']
            orig_cap = curr_orig_caps[j]['cleaned_caption'] if 'cleaned_caption' in curr_orig_caps[j] else curr_orig_caps[j]['caption']
            assert pos_cap == orig_cap, f'{pos_cap} != {orig_cap} at idx={j}, img_path = {curr_orig_caps[j]["img_path"]}'
            
            best_idx = -1
            best_score = 0.
            
            for k, cap in enumerate(sample): # 5 trials per sample
                if cap['neg_caption'] == 'NA':
                    break
                
                word_diff_score = round((len(cap['extra_words']) + len(cap['missing_words'])) / float(2 * len(cap['pos_cap_words'])) * 20.) / 20.
                cap['word_diff_score'] = word_diff_score
                avg_score = (cap['grammar_score'] + cap['plausibility_score'] + (1. - cap['word_diff_score']))/3.
                cap['avg_score'] = avg_score
                
                if cap['diff_score'].lower().startswith('no'):
                    continue
                if avg_score > best_score:
                    best_score = avg_score
                    best_idx = k
            
            if best_idx != -1:
                best_caps.append({
                    'img_path': curr_orig_caps[j]['img_path'],
                    'caption': pos_cap,
                    'neg_caption': sample[best_idx]['neg_caption'],
                    'neg_grammar_score': sample[best_idx]['grammar_score'],
                    'neg_plausibility_score': sample[best_idx]['plausibility_score'],
                    'neg_word_diff_score': sample[best_idx]['word_diff_score'],
                    'neg_avg_score': sample[best_idx]['avg_score'],
                })
                
    with open(os.path.join(args.data_dir, f'best_caps.json'), 'w') as f:
        json.dump(best_caps, f, indent=4)
    
    print(f'Saved best {len(best_caps)} captions to {os.path.join(args.data_dir, f"best_caps.json")}')