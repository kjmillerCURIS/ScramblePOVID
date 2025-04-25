import sys
import os
sys.path.append(os.path.abspath('.'))
import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_scores_file', type=str, default='data/coco_train_syn/coco_caps_neg_cot/combined_caps_w_scores.json')
    parser.add_argument('--missing_scores_file', type=str, default='data/coco_train_syn/coco_caps_neg_feedback/best_caps_pos_scores_not_found.json')
    parser.add_argument('--cap_file', type=str, default='data/coco_train_syn/coco_caps_neg_feedback/best_caps.json')
    args = parser.parse_args()
    
    pos_scores = json.load(open(args.pos_scores_file, 'r'))
    missing_scores = json.load(open(args.missing_scores_file, 'r'))
    
    img_to_pos_scores = {}
    for x in pos_scores:
        if x['img_path'] in img_to_pos_scores:
            print('Duplicate img_path in pos_scores:', x['img_path'])
        img_to_pos_scores[x['img_path']] = x
    
    for x in missing_scores:
        if x['img_path'] in img_to_pos_scores:
            print('Duplicate img_path in missing_scores:', x['img_path'])
        img_to_pos_scores[x['img_path']] = x
        
    caps = json.load(open(args.cap_file, 'r'))
    num_not_found = 0
    scores_not_found = []
    for cap in caps:
        if cap['img_path'] not in img_to_pos_scores:
            print('Img_path not found in pos_scores:', cap['img_path'])
            scores_not_found.append(cap)
            continue
        if cap['caption'] != img_to_pos_scores[cap['img_path']]['caption']:
            print('Caption mismatch for ', cap['img_path'])
            scores_not_found.append(cap)
            continue
        
        cap['pos_grammar_score'] = img_to_pos_scores[cap['img_path']]['pos_grammar_score']
        cap['pos_plausibility_score'] = img_to_pos_scores[cap['img_path']]['pos_plausibility_score']
    json.dump(caps, open(Path(args.cap_file).parent / 'best_caps_w_scores.json', 'w'))
    print('Num not found:', len(scores_not_found))
    # json.dump(scores_not_found, open(Path(args.cap_file).parent / 'best_caps_pos_scores_not_found.json', 'w'))