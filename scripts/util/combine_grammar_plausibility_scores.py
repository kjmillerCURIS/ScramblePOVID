import os
import sys
sys.path.append(os.path.abspath('.'))
import json
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='data/coco_train_syn/coco_caps_neg_feedback/best_caps_pos_scores_not_found.json')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path(args.caption_file).parent
    
    grammar_scores_file = Path(args.output_dir) / 'grammar_scores_textattack.json'
    plausibility_scores_file = Path(args.output_dir) / 'plausibility_scores_vera.json'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    with open(args.caption_file, 'r') as f:
        captions = json.load(f)
        
    with open(grammar_scores_file, 'r') as f:
        grammar_scores = json.load(f)
        
    with open(plausibility_scores_file, 'r') as f:
        plausibility_scores = json.load(f)
    
    for c, g, p in zip(captions, grammar_scores, plausibility_scores):
        assert c['img_path'] == g['img_path'] == p['img_path']
        c['pos_grammar_score'] = g['pos_grammar_score']
        c['neg_grammar_score'] = g['neg_grammar_score']
        c['pos_plausibility_score'] = p['pos_plausibility_score']
        c['neg_plausibility_score'] = p['neg_plausibility_score']
        
    with open(Path(args.output_dir) / 'best_caps_pos_scores_not_found.json', 'w') as f:
        json.dump(captions, f, indent=4)