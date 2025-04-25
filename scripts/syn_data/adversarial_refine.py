import os
import sys
sys.path.append(os.path.abspath('.'))
import numpy as np
from tqdm import trange
import json
import argparse
from pathlib import Path

def adversarial_refine(model1_score_gap, model2_score_gap):
    n_grids = 50
    keep = []  # list of indices to keep

    score_to_idx = {}
    for i in range(1, n_grids + 1):
        for j in range(1, n_grids + 1):
            score_to_idx[(i, j)] = []
            score_to_idx[(-i, -j)] = []
            score_to_idx[(i, -j)] = []
            score_to_idx[(-i, j)] = []

    model1_zero_idx = []
    model2_zero_idx = []
    for i, (model1_gap, model2_gap) in enumerate(zip(model1_score_gap, model2_score_gap)):
        if model1_gap == 0 and model2_gap == 0:
            keep.append(i)
            continue
        if model1_gap == 0:
            model1_zero_idx.append(i)
            continue
        if model2_gap == 0:
            model2_zero_idx.append(i)
            continue

        model1_id = int(abs(model1_gap) * n_grids) + 1
        model2_id = int(abs(model2_gap) * n_grids) + 1
        if model1_gap > 0 and model2_gap > 0:
            score_to_idx[(model1_id, model2_id)].append(i)
        elif model1_gap < 0 and model2_gap < 0:
            score_to_idx[(-model1_id, -model2_id)].append(i)
        elif model1_gap > 0 and model2_gap < 0:
            score_to_idx[(model1_id, -model2_id)].append(i)
        elif model1_gap < 0 and model2_gap > 0:
            score_to_idx[(-model1_id, model2_id)].append(i)
        else:
            raise ValueError

    for i in range(1, n_grids + 1):
        for j in range(1, n_grids + 1):
            idx = score_to_idx[(i, j)]
            op_idx = score_to_idx[(-i, -j)]
            if len(idx) > 0 and len(op_idx) > 0:
                if len(idx) > len(op_idx):
                    keep.extend(op_idx)
                    keep.extend(list(np.random.choice(idx, len(op_idx), replace=False)))
                else:
                    keep.extend(idx)
                    keep.extend(list(np.random.choice(op_idx, len(idx), replace=False)))

            idx = score_to_idx[(i, -j)]
            op_idx = score_to_idx[(-i, j)]
            if len(idx) > 0 and len(op_idx) > 0:
                if len(idx) > len(op_idx):
                    keep.extend(op_idx)
                    keep.extend(list(np.random.choice(idx, len(op_idx), replace=False)))
                else:
                    keep.extend(idx)
                    keep.extend(list(np.random.choice(op_idx, len(idx), replace=False)))

    n_grids = 10

    if len(model1_zero_idx) > 0:
        pos_score_to_idx, neg_score_to_idx = {i: [] for i in range(n_grids)}, {i: [] for i in range(n_grids)}
        for i in model1_zero_idx:
            gap = model2_score_gap[i]
            if gap > 0:
                pos_score_to_idx[int(gap * n_grids)].append(i)
            else:
                neg_score_to_idx[int(-gap * n_grids)].append(i)
        for i in trange(n_grids):
            pos_idx = pos_score_to_idx[i]
            neg_idx = neg_score_to_idx[i]
            if len(pos_idx) > len(neg_idx):
                keep.extend(neg_idx)
                keep.extend(list(np.random.choice(pos_idx, len(neg_idx), replace=False)))
            else:
                keep.extend(pos_idx)
                keep.extend(list(np.random.choice(neg_idx, len(pos_idx), replace=False)))

    if len(model2_zero_idx) > 0:
        pos_score_to_idx, neg_score_to_idx = {i: [] for i in range(n_grids)}, {i: [] for i in range(n_grids)}
        for i in model2_zero_idx:
            gap = model1_score_gap[i]
            if gap > 0:
                pos_score_to_idx[int(gap * n_grids)].append(i)
            else:
                neg_score_to_idx[int(-gap * n_grids)].append(i)
        for i in trange(n_grids):
            pos_idx = pos_score_to_idx[i]
            neg_idx = neg_score_to_idx[i]
            if len(pos_idx) > len(neg_idx):
                keep.extend(neg_idx)
                keep.extend(list(np.random.choice(pos_idx, len(neg_idx), replace=False)))
            else:
                keep.extend(pos_idx)
                keep.extend(list(np.random.choice(neg_idx, len(pos_idx), replace=False)))

    return keep

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1_scores', type=str, default='data/coco_train_syn/coco_caps_neg_feedback/best_caps_w_scores.json')
    parser.add_argument('--model2_scores', type=str, default='data/coco_train_syn/coco_caps_neg_feedback/best_caps_w_scores.json')
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = Path(args.model1_scores).parent
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.model1_scores, 'r') as f:
        model1_scores = json.load(f)
    with open(args.model2_scores, 'r') as f:
        model2_scores = json.load(f)
        
    model1_score_gap = [c['pos_grammar_score'] - c['neg_grammar_score'] for c in model1_scores]
    model2_score_gap = [c['pos_plausibility_score'] - c['neg_plausibility_score'] for c in model2_scores]
        
    keep = adversarial_refine(model1_score_gap, model2_score_gap)
    out_caps = []
    
    print(f'Kept {len(keep)} out of {len(model1_scores)}.')
    for i in keep:
        curr_cap = model1_scores[i]
        curr_cap['pos_grammar_score'] = model1_scores[i]['pos_grammar_score']
        curr_cap['neg_grammar_score'] = model1_scores[i]['neg_grammar_score']
        curr_cap['pos_plausibility_score'] = model2_scores[i]['pos_plausibility_score']
        curr_cap['neg_plausibility_score'] = model2_scores[i]['neg_plausibility_score']
        out_caps.append(curr_cap)
        
    with open(os.path.join(args.out_dir, 'adversarial_refine.json'), 'w') as f:
        json.dump(out_caps, f, indent=4)