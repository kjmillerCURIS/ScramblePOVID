import sys
import os
sys.path.append(os.path.abspath('.'))

import json

all_captions = [
    {"caption": "A small child drinking something in front of an open refrigerator."},
    {"caption": "A young zebra sniffing the ground in a dusty area."},
    {"caption": "Three chairs, a couch, and a table sitting on a rug."},
    {"caption": "A child in a blue hat, black shirt, and jeans hitting a green ball with a bat."},
    {"caption": "This pizzeria sells pizza for one euro at their place."},
    {"caption": "A white horse pulling a cart down a street."},
    {"caption": "A woman jumping a horse over an obstacle made of tires."},
    {"caption": "A woman showing a teddy bear to a child and her mother."},
    {"caption": "Standing on a bridge looking down at railroad tracks."},
    {"caption": "A person standing by some very cute tall giraffes."}
]

if __name__ == "__main__":
    with open('playground/dev/feedbacks2_70b.json', 'r') as f:
        data = json.load(f)
        
    outfile = open('playground/dev/highlighted_best_neg_cap_70b.md', 'w')
    for j, cap_data in enumerate(data):
        best_idx = 0
        best_score = 0.
        word_diff_thres = 0.5
        # ignore if too many diff extra/missing words
        # Get best caption
        for i, cap in enumerate(cap_data):
            # TODO : print and highlight
            word_diff_score = round((len(cap['extra_words']) + len(cap['missing_words'])) / float(2 * len(cap['pos_cap_words'])) * 20.) / 20.
            cap['word_diff_score'] = word_diff_score
            avg_score = (cap['grammar_score'] + cap['plausibility_score'] + (1. - word_diff_score))/3.
            cap['avg_score'] = avg_score
            
            # if word_diff_score > word_diff_thres:
            #     continue
            if cap['diff_score'].lower().startswith('no'):
                continue
            if avg_score > best_score:
                best_score = avg_score
                best_idx = i
        
        print('## Caption ' + str(j), file=outfile)
        print('Original caption: ' + all_captions[j]['caption'], file=outfile)
        for i, cap in enumerate(cap_data):
            if i == best_idx:
                print(f'### Candidate {i} (BEST) *************************', file=outfile)
            else:
                print(f'### Candidate {i}', file=outfile)
            
            print(f'**Grammar score:** {cap["grammar_score"]:.4f}  ', file=outfile)
            print(f'**Plausibility score:** {cap["plausibility_score"]:.4f}  ', file=outfile)
            print(f'**Average score:** {cap["avg_score"]:.4f}  ', file=outfile)
            print(f'**Word diff score:** {cap["word_diff_score"]:.4f}  ', file=outfile)
            print(f'**Diff score:** {cap["diff_score"]}  ', file=outfile)
            
            print(f'**Neg caption:** {cap["neg_caption"]}  ', file=outfile)
            if i == best_idx:
                print('*************************', file=outfile)
