import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import spacy
from tqdm import tqdm

if __name__ == '__main__':
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")

    # inp_caps = json.load(open('data/coco_syn/coco_single_rule_neg_swap_att/combined_caps_w_scores_len_filtered.json', 'r'))
    # inp_caps = json.load(open('data/coco_train_syn/coco_single_rule_neg_v2_swap_obj/combined_caps_len_filtered.json', 'r'))
    inp_caps = json.load(open('data/coco_train_syn/coco_caps_neg_cot/combined_caps_w_scores_len_filtered.json', 'r'))
    
    
    # inp_caps = json.load(open('data/coco_syn/coco_syn_swap_v2_len_filtered.json', 'r'))
    
    print('Length before filtering: ', len(inp_caps))
    out_caps = []
    for cap in tqdm(inp_caps, total=len(inp_caps)):
        doc1 = nlp(cap['caption'].lower())
        doc2 = nlp(cap['neg_caption'].lower())
        
        # NOTE : essentially a way of testing if the two captions are exactly the same
        doc1_tokens = [tok.lemma_ for tok in doc1 if not tok.is_punct]
        doc2_tokens = [tok.lemma_ for tok in doc2 if not tok.is_punct] 
        if doc1_tokens != doc2_tokens:
            out_caps.append(cap)
    print('Length after filtering: ', len(out_caps))
    # json.dump(out_caps, open('data/coco_syn/coco_syn_swap_v2_len_plus_stem_order_filtered.json', 'w'), indent=4)
    # json.dump(out_caps, open('data/coco_syn/coco_single_rule_neg_swap_att/combined_caps_w_scores_len_plus_stem_order_filtered.json', 'w'))
    # json.dump(out_caps, open('data/coco_train_syn/coco_single_rule_neg_v2_swap_obj/combined_caps_len_plus_stem_order_filtered.json', 'w'))
    json.dump(out_caps, open('data/coco_train_syn/coco_caps_neg_cot/combined_caps_w_scores_len_plus_stem_order_filtered.json', 'w'))
    