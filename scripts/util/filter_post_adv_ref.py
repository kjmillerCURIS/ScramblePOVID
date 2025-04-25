import os
import sys
sys.path.append(os.path.abspath('.'))
from pathlib import Path
import json
import spacy
from tqdm import tqdm

def find_np_head(token):
    if token.dep_ in ['nsubj', 'dobj', 'pobj', 'ROOT']:
        return token
    elif token.head != token:  # Check if we're not at the root
        # TODO : add the index of the current token to and_token, so that it does not get into the left NP
        return find_np_head(token.head)
    else:
        return token  # Return the root if no suitable head is found


def swap_noun_phrases(sentence, nlp, verbose=False):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    if verbose:
        print("Dependency Parse:")
        for token in doc:
            print(f"{token.text:<15} {token.dep_:<10} {token.head.text:<15} {[child.text for child in token.children]}")
    
    
    # Find 'and' conjunction
    and_token = None
    for token in doc:
        if (token.text.lower() == 'and' and token.dep_ == 'cc') or (token.text.lower() == 'beside' and token.dep_ == 'prep') or (token.text.lower() == 'to' and token.dep_ == 'prep'):
            and_token = token
            break
    
    if and_token is None:
        if verbose:
            print("No 'and' conjunction found")
        return sentence
    
    if verbose:
        print(f"\nAnd token: {and_token}")
        print(f"And token head: {and_token.head}")
    
    # Find the two noun phrases to swap
    np_head = find_np_head(and_token.head)
    left_np = []
    right_np = []
    
    for token in np_head.subtree:
        if token.i < and_token.i and token.dep_ not in ['prep', 'acl', 'advmod']:
            left_np.append(token)
        elif token.i > and_token.i:
            if token.dep_ in ['conj', 'pobj']:
                # TODO : this has to be more sophisticated for complex phrases like "a plate of pasta" 
                # because otherwise it just skips tokens like "of" and retains "a plate pasta"
                right_np = list(token.subtree)
                break
    
    if verbose:
        print(f"\nLeft NP: {left_np}")
        print(f"Right NP: {right_np}")
    
    if not left_np or not right_np:
        if verbose:
            print("Couldn't find both noun phrases")
        return sentence
    
    # Create new tokens list with swapped noun phrases
    new_tokens = []
    i = 0
    while i < len(doc):
        if i == left_np[0].i:
            new_tokens.extend([t.text_with_ws for t in right_np])
            i += len(left_np)
        elif i == right_np[0].i:
            new_tokens.extend([t.text_with_ws for t in left_np])
            i += len(right_np)
        else:
            new_tokens.append(doc[i].text_with_ws)
            i += 1
    
    return ''.join(new_tokens).strip()


if __name__ == '__main__':
    cap_file = Path('data/coco_syn/coco_single_rule_neg_v2_swap_obj/adversarial_refine.json')
    with open(cap_file, 'r') as f:
        data = json.load(f)
    
    num_removed = 0
    all_caps = []
    for d in data:
        if d['caption'].strip().lower() == d['neg_caption'].strip().lower():
            num_removed += 1
        else:
            all_caps.append(d)
            
    print(f'Removed {num_removed} captions')
    print(f'Remaining {len(all_caps)} captions')
    # with open(cap_file.parent / 'adversarial_refine_same_neg_removed.json', 'w') as f:
    #     json.dump(all_caps, f, indent=4)
        
        
    nlp = spacy.load('en_core_web_sm')
    num_removed_swap_test = 0
    all_caps_swap_test = []
    for d in tqdm(all_caps, total=len(all_caps)):
        cap_comp = d['caption']
        if cap_comp[-1] == '.':
            cap_comp = cap_comp[:-1] + ' .'
        cap_comp = swap_noun_phrases(cap_comp, nlp)
        if cap_comp.strip().lower() == d['neg_caption'].strip().lower():
            num_removed_swap_test += 1
        else:
            all_caps_swap_test.append(d)
                
    print(f'Removed {num_removed_swap_test} captions')
    print(f'Remaining {len(all_caps_swap_test)} captions')
    with open(cap_file.parent / 'adversarial_refine_post_filter.json', 'w') as f:
        json.dump(all_caps_swap_test, f, indent=4)
    
    