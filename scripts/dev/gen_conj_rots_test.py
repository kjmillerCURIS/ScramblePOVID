import spacy

def find_np_head(token):
    if token.dep_ in ['nsubj', 'dobj', 'pobj', 'ROOT']:
        return token
    elif token.head != token:  # Check if we're not at the root
        # TODO : add the index of the current token to and_token, so that it does not get into the left NP
        return find_np_head(token.head)
    else:
        return token  # Return the root if no suitable head is found


def swap_noun_phrases(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    print("Dependency Parse:")
    for token in doc:
        print(f"{token.text:<15} {token.dep_:<10} {token.head.text:<15} {[child.text for child in token.children]}")
    
    # import ipdb; ipdb.set_trace()
    
    # Find 'and' conjunction
    and_token = None
    for token in doc:
        if (token.text.lower() == 'and' and token.dep_ == 'cc') or (token.text.lower() == 'beside' and token.dep_ == 'prep') or (token.text.lower() == 'to' and token.dep_ == 'prep'):
            and_token = token
            break
    
    if and_token is None:
        print("No 'and' conjunction found")
        return sentence
    
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
    
    print(f"\nLeft NP: {left_np}")
    print(f"Right NP: {right_np}")
    
    if not left_np or not right_np:
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

# Example usage
# sentence = "The cat and the dog are playing in the yard."
# sentence = "Several containers of fruits and vegetables sitting on a table."
# sentence = "The big brown cat and the small white dog are playing in the yard."
# sentence = "The big brown cat next to the small white dog."
# sentence = "A polar bear standing beside a pool of water ."
sentence = "A plate of pasta sitting beside a salad and a bowl with sausage ."

# sentence = "The old man with a cane and the young woman in a red dress walked down the street."
swapped_sentence = swap_noun_phrases(sentence)
print(f"\nOriginal: {sentence}")
print(f"Swapped:  {swapped_sentence}")

# TODO : 'next to' preceeded by a verb like 'sitting next to' does not work for the above example. 
# Sitting/sits seems to become ROOT.