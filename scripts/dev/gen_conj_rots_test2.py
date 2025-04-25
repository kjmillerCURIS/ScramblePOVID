import spacy

def get_full_noun_phrase(token):
    """Recursively gather all tokens in a noun phrase."""
    phrase = [token]
    for child in token.children:
        if child.dep_ in ['compound', 'amod', 'det', 'nummod', 'poss', 'advmod', 'prep', 'relcl']:
            phrase = get_full_noun_phrase(child) + phrase
    return phrase

def find_coordinated_nps(doc):
    """Find coordinated noun phrases in the sentence."""
    for token in doc:
        if token.dep_ == 'cc' and token.head.pos_ in ['NOUN', 'PROPN']:
            left_np = get_full_noun_phrase(token.head)
            right_np = None
            for sibling in token.head.rights:
                if sibling.dep_ == 'conj':
                    right_np = get_full_noun_phrase(sibling)
                    break
            if left_np and right_np:
                return left_np, right_np
    return None, None

def swap_noun_phrases(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    print("Dependency Parse:")
    for token in doc:
        print(f"{token.text:<15} {token.dep_:<10} {token.head.text:<15} {[child.text for child in token.children]}")
    
    left_np, right_np = find_coordinated_nps(doc)
    
    if not left_np or not right_np:
        print("Couldn't find coordinated noun phrases")
        return sentence
    
    print(f"\nLeft NP: {[token.text for token in left_np]}")
    print(f"Right NP: {[token.text for token in right_np]}")
    
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
sentences = [
    "The old man with a cane and the young woman in a red dress walked down the street.",
    "The big brown cat and the small white dog are playing in the yard.",
    "The shiny new car in the garage and the rusty old bicycle on the porch belong to my neighbor."
]

for sentence in sentences:
    swapped_sentence = swap_noun_phrases(sentence)
    print(f"\nOriginal: {sentence}")
    print(f"Swapped:  {swapped_sentence}")