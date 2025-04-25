import spacy

def swap_noun_phrases(sentence):
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")
    
    # Parse the sentence
    doc = nlp(sentence)
    
    # Find 'and' conjunction
    and_token = None
    for token in doc:
        if token.text.lower() == 'and' and token.dep_ == 'cc':
            and_token = token
            break
    
    if and_token is None:
        return sentence  # No 'and' conjunction found
    
    # Find the two noun phrases to swap
    left_np = None
    right_np = None
    
    # Look for left noun phrase
    for token in and_token.lefts:
        if token.dep_ in ['conj', 'nsubj', 'dobj', 'pobj']:
            left_np = token.subtree
            break
    
    # Look for right noun phrase
    for token in and_token.rights:
        if token.dep_ in ['conj', 'nsubj', 'dobj', 'pobj']:
            right_np = token.subtree
            break
    
    # Print dependency information for debugging
    print("Dependency Parse:")
    for token in doc:
        print(f"{token.text:<15} {token.dep_:<10} {token.head.text:<15} {[child for child in token.children]}")
    
    import ipdb; ipdb.set_trace()
    
    if left_np is None or right_np is None:
        return sentence  # Couldn't find both noun phrases
    
    # Convert subtrees to lists and sort by token index
    left_np = sorted(list(left_np), key=lambda x: x.i)
    right_np = sorted(list(right_np), key=lambda x: x.i)
    
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
sentence = "The cat and the dog are playing in the yard."
swapped_sentence = swap_noun_phrases(sentence)
print(f"Original: {sentence}")
print(f"Swapped:  {swapped_sentence}")