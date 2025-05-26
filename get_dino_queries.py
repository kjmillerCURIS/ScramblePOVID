import pandas as pd
import spacy
import random

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define quantifier mapping and type
QUANTIFIER_MAP = {
    "one": ("exactly", 1),
    "two": ("exactly", 2),
    "three": ("exactly", 3),
    "four": ("exactly", 4),
    "five": ("exactly", 5),
    "several": ("at least", 2),
    "a group": ("at least", 2),
    "a variety": ("at least", 2),
    "many": ("at least", 2),
    "multiple": ("at least", 2)
}

SPATIAL_TERMS = {
    "front", "background", "middle", "center", "top", "bottom",
    "side", "left", "right", "corner", "edge", "rear", "back"
}

EXTRA_FILTERS = {"it"}


# Function to extract and normalize noun phrases while preserving coordination and adjectives
def extract_noun_phrases_with_counts(caption):
    doc = nlp(caption)
    results = {}

    for np in doc.noun_chunks:
        np_text = np.text.lower().strip()
        quant_type = "exactly"
        count = 1

        # Detect and strip quantifier
        for quant, (qtype, qval) in QUANTIFIER_MAP.items():
            if np_text.startswith(quant):
                quant_type = qtype
                count = qval
                np_text = np_text[len(quant):].strip()
                break

        # Remove leading determiners (e.g., "a", "the")
        np_tokens = nlp(np_text)
        filtered_tokens = [token for token in np_tokens if token.pos_ != "DET"]

        # Keep adjectives, conjunctions, nouns, and proper nouns
        phrase_tokens = []
        last_noun_token = None
        for token in filtered_tokens:
            if token.pos_ in {"ADJ", "CCONJ", "NOUN", "PROPN"}:
                if token.pos_ in {"NOUN", "PROPN"}:
                    last_noun_token = token
                if token.pos_ in {"NOUN", "PROPN"}:
                    phrase_tokens.append(token.lemma_)
                else:
                    phrase_tokens.append(token.text)
                    
         # Determine if plural and no explicit quantifier
        if last_noun_token and quant_type == "exactly" and count == 1:
            if last_noun_token.tag_ in {"NNS", "NNPS"}:
                quant_type = "at least"
                count = 2

        if phrase_tokens:
            phrase = " ".join(phrase_tokens)
            if phrase in SPATIAL_TERMS or phrase in EXTRA_FILTERS:
                continue
            results[phrase] = (quant_type, count)

    return results


##Set captions here - right now it just samples random captions from val.csv
#
## Load CSV
#csv_path = "/projectnb/ivc-ml/ac25/Datasets/Open Images/val.csv"  
#df = pd.read_csv(csv_path)
#
## Ensure the column exists
#if "positive_caption" not in df.columns:
#    raise ValueError("Column 'positive_caption' not found in CSV.")
#
## Drop missing entries and sample 100 random captions
#captions = df["positive_caption"].dropna().tolist()
#sampled_captions = random.sample(captions, min(100, len(captions)))
#
## Process and print
#for caption in sampled_captions:
#    np_counts = extract_noun_phrases_with_counts(caption)
#    print(f"{caption} -> {np_counts}")
