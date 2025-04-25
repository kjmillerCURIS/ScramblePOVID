import os
import sys
sys.path.append(os.path.abspath('.'))

import transformers
import torch
import json
from tqdm import tqdm

from transformers import BitsAndBytesConfig
import numpy as np

from grammar_score import load_model as load_grammar_model
from plausibility_score import load_model as load_plausibility_model
from grammar_score import get_score as get_grammar_score
from plausibility_score import get_score as get_plausibility_score

import spacy
# Load the English language model
nlp = spacy.load("en_core_web_sm")

from sentence_transformers import SentenceTransformer

import open_clip

import warnings
warnings.filterwarnings("ignore")


NEG_PROMPT_2 = """
Given an input caption describing a scene, your task is to rearrange words in it to make a new caption. The new caption must meet the following three requirements:
1. It must describe a scene with visual differences compared to the scene described by the input caption.
2. It must be fluent and grammatically correct.
3. It must make logical sense.
Note that you can choose to abstain and output 'NA' if it is not possible to generate a negative caption for the given input.
To help with your task, I will rate your output based on grammar (0-1), plausibility (0-1), and a dissimilarity score (0-1, 1 being the most dissimilar from the original caption). You should try to maximize all. 

In your output, please follow the format 

Final Output Caption: <caption>.

Here is the input caption: {caption}
"""

def get_score_feedback(curr_score, prev_score, score_type):
    if prev_score is None:
        return f"Your {score_type} score is {curr_score:.2f}.\n"
    elif curr_score > prev_score:
        return f"Your {score_type} score improved to {curr_score:.2f}.\n"
    elif curr_score < prev_score:
        return f"Your {score_type} score degraded to {curr_score:.2f}.\n"
    else:
        return f"Your {score_type} score remained the same at {curr_score:.2f}.\n"
   
def caption_words(caption):
    if caption[-1] == '.':
        caption = caption[:-1]
    # Process both captions
    doc = nlp(caption.lower())
    # Get lemmas (base forms) of words, excluding punctuation
    words = [token.lemma_ for token in doc if not token.is_punct]
    return words

RNG = np.random.RandomState(44)

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
# pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"quantization_config": nf4_config}, device_map="auto")
# pipeline.model = torch.compile(pipeline.model, mode="max-autotune", fullgraph=True, dynamic=False, backend="inductor")

pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
pipeline_params = {
    "max_new_tokens": 800,
    "eos_token_id": terminators,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.9,
    "return_full_text": False,
}
apply_chat_template_params = {
    "tokenize": False,
    "add_generation_prompt": True
}

grammar_model = load_grammar_model()
plausibility_model = load_plausibility_model()

clip_model = open_clip.create_model('ViT-L-16-SigLIP-384', 'webli')
text_encoder = clip_model.text
text_encoder.cuda()
clip_tokenizer = open_clip.get_tokenizer('ViT-L-16-SigLIP-384')

# similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def genfn(messages):
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        **apply_chat_template_params
    )
    outputs = pipeline(prompt, **pipeline_params)
    return outputs

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

all_neg_captions = [None] * len(all_captions)
NUM_ITERS = 5

for i, caption in enumerate(all_captions):
    print(f'############# Caption {i+1}/{len(all_captions)} ##############')
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Please respond to the user's questions succinctly."},
        {"role": "user", "content": f'{NEG_PROMPT_2.format(caption=caption["caption"])}'},
    ]
    print(f'USER: {NEG_PROMPT_2.format(caption=caption["caption"])}')
    
    all_cap_feedback = []
    pos_cap_words = caption_words(caption["caption"])
    with torch.no_grad():
        # pos_embed = similarity_model.encode(caption["caption"])
        pos_embed = text_encoder(clip_tokenizer(caption["caption"]).cuda())
        pos_embed = pos_embed / pos_embed.norm(dim=-1, keepdim=True)
        
    prev_grammar_score = None
    prev_plaus_score = None
    prev_dissim_score = None
    for iter in range(NUM_ITERS):
        outputs = genfn(messages)
        print(f'MODEL: {outputs[0]["generated_text"]}')
        neg_caption_idx = outputs[0]["generated_text"].find('Final Output Caption:') + len('Final Output Caption:')
        neg_caption = outputs[0]["generated_text"][neg_caption_idx:].strip()
        if neg_caption == 'NA':
            all_cap_feedback.append({
                'neg_caption' : neg_caption,
            })
            break
        
        # mention if score improved or not
        # mention which words were different
        # mention how similar the original caption and the negative caption are : currently only checking for extra and missing words
        
        plaus_score = get_plausibility_score(neg_caption, **plausibility_model)
        grammar_score = get_grammar_score(neg_caption, **grammar_model)
        neg_cap_words = caption_words(neg_caption)
        with torch.no_grad():
            neg_embed = text_encoder(clip_tokenizer(neg_caption).cuda())
            neg_embed = neg_embed / neg_embed.norm(dim=-1, keepdim=True)
            
        extra_words = set(neg_cap_words) - set(pos_cap_words)
        missing_words = set(pos_cap_words) - set(neg_cap_words)
        
        same_words = len(extra_words) == 0 and len(missing_words) == 0
        exact_same = pos_cap_words == neg_cap_words
        
        messages.append({"role": "assistant", "content": outputs[0]["generated_text"]})
        
        user_mesg = "FEEDBACK:\n"
        if exact_same:
           user_mesg += "Your output caption is exactly the same as the original caption. Can you please try again?\n"
        else:
            user_mesg += get_score_feedback(grammar_score, prev_grammar_score, "grammar")
            user_mesg += get_score_feedback(plaus_score, prev_plaus_score, "plausibility")
            sim_score = (pos_embed * neg_embed).sum()
            dissim_score = ((1. - sim_score)/2.).item()
            user_mesg += get_score_feedback(dissim_score, prev_dissim_score, "dissimilarity")
            
            if len(extra_words) > 0:
                user_mesg += f"Your output caption has extra words (lemmatized): {extra_words}.\n"
            if len(missing_words) > 0:
                user_mesg += f"Your output caption has missing words (lemmatized): {missing_words}.\n"
            user_mesg += "Can you please try again?\n"
        
        curr_feedback = {
            'neg_caption': neg_caption,
            'exact_same': exact_same,
            'grammar_score': grammar_score,
            'plausibility_score': plaus_score,
            'dissimilarity_score': dissim_score,
            'extra_words': list(extra_words),
            'missing_words': list(missing_words),
            'feedback_str': user_mesg
        }
        all_cap_feedback.append(curr_feedback)
        
        prev_grammar_score = grammar_score
        prev_plaus_score = plaus_score
        prev_dissim_score = dissim_score
        # user_mesg += f"Your output caption got a grammar score of {grammar_score:.2f} and plausibility score of {plaus_score:.2f}. Can you please try again?"
        
        print(f'USER: {user_mesg}')
        messages.append({"role": "user", "content": user_mesg})
        
        # outputs = genfn(messages)
        # print(f'MODEL: {outputs[0]["generated_text"]}')
        # neg_caption_idx = outputs[0]["generated_text"].find('Final Output Caption:') + len('Final Output Caption:')
        # neg_caption = outputs[0]["generated_text"][neg_caption_idx:].strip()
    
    all_neg_captions[i] = neg_caption
    # print(f"Negative caption : {all_neg_captions[i]}")
    print('#############################################################')
    
    # save the feedback 
    # TODO : get best caption from feedback
    # dumping every round
    with open(f'playground/dev/feedbacks_8b_siglip_no_reason.json', 'w') as f:
        json.dump(all_cap_feedback, f, indent=4)