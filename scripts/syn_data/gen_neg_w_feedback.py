import os
import sys
sys.path.append(os.path.abspath('.'))

import transformers
import torch
import json
from tqdm import tqdm

from transformers import BitsAndBytesConfig
import numpy as np
import argparse
from pathlib import Path

from datetime import datetime
import time

import spacy
# Load the English language model
nlp = spacy.load("en_core_web_sm")

NEG_PROMPT = """
Given an input caption describing a scene, your task is to rearrange words in it to make a new caption. The new caption must meet the following three requirements:
1. It must describe a scene with visual differences compared to the scene described by the input caption.
2. It must be fluent and grammatically correct.
3. It must make logical sense.
Note that you can choose to abstain and output 'NA' if it is not possible to generate a negative caption for the given input.

To help with your task, I will rate your output based on grammar (0-1), plausibility (0-1), and whether there are visual differences between the original caption and your output (Yes/No).

In your output, please follow the format 

Final Output Caption: <caption>.

Here is the input caption: {caption}
"""

DIFF_PROMPT = """
Are there visual differences between the following two captions? Please respond with 'Yes' or 'No'.

Caption 1: {caption}
Caption 2: {neg_caption}
"""

NUM_ITERS = 5

def get_sha():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha

def load_grammar_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained('textattack/distilbert-base-uncased-CoLA')
    model = transformers.DistilBertForSequenceClassification.from_pretrained('textattack/distilbert-base-uncased-CoLA')
    model.to(device='cpu')
    model.eval()
    return {
        "tokenizer": tokenizer,
        "model": model
    }
    
@torch.no_grad()
def get_grammar_score(caption, tokenizer, model):
    toks = tokenizer.batch_encode_plus(
        [caption], 
        return_tensors='pt', 
        padding='longest', 
        truncation='longest_first', 
        max_length=128
    )
    input_ids = toks.input_ids.to(model.device)
    attention_mask = toks.attention_mask.to(model.device)
            
    output = model(input_ids, attention_mask=attention_mask)
    score = output['logits'].softmax(dim=1)[:, 1]
            
    return score[0].item()

def load_plausibility_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained('liujch1998/vera', device_map='cpu')
    model = transformers.T5EncoderModel.from_pretrained('liujch1998/vera', device_map='cpu')
    model.D = model.shared.embedding_dim
    linear = torch.nn.Linear(model.D, 1, dtype=model.dtype, device=model.device)
    linear.weight = torch.nn.Parameter(model.shared.weight[32099, :].unsqueeze(0))
    linear.bias = torch.nn.Parameter(model.shared.weight[32098, 0].unsqueeze(0))
    model.eval()
    t = model.shared.weight[32097, 0].item() # temperature for calibration
    return {
        "tokenizer": tokenizer,
        "model": model,
        "linear": linear,
        "t": t
    }

@torch.no_grad()
# @torch.autocast(device_type='cuda', dtype=torch.float16)
def get_plausibility_score(caption, tokenizer, model, linear, t):
    toks = tokenizer.batch_encode_plus(
        [caption], 
        return_tensors='pt', 
        padding='longest', 
        truncation='longest_first', 
        max_length=128
    )
    input_ids = toks.input_ids.to(model.device)
    attention_mask = toks.attention_mask.to(model.device)
        
    output = model(input_ids, attention_mask=attention_mask)
    last_indices = attention_mask.sum(dim=1, keepdim=True) - 1
    last_indices = last_indices.unsqueeze(-1).expand(-1, -1, model.D)
    last_hidden_state = output.last_hidden_state.gather(dim=1, index=last_indices).squeeze(1)
    logit = linear(last_hidden_state).squeeze(-1)
    logit_calibrated = logit / t
    score_calibrated = logit_calibrated.sigmoid()
    
    return score_calibrated[0].item()

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


def parse_args(ret='parsed'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='playground/dev/coco_25k_caps_train3_cleaned.json')
    parser.add_argument('--num_caps', type=int, default=None, help='choose first N captions from the file')
    parser.add_argument('--cap_start_idx', type=int, default=0, help='start index for captions')
    parser.add_argument('--base_seed', type=int, default=44)
    parser.add_argument('--job_idx', type=int, default=0)
    parser.add_argument('--num_jobs', type=int, default=1)
    parser.add_argument('--flush_every', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='data/coco_syn/coco_caps_neg')
    
    if ret == 'parsed':
        return parser.parse_args()
    elif ret == 'default':
        return parser.parse_args([])
    elif ret == 'parser':
        return parser
    else:
        raise ValueError(f"Invalid return type: {ret}")
    

def genfn(messages, pipeline, pipeline_params, apply_chat_template_params):
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        **apply_chat_template_params
    )
    outputs = pipeline(prompt, **pipeline_params)
    return outputs

def get_diff_score(caption, neg_caption, pipeline, pipeline_params, apply_chat_template_params):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Please respond to the user's questions succinctly."},
        {"role": "user", "content": f'{DIFF_PROMPT.format(caption=caption, neg_caption=neg_caption)}'},
    ]
    outputs = genfn(messages, pipeline, pipeline_params, apply_chat_template_params)
    diff_score = outputs[0]["generated_text"].strip()
    return diff_score

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the args
    if args.job_idx == 0:
        save_args = vars(args)
        save_args['sha'] = get_sha()
        with open(Path(args.output_dir) / 'args.json', 'w') as f:
            json.dump(save_args, f, indent=4)
    
    out_log_file = open(Path(args.output_dir) / f'out_{args.job_idx}.log', 'w')
    out_caps_file = open(Path(args.output_dir) / f'caps_{args.job_idx}.jsonl', 'w')
    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    # pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"quantization_config": nf4_config}, device_map="auto")
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
    
    with open(args.caption_file, 'r') as f:
        img_cap_pairs = json.load(f)
        
    if args.num_caps is None:
        args.num_caps = len(img_cap_pairs)
        
    img_cap_pairs = img_cap_pairs[args.cap_start_idx:args.cap_start_idx+args.num_caps]
    
    caps_per_job = int(np.ceil(len(img_cap_pairs) / args.num_jobs))
    job_img_cap_pairs = img_cap_pairs[args.job_idx * caps_per_job: (args.job_idx + 1) * caps_per_job]
    
    print('START TIME : ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file=out_log_file)
    start_time = time.time()
    
    for i, img_cap_pair in tqdm(enumerate(job_img_cap_pairs), total=len(job_img_cap_pairs)):
        img_path = img_cap_pair['img_path']
        caption = img_cap_pair['cleaned_caption'] if 'cleaned_caption' in img_cap_pair else img_cap_pair['caption']
        print(f'############# Caption {i+1}/{len(job_img_cap_pairs)} ##############', file=out_log_file)
        print(f'Image path: {img_path}', file=out_log_file)
        print(f'Caption: {caption}', file=out_log_file)
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Please respond to the user's questions succinctly."},
            {"role": "user", "content": f'{NEG_PROMPT.format(caption=caption)}'},
        ]
        print(f'USER: {NEG_PROMPT.format(caption=caption)}', file=out_log_file)
        
        all_cap_feedback = []
        pos_cap_words = caption_words(caption)
        
        prev_grammar_score = None
        prev_plaus_score = None
        
        for iter in range(NUM_ITERS):
            outputs = genfn(messages, pipeline, pipeline_params, apply_chat_template_params)
            print(f'MODEL: {outputs[0]["generated_text"]}', file=out_log_file)
            neg_caption_idx = outputs[0]["generated_text"].find('Final Output Caption:') + len('Final Output Caption:')
            neg_caption = outputs[0]["generated_text"][neg_caption_idx:].split('\n')[0].strip() # get everything upto newline
            if neg_caption == 'NA':
                all_cap_feedback.append({
                    'neg_caption' : neg_caption,
                })
                break
            
            plaus_score = get_plausibility_score(neg_caption, **plausibility_model)
            grammar_score = get_grammar_score(neg_caption, **grammar_model)
            
            # Not a numeric score. Just Yes/No
            diff_score = get_diff_score(
                caption, neg_caption, pipeline, pipeline_params, apply_chat_template_params) 
            neg_cap_words = caption_words(neg_caption)
                
            extra_words = set(neg_cap_words) - set(pos_cap_words)
            missing_words = set(pos_cap_words) - set(neg_cap_words)
            
            exact_same = pos_cap_words == neg_cap_words
            
            messages.append({"role": "assistant", "content": outputs[0]["generated_text"]})
            
            user_mesg = "FEEDBACK:\n"
            if exact_same:
                user_mesg += "Your output caption is exactly the same as the original caption. Can you please try again?\n"
            else:
                user_mesg += get_score_feedback(grammar_score, prev_grammar_score, "grammar")
                user_mesg += get_score_feedback(plaus_score, prev_plaus_score, "plausibility")
                user_mesg += "Is the output caption visually different from the original caption? : " + diff_score + "\n"
                
                if len(extra_words) > 0:
                    user_mesg += f"Your output caption has extra words (lemmatized): {extra_words}.\n"
                if len(missing_words) > 0:
                    user_mesg += f"Your output caption has missing words (lemmatized): {missing_words}.\n"
                user_mesg += "Can you please try again?\n"
            
            curr_feedback = {
                'caption': caption,
                'neg_caption': neg_caption,
                'exact_same': exact_same,
                'grammar_score': grammar_score,
                'plausibility_score': plaus_score,
                'diff_score': diff_score,
                'extra_words': list(extra_words),
                'missing_words': list(missing_words),
                'pos_cap_words': pos_cap_words,
                'neg_cap_words': neg_cap_words,
                'feedback_str': user_mesg
            }
            all_cap_feedback.append(curr_feedback)
            
            prev_grammar_score = grammar_score
            prev_plaus_score = plaus_score
            
            print(f'USER: {user_mesg}', file=out_log_file)
            messages.append({"role": "user", "content": user_mesg})
    
        print(json.dumps(all_cap_feedback), file=out_caps_file)
        print('#'*80, file=out_log_file)
            
        if (i+1)%args.flush_every == 0: # flush every few iterations
            out_caps_file.flush()
            out_log_file.flush()
    
    end_time = time.time()
    print('END TIME : ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file=out_log_file)
    print('TOTAL TIME TAKEN : ', time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)), file=out_log_file)
    
    out_log_file.close()
    out_caps_file.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)