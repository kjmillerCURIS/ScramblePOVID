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


# FIRST_MESG = """
# I will provide you with a caption and a negative caption. Can you check if the following rules hold?
# 1. The negative caption should have the same set of words as the original caption. You can allow for a new preposition or one new word in some cases.
# 2. The negative caption should have a semantically different meaning than the original caption.
# 3. The negative caption should be grammatical and meaningful.

# Here are the caption and the negative caption. Do the above rules hold? Please answer yes or no.
# ```
# {{"caption": "{caption}"}}
# {{"neg_caption": "{neg_caption}"}}
# ```
# """

# You can allow for at most one new word in some cases.
FIRST_MESG = """
I will provide you with a caption and a negative caption. Can you check if the following rules hold?
1. The negative caption should have the same set of words as the original caption possibly in a different order.
2. The negative caption should have a semantically different meaning than the original caption.
3. The negative caption should be grammatical, meaningful and should be possible in a realistic scenario.

Here are the caption and the negative caption. Do the above rules hold? Please answer yes or no.
```
{{"caption": "{caption}"}}
{{"neg_caption": "{neg_caption}"}}
```
"""

FIRST_MESG_v2 = """
I will provide you with a caption and a negative caption. Can you check if the following rules hold?
1. The negative caption should have the same set of words as the original caption possibly in a different order.
2. The negative caption should correspond to a scene with visual differences compared to the original caption. It should not simply be a different way of describing the same exact scene.
3. The negative caption should be grammatical, meaningful and should be possible in a realistic scenario.

Here are the caption and the negative caption. For the captions, first analyze each rule and state whether it holds or not.
```
{{"caption": "{caption}"}}
{{"neg_caption": "{neg_caption}"}}
```
"""

OUTPUT_MESG = """
Do all the rules hold? Please answer yes or no.
"""

def genfn(messages, pipeline, pipeline_params, apply_chat_template_params):
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        **apply_chat_template_params
    )
    outputs = pipeline(prompt, **pipeline_params)
    return outputs

def parse_args(ret='parsed'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='data/coco_syn/coco_caps_neg_v2/caps_0.jsonl')
    parser.add_argument('--job_idx', type=int, default=None)
    parser.add_argument('--flush_every', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default=None)
    
    if ret == 'parsed':
        return parser.parse_args()
    elif ret == 'default':
        return parser.parse_args([])
    elif ret == 'parser':
        return parser
    else:
        raise ValueError(f"Invalid return type: {ret}")


def main(args):
    # Create output directory
    if args.output_dir is None:
        args.output_dir = Path(args.caption_file).parent
    os.makedirs(args.output_dir, exist_ok=True)
    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation", model=model_id, 
        model_kwargs={"quantization_config": nf4_config}, 
        device_map="auto")
    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    pipeline_params = {
        "max_new_tokens": 4096,
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
    all_caps = []
    with open(args.caption_file, 'r') as f:
        for line in f.readlines():
            all_caps.append(json.loads(line))
    
    out_log_file = open(Path(args.output_dir) / f'out_tested_{args.job_idx}.log', 'w')
    out_caps_file = open(Path(args.output_dir) / f'caps_tested_{args.job_idx}.jsonl', 'w')

    for i, img_cap_pair in tqdm(enumerate(all_caps), total=len(all_caps)):
        img_path = img_cap_pair['img_path']
        caption = img_cap_pair['caption']
        neg_caption = img_cap_pair['neg_caption']
        print(f'############# Caption {i+1}/{len(all_caps)} ##############', file=out_log_file)
        print(f'Image path: {img_path}', file=out_log_file)
        print(f'Caption: {caption}', file=out_log_file)
        print(f'Negative caption: {neg_caption}', file=out_log_file)
        
        if neg_caption.lower().startswith('impossible'):
            print('Negative caption is impossible. Skipping.', file=out_log_file)
            caption_corr = None
        else:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Please respond to the user's questions succinctly."},
                {"role": "user", "content": FIRST_MESG_v2.format(caption=caption, neg_caption=neg_caption)},
            ]
            outputs = genfn(messages, pipeline, pipeline_params, apply_chat_template_params)
            print('MODEL REASONING : \n', outputs[0]["generated_text"], file=out_log_file)
            
            messages.append({"role" : "assistant", "content" : outputs[0]["generated_text"]})
            messages.append({"role" : "user", "content" : OUTPUT_MESG})
            outputs = genfn(messages, pipeline, pipeline_params, apply_chat_template_params)
            
            print('MODEL OUTPUT : \n', outputs[0]["generated_text"], file=out_log_file)
            caption_corr = outputs[0]["generated_text"]
        
        out_json = {
            'img_path': img_path,
            'caption': caption,
            'neg_caption': neg_caption,
            'caption_corr': caption_corr
        }
            
        print(json.dumps(out_json), file=out_caps_file)
        print('#'*80, file=out_log_file)
        
        if (i+1)%args.flush_every == 0: # flush every few iterations
            out_caps_file.flush()
            out_log_file.flush()
    
    out_log_file.close()
    out_caps_file.close()
    

if __name__ == "__main__":
    args = parse_args()
    main(args)