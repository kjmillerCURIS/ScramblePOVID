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



FIRST_MESG = """
Following are captions for two images: 
```
{{"caption1": "{caption}"}}
{{"caption2": "{neg_caption}"}}
```
Are there any visual differences between the two images? Please respond yes or no.
"""

# OUTPUT_MESG = """
# Do all the rules hold? Please answer yes or no.
# """

def genfn(messages, pipeline, pipeline_params, apply_chat_template_params):
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        **apply_chat_template_params
    )
    outputs = pipeline(prompt, **pipeline_params)
    return outputs

def parse_args(ret='parsed'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='data/coco_syn/coco_caps_neg_v2/adv_ref/adversarial_refine.json')
    parser.add_argument('--num_jobs', type=int, default=1)
    parser.add_argument('--job_idx', type=int, default=0)
    parser.add_argument('--flush_every', type=int, default=2)
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
        "max_new_tokens": 256,
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
    
    all_caps = json.load(open(args.caption_file, 'r'))
    caps_per_job = int(np.ceil(len(all_caps) / args.num_jobs))
    job_caps = all_caps[args.job_idx * caps_per_job: (args.job_idx+1) * caps_per_job]
    
    out_log_file = open(Path(args.output_dir) / f'out_paraphrase_test_{args.job_idx}.log', 'w')
    out_caps_file = open(Path(args.output_dir) / f'caps_paraphrase_test_{args.job_idx}.jsonl', 'w')

    for i, img_cap_pair in tqdm(enumerate(job_caps), total=len(job_caps)):
        img_path = img_cap_pair['img_path']
        caption = img_cap_pair['caption']
        neg_caption = img_cap_pair['neg_caption']
        print(f'############# Caption {i+1}/{len(job_caps)} ##############', file=out_log_file)
        print(f'Image path: {img_path}', file=out_log_file)
        print(f'Caption: {caption}', file=out_log_file)
        print(f'Negative caption: {neg_caption}', file=out_log_file)
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Please respond to the user's questions succinctly."},
            {"role": "user", "content": FIRST_MESG.format(caption=caption, neg_caption=neg_caption)},
        ]
        outputs = genfn(messages, pipeline, pipeline_params, apply_chat_template_params)
        
        print('MODEL OUTPUT : \n', outputs[0]["generated_text"], file=out_log_file)
        
        if outputs[0]["generated_text"].strip().lower().startswith('yes'):
            viz_diff = True
        elif outputs[0]["generated_text"].strip().lower().startswith('no'):
            viz_diff = False
        else:
            viz_diff = None
        
        llm_out = outputs[0]["generated_text"]
        
        out_json = {
            'img_path': img_path,
            'caption': caption,
            'neg_caption': neg_caption,
            'viz_diff': viz_diff,
            'llm_out': llm_out
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