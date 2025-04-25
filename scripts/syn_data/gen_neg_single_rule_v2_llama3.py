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

from prompts.swap_obj import SWAP_OBJ_PROMPT_NO_INCONTEXT
from prompts.swap_att import SWAP_ATT_PROMPT_NO_INCONTEXT

rule_dict = {
    'swap_obj': SWAP_OBJ_PROMPT_NO_INCONTEXT,
    'swap_att': SWAP_ATT_PROMPT_NO_INCONTEXT
}

OUTPUT_PROMPT = """
Can you output only the new sentence and nothing else? Output "NA" if the answer to the first question was No. 
"""

def get_sha():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha

def genfn(messages, pipeline, pipeline_params, apply_chat_template_params):
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        **apply_chat_template_params
    )
    outputs = pipeline(prompt, **pipeline_params)
    return outputs

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
    parser.add_argument('--rule', type=str, default='swap_obj', choices=list(rule_dict.keys()))
    
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
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the args
    if args.job_idx == 0:
        save_args = vars(args)
        save_args['sha'] = get_sha()
        with open(Path(args.output_dir) / 'args.json', 'w') as f:
            json.dump(save_args, f, indent=4)
    
    out_log_file = open(Path(args.output_dir) / f'out_{args.job_idx}.log', 'w')
    out_caps_file = open(Path(args.output_dir) / f'caps_{args.job_idx}.jsonl', 'w')    
    
    with open(args.caption_file, 'r') as f:
        img_cap_pairs = json.load(f)
        
    if args.num_caps is None:
        args.num_caps = len(img_cap_pairs)
    
    img_cap_pairs = img_cap_pairs[args.cap_start_idx:args.cap_start_idx+args.num_caps]
    
    caps_per_job = int(np.ceil(len(img_cap_pairs) / args.num_jobs))
    job_img_cap_pairs = img_cap_pairs[args.job_idx * caps_per_job: (args.job_idx + 1) * caps_per_job]
    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation", model=model_id, 
        model_kwargs={"quantization_config": nf4_config}, 
        # model_kwargs={"torch_dtype": torch.bfloat16}, 
        device_map="auto")
    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    pipeline_params = {
        "max_new_tokens": 2048,
        "eos_token_id": terminators,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "return_full_text": False,
    }
    apply_chat_template_params = {
        "tokenize": False,
        "add_generation_prompt": True
    }
    RNG = np.random.RandomState(args.base_seed+args.job_idx)

    for i, img_cap_pair in tqdm(enumerate(job_img_cap_pairs), total=len(job_img_cap_pairs)):
        img_path = img_cap_pair['img_path']
        caption = img_cap_pair['cleaned_caption'] if 'cleaned_caption' in img_cap_pair else img_cap_pair['caption']
        print(f'############# Caption {i+1}/{len(job_img_cap_pairs)} ##############', file=out_log_file)
        print(f'Image path: {img_path}', file=out_log_file)
        print(f'Caption: {caption}', file=out_log_file)
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Please respond to the user's questions succinctly."},
            {"role": "user", "content": rule_dict[args.rule].format(caption=caption)},
        ]
        outputs = genfn(messages, pipeline, pipeline_params, apply_chat_template_params)
        print('INIT MODEL OUTPUT : \n', outputs[0]["generated_text"], file=out_log_file)
        
        messages.append({"role": "user", "content": OUTPUT_PROMPT})
        outputs = genfn(messages, pipeline, pipeline_params, apply_chat_template_params)
        print('FINAL MODEL OUTPUT : \n', outputs[0]["generated_text"], file=out_log_file)
        
        # start_idx = outputs[0]["generated_text"].find('Output:') + len('Output:')
        neg_caption = outputs[0]["generated_text"].strip()
        
        out_json = {
            'img_path': img_path,
            'caption': caption,
            'neg_caption': neg_caption,
            'model_output': outputs[0]["generated_text"]
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