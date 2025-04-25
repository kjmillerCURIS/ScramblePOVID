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

def get_sha():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha

RULES = [
    "**Interchange colors or attributes between objects**\n- Example: 'a pink bird with a white beak' → 'a white bird with a pink beak'",
    "**Switch quantities or numbers**\n- Example: 'there are three bananas and two apples' → 'there are two bananas and three apples'\n- Example: 'there are more skiers than snowboarders' → 'there are more snowboarders than skiers'",
    "**Invert the spatial relationship between objects**\n- Example: 'some plants surrounding a lightbulb' → 'a lightbulb surrounding some plants'",
    "**Swap the subject and object of the sentence**\n- Example: 'I had cleaned my car' → 'I had my car cleaned'\n- Example: 'a bottle is in water' → 'water is in a bottle'\n- Example: 'manning a ship' → 'shipping a man'",
    "**Reverse the order or position of elements**\n- Example: 'the happy person is on the right and the sad person is on the left' → 'the sad person is on the right and the happy person is on the left'\n- Example: 'the red car is behind the blue car' → 'the blue car is behind the red car'",
    "**Reverse the relationship between elements in idiomatic expressions**\n- Example: 'fishing for compliments' → 'compliments for fishing'",
]

# **Reverse the order, position, or spatial relationship of elements**
# changing above to **Reverse the spatial relationship of elements**

RULES_V2 = [
# """**Shift the agent of the action while maintaining the subject's involvement**
#    - Example: "I had cleaned my car" → "I had my car cleaned"
#      (Changes from the subject directly performing the action to the subject causing the action to be performed by an implied agent)
#    - Example: "She had baked a cake" → "She had a cake baked"
#      (Again, shifts from direct action to arranged action))
# """,
"""**Reverse the spatial relationship of elements**
   - Example: "A bottle is in water" → "Water is in a bottle"
   - Example: "The red car is behind the blue car" → "The blue car is behind the red car"
   - Example: "The happy person is on the right and the sad person is on the left" → 
     "The sad person is on the right and the happy person is on the left"
   - Example: "Some plants surrounding a lightbulb" → "A lightbulb surrounding some plants"
""",
"""**Transform parts of speech and reverse relationships in expressions**
   - Example: "Manning a ship" → "Shipping a man"
     (Turns the verb "manning" into the object "man", and the object "ship" into the verb "shipping")
   - Example: "Fishing for compliments" → "Compliments for fishing"
     (Reverses the relationship between elements in the idiomatic expression)
""",
"""**Switch quantities or numbers**
   - Example: "There are three bananas and two apples" → 
     "There are two bananas and three apples"
   - Example: "There are more skiers than snowboarders" → 
     "There are more snowboarders than skiers"
""",
"""**Interchange colors or attributes between objects**
   - Example: "A pink bird with a white beak" → "A white bird with a pink beak"
""",
]

FIRST_MESG = """
Can you generate a negative caption for the provided original caption? The negative caption should use the **same set of words as the original caption**, but create a **semantically different meaning**. The negative caption should be **grammatical**, **meaningful** and **make sense in a realistic scenario**. Here is the original caption: 
Here is a set of transformations. You could apply one of these transformations to the original caption to generate a negative caption: 
```markdown

# Possible Transformations for Generating Negative Captions

{rules_str}
```
For each transformation first state the reason why it may or may not be applicable to the caption. If applicable, output the transformed caption. If not, state the reason. If none of the transformations are applicable, simply state so at the end. Following is the original caption:
```
{{"caption": "{caption}"}}
```
"""

PICKING_INSTR = """
Copy and output the most appropriate negative caption from above : one that conveys a semantically different meaning than the original, is grammatical, meaningful, and uses the same set of words as the original caption. If no transformation was applicable simply state "impossible".
"""

OUTPUT_INSTR = """
Output only the negative caption you picked in json format. Output {"caption": "impossible"} if none of the transformations were applicable.
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
    parser.add_argument('--caption_file', type=str, default='playground/dev/coco_25k_caps_train3_cleaned.json')
    parser.add_argument('--num_caps', type=int, default=10000, help='choose first N captions from the file')
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
    img_cap_pairs = img_cap_pairs[args.cap_start_idx:args.cap_start_idx+args.num_caps]
    
    caps_per_job = int(np.ceil(len(img_cap_pairs) / args.num_jobs))
    job_img_cap_pairs = img_cap_pairs[args.job_idx * caps_per_job: (args.job_idx + 1) * caps_per_job]
    
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
        caption = img_cap_pair['cleaned_caption']
        print(f'############# Caption {i+1}/{len(job_img_cap_pairs)} ##############', file=out_log_file)
        print(f'Image path: {img_path}', file=out_log_file)
        print(f'Caption: {caption}', file=out_log_file)
        
        curr_rules = RNG.permutation(RULES_V2)
        rules_str = "\n".join([f'{j+1}. {rule}' for j, rule in enumerate(curr_rules)])
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Please respond to the user's questions succinctly."},
            {"role": "user", "content": FIRST_MESG.format(caption=caption, rules_str=rules_str)},
        ]
        outputs = genfn(messages, pipeline, pipeline_params, apply_chat_template_params)
        print('MODEL REASONING : \n', outputs[0]["generated_text"], file=out_log_file)
        
        messages.append({"role": "assistant", "content": outputs[0]["generated_text"]})
        messages.append({"role": "user", "content": PICKING_INSTR})
        outputs = genfn(messages, pipeline, pipeline_params, apply_chat_template_params)
        
        print('MODEL PICKING : \n', outputs[0]["generated_text"], file=out_log_file)
        
        messages.append({"role": "assistant", "content": outputs[0]["generated_text"]})
        messages.append({"role": "user", "content": OUTPUT_INSTR})
        outputs = genfn(messages, pipeline, pipeline_params, apply_chat_template_params)
        
        print('MODEL OUTPUT : \n', outputs[0]["generated_text"], file=out_log_file)
        
        neg_caption_json = outputs[0]["generated_text"]
        
        try:
            neg_caption = json.loads(neg_caption_json)
            neg_caption = neg_caption['caption']
        except:
            print(f'[ERROR] could not parse json : {neg_caption_json}', file=out_log_file)
            neg_caption = None
        
        out_json = {
            'img_path': img_path,
            'caption': caption,
            'neg_caption': neg_caption
        }
        if out_json['neg_caption'] is None:
            out_json['neg_caption_json'] = neg_caption_json
            
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