import os
import sys
sys.path.append(os.path.abspath('.'))

import transformers
import torch
import json
from tqdm import tqdm

from transformers import BitsAndBytesConfig
import numpy as np

RNG = np.random.RandomState(44)

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
# pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"quantization_config": nf4_config}, device_map="auto")

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

def genfn(messages):
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        **apply_chat_template_params
    )
    outputs = pipeline(prompt, **pipeline_params)
    return outputs

def test_yes(outputs):
    return outputs[0]["generated_text"].startswith("yes") or outputs[0]["generated_text"].startswith("Yes")

def get_neg_caption(messages, outputs):
    messages.append({"role": "assistant", "content": outputs[0]["generated_text"]})
    messages.append({"role": "user", "content": "Can you use the rule to generate the negative caption?"})
    outputs = genfn(messages)
    
    messages.append({"role": "assistant", "content": outputs[0]["generated_text"]})
    messages.append({"role": "user", "content": "Can you output only the negative caption in json format?"})
    
    outputs = genfn(messages)
    return outputs[0]["generated_text"]

rules = [
    "**Interchange colors or attributes between objects**\n- Example: 'a pink bird with a white beak' → 'a white bird with a pink beak'",
    "**Switch quantities or numbers**\n- Example: 'there are three bananas and two apples' → 'there are two bananas and three apples'",
    "**Invert the spatial relationship between objects**\n- Example: 'some plants surrounding a lightbulb' → 'a lightbulb surrounding some plants'",
    "**Swap the subject and object of the sentence**\n- Example: 'I had cleaned my car' → 'I had my car cleaned'",
    "**Reverse the order or position of elements**\n- Example: 'the happy person is on the right and the sad person is on the left' → 'the sad person is on the right and the happy person is on the left'",
    "**Reverse the relationship between elements in idiomatic expressions**\n- Example: 'fishing for compliments' → 'compliments for fishing'",
]

# rules_str = """
# 1.  **Interchange colors or attributes between objects**
#    - Example: "a pink bird with a white beak" → "a white bird with a pink beak" 

# 2. **Switch quantities or numbers**
#    - Example: "there are three bananas and two apples" → 
#      "there are two bananas and three apples"

# 3. **Invert the spatial relationship between objects**
#    - Example: "some plants surrounding a lightbulb" → 
#      "a lightbulb surrounding some plants"

# 4. **Swap the subject and object of the sentence**
#    - Example: "I had cleaned my car" → "I had my car cleaned"

# 5. **Reverse the order or position of elements**
#    - Example: "the happy person is on the right and the sad person is on the left" → 
#      "the sad person is on the right and the happy person is on the left"

# 6. **Reverse the relationship between elements in idiomatic expressions**
#    - Example: "fishing for compliments" → "compliments for fishing"
# """

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


# first_mesg = """
# Can you generate a negative caption for the given original caption? The negative caption should use the **same set of words as the original caption**, but create a semantically different meaning. The negative caption should be grammatical and meaningful. Here is the original caption: 
# ```
# {{"caption": "{caption}"}}
# ```
# I can provide a set of rules that you may follow to generate the negative caption. Here is the first one: 
# ```
# {rule}
# ```
# Can you first tell me if this rule is applicable to the caption? Answer yes or no.
# """

first_mesg = """
Can you generate a negative caption for the given original caption? The negative caption should use the **same set of words as the original caption**, but create a semantically different meaning. The negative caption should be grammatical and meaningful. Here is the original caption: 
```
{{"caption": "{caption}"}}
```
Here is a set of rules that you may follow to generate the negative caption: 
```
{rules_str}
```
For each rule first reason whether it is applicable to the caption.
"""

# next_mesg = """
# Is this rule applicable? Answer yes or no.
# ```
# {rule}
# ```
# """

picking_instr = """
Pick the rule that was the most applicable, i.e., generated a negative caption that was grammatical, meaningful, used the same set of words as the original caption and was semantically different. Output only the negative caption in json format. Output \"impossible\" if no rule was applicable.
"""

# picking_instr = """
# Pick the first rule that was applicable and generate the negative caption using that rule. Only output the negative caption in json format. Output \"impossible\" if no rule was applicable.
# """

all_neg_captions = [None] * len(all_captions)

for i, caption in enumerate(all_captions):
    print(f'############# Caption {i+1}/{len(all_captions)} ##############')
    curr_rules = RNG.permutation(rules)
    
    ################ Using rules in context one by one ################
    # rule = curr_rules[0]
    # messages = [
    #     {"role": "system", "content": "You are a helpful AI assistant. Please respond to the user's questions succinctly."},
    #     # {"role": "user", "content": "Can you combine the the words 'pizza' and 'wine glass' in multiple ways using conjunctions? Output only 5 such combinations separated by semi-colons. Do not output anything else."},
    #     {"role": "user", "content": first_mesg.format(caption=caption, rule=rule)},
    # ]
    # outputs = genfn(messages)
    
    # if test_yes(outputs):
    #     all_neg_captions[i] = get_neg_caption(messages, outputs)
    # else:
    #     for rule in curr_rules[1:]:
    #         messages.append({"role": "assistant", "content": outputs[0]["generated_text"]})
    #         messages.append({"role": "user", "content": next_mesg.format(rule=rule)})
    #         outputs = genfn(messages)
    #         if test_yes(outputs):
    #             all_neg_captions[i] = get_neg_caption(messages, outputs)
    #             break
    ###################################################################
    
    ################ Using only one rule in context at a time ################
    # for j, rule in enumerate(curr_rules):
    #     messages = [
    #         {"role": "system", "content": "You are a helpful AI assistant. Please respond to the user's questions succinctly."},
    #         {"role": "user", "content": first_mesg.format(caption=caption, rule=rule)},
    #     ]
    #     outputs = genfn(messages)
    #     if test_yes(outputs):
    #         all_neg_captions[i] = get_neg_caption(messages, outputs)
    #         break
    ###################################################################
    
    ################ All rules together ################
    rules_str = "\n".join([f'{j+1}. {rule}' for j, rule in enumerate(curr_rules)])
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Please respond to the user's questions succinctly."},
        {"role": "user", "content": first_mesg.format(caption=caption, rules_str=rules_str)},
    ]
    outputs = genfn(messages)
    print('MODEL REASONING : \n', outputs[0]["generated_text"])
    
    messages.append({"role": "assistant", "content": outputs[0]["generated_text"]})
    messages.append({"role": "user", "content": picking_instr})
    outputs = genfn(messages)
    all_neg_captions[i] = outputs[0]["generated_text"]
    ###################################################################
    
    # print(f'[{i+1}/{len(all_captions)}]')
    # print(f'Used rule ({j+1}/{len(curr_rules)}):\n {rule}')
    print(f"Negative caption : {all_neg_captions[i]}")
    print('#############################################################')