import os
import sys
sys.path.append(os.path.abspath('.'))
import transformers
import torch
import json
from tqdm import tqdm

caption_path = 'playground/dev/coco_train_caps.json'
outpath = 'playground/dev/coco_train_caps_cleaned.json'
all_img_captions = json.load(open(caption_path, 'r'))
all_captions = [img_caption['caption'] for img_caption in all_img_captions]

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
all_messages = [[
    {"role": "system", "content": "You are a helpful AI assistant. Please respond to the user's questions succinctly."},
    # {"role": "user", "content": f"Can you clean up any typos or grammatical errors in the following? Output only the corrected caption (the same caption if there are no errors) and nothing else in json format : {{\"caption\": \"{caption}\"}}"},
    {"role": "user", "content": f"Can you clean up any typos or grammatical errors in the following? Output only the corrected caption (the same caption if there are no errors) and nothing else in json format : {json.dumps({'caption': caption})}"},
] for caption in all_captions]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
pipeline.tokenizer.padding_side = "left"

all_prompts = [pipeline.tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True,
    # padding=True,
) for messages in all_messages]

batch_size = 128
with torch.inference_mode():
    for i in tqdm(range(0, len(all_prompts), batch_size), total=len(all_prompts)//batch_size):
        outputs = pipeline(
            all_prompts[i:i+batch_size],
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.6,
            top_p=0.9,
            batch_size=batch_size,
            return_full_text=False,
        )

        for j, output in enumerate(outputs):
            try:
                cleaned_caption = json.loads(output[0]["generated_text"], )['caption']
                all_img_captions[i+j]['cleaned_caption'] = cleaned_caption
            except Exception as e:
                print(f"Error at idx {i+j}: {e}")
                print('Generated text:')
                print(output[0]["generated_text"])
                all_img_captions[i+j]['llm_output'] = output[0]["generated_text"]
        
        if (i+1) % 10 == 0:
            with open(outpath, 'w') as f:
                json.dump(all_img_captions, f, indent=4)
                
with open(outpath, 'w') as f:
    json.dump(all_img_captions, f, indent=4)
