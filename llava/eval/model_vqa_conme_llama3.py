import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import pandas as pd
import re

# from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path

from transformers import MllamaProcessor, MllamaForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Model
    # disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    # model_name = args.model_name or get_model_name_from_path(model_path)
    
    model = MllamaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
    processor = MllamaProcessor.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
    
    # Load ConMe dataset
    ques_df = pd.read_csv(os.path.join(args.question_root, f'{args.split}.csv'))
    
    if 'HUMAN_FILTER' in args.split:
        ques_df['text_options'] = ques_df['text_options'].apply(
            lambda x: [match[1] for match in re.findall(r'(["\'])(.*?)\1', x)]
        )
        question = ques_df['question'].tolist()
        image = ques_df['image'].tolist()
        question_id = ques_df['id'].tolist()
        text_options = ques_df['text_options'].tolist()
        zip_list = list(zip(question, image, question_id, text_options))
    else:
        question = ques_df['question'].tolist()
        image = ques_df['image'].tolist()
        question_id = ques_df.index.tolist()
        zip_list = list(zip(question, image, question_id))
    
    questions = get_chunk(zip_list, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    print('Creating answers file at', answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions, total=len(questions)):
        idx = line[2]
        image_file = line[1]
        qs = line[0]
        
        if 'HUMAN_FILTER' in args.split:
            text_options = line[3]
            # Format question with options
            letters_2options = ['A. ', 'B. ']
            qs = (qs + '\n' + '\n'.join(
                [letters_2options[i] + text_options[i] for i in range(len(text_options))])
                + '\n' + "Answer with the option's letter from the given choices directly.")
        else:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
        
        cur_prompt = qs
        chat_input = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': qs}]}]
        formatted_prompt = processor.apply_chat_template(chat_input, add_generation_prompt=True)
        
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        
        inputs = processor(
            images = image,
            text = formatted_prompt,
            return_tensors = 'pt',
        ).to(model.device)
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
            )
                
        generated_text = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": generated_text,
            "answer_id": ans_id,
            "model_id": model_path,
            "metadata": {}
        }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/projectnb/ivc-ml/samarth/datasets/COCO/images/")
    parser.add_argument("--question-root", type=str, default="playground/data/eval/conme")
    parser.add_argument("--answers-file", type=str, default="playground/data/eval/conme/answers/replace-att_HUMAN_FILTER/llava-v1.5-13b/1_0.jsonl")
    parser.add_argument("--split", type=str, default="replace-att_HUMAN_FILTER")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--temperature", type=float, default=0)
    # parser.add_argument("--top_p", type=float, default=None)
    # parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
