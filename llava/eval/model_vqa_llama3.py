import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

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
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions, total=len(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        cur_prompt = qs = line["text"]
        qs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': qs}]}]
        qs = processor.apply_chat_template(qs, add_generation_prompt=True)
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        
        inputs = processor(
            images = image,
            text = qs,
            return_tensors = 'pt',
        ).to(model.device)
        
        # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            # do_sample=True if args.temperature > 0 else False,
            # temperature=args.temperature,
            # top_p=args.top_p,
            # num_beams=args.num_beams
        )
            
        generated_text = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": generated_text,
                                   "answer_id": ans_id,
                                   "model_id": model_path,
                                   "metadata": {}}) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--temperature", type=float, default=0)
    # parser.add_argument("--top_p", type=float, default=None)
    # parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
