import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import pandas as pd
import re
from pathlib import Path


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name or get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

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
    # read questions from csv and output answers to new jsonl file
    # TODO : this way would require a postprocess step to merge the answers and evaluate, similar to scripts/eval/convert_seed_for_submission.py
    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(zip_list, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    print('Creating answers file at', answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        idx = line[2]
        image_file = line[1]
        qs = line[0]
        
        if 'HUMAN_FILTER' in args.split:
            text_options = line[3]
            letters_2options = ['A. ', 'B. ']
            qs = (qs + '\n' + '\n'.join(
                [letters_2options[i] + text_options[i] for i in range(len(text_options))])
                                    + '\n' + "Answer with the option's letter from the given choices directly.")
        else:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=10,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/projectnb/ivc-ml/samarth/datasets/COCO/images/")
    parser.add_argument("--question-root", type=str, default="playground/data/eval/conme")
    parser.add_argument("--answers-file", type=str, default="playground/data/eval/conme/answers/replace-att_HUMAN_FILTER/llava-v1.5-13b/1_0.jsonl")
    parser.add_argument("--split", type=str, default="replace-att_HUMAN_FILTER")
    
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
