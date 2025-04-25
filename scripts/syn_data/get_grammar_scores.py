import os
import sys
sys.path.append(os.path.abspath('.'))

import transformers
import torch
import json
from tqdm import tqdm

import argparse
from pathlib import Path

def genfn(messages, pipeline, pipeline_params, apply_chat_template_params):
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        **apply_chat_template_params
    )
    outputs = pipeline(prompt, **pipeline_params)
    return outputs

def parse_args(ret='parsed'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='data/coco_train_syn/coco_caps_neg_cot/combined_caps.json')
    parser.add_argument('--batch_size', type=int, default=256)
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
    
    tokenizer = transformers.AutoTokenizer.from_pretrained('textattack/distilbert-base-uncased-CoLA')
    model = transformers.DistilBertForSequenceClassification.from_pretrained('textattack/distilbert-base-uncased-CoLA')
    model.to(device='cuda')
    model.eval()
    
    all_caps = json.load(open(args.caption_file, 'r'))
    num_batches = len(all_caps) // args.batch_size
    if len(all_caps) % args.batch_size != 0:
        num_batches += 1

    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            batch_caps = all_caps[i*args.batch_size: (i+1)*args.batch_size]
            pos_toks = tokenizer.batch_encode_plus(
                [cap['caption'] for cap in batch_caps], 
                return_tensors='pt', 
                padding='longest', 
                truncation='longest_first', 
                max_length=128
            )
            neg_toks = tokenizer.batch_encode_plus(
                [cap['neg_caption'] for cap in batch_caps], 
                return_tensors='pt', 
                padding='longest', 
                truncation='longest_first', 
                max_length=128
            )
            pos_input_ids = pos_toks.input_ids.to(model.device)
            pos_attention_mask = pos_toks.attention_mask.to(model.device)
            neg_input_ids = neg_toks.input_ids.to(model.device)
            neg_attention_mask = neg_toks.attention_mask.to(model.device)
            
            pos_output = model(pos_input_ids, attention_mask=pos_attention_mask)
            pos_score = pos_output['logits'].softmax(dim=1)[:, 1]
            
            neg_output = model(neg_input_ids, attention_mask=neg_attention_mask)
            neg_score = neg_output['logits'].softmax(dim=1)[:, 1]            
            
            for j in range(len(batch_caps)):
                all_caps[i*args.batch_size + j]['pos_grammar_score'] = pos_score[j].item()
                all_caps[i*args.batch_size + j]['neg_grammar_score'] = neg_score[j].item()
        
    with open(Path(args.output_dir) / f'grammar_scores_textattack.json', 'w') as f:
        json.dump(all_caps, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)