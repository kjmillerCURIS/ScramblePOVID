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
    
    tokenizer = transformers.AutoTokenizer.from_pretrained('liujch1998/vera', device_map='auto')
    model = transformers.T5EncoderModel.from_pretrained('liujch1998/vera', device_map='auto')
    model.D = model.shared.embedding_dim
    linear = torch.nn.Linear(model.D, 1, dtype=model.dtype, device=model.device)
    linear.weight = torch.nn.Parameter(model.shared.weight[32099, :].unsqueeze(0))
    linear.bias = torch.nn.Parameter(model.shared.weight[32098, 0].unsqueeze(0))
    model.eval()
    t = model.shared.weight[32097, 0].item() # temperature for calibration
    
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
            pos_input_ids = pos_toks.input_ids.to(model.device)
            pos_attention_mask = pos_toks.attention_mask.to(model.device)
            neg_toks = tokenizer.batch_encode_plus(
                [cap['neg_caption'] for cap in batch_caps], 
                return_tensors='pt', 
                padding='longest', 
                truncation='longest_first', 
                max_length=128
            )
            neg_input_ids = neg_toks.input_ids.to(model.device)
            neg_attention_mask = neg_toks.attention_mask.to(model.device)
            
            pos_output = model(pos_input_ids, attention_mask=pos_attention_mask)
            pos_last_indices = pos_attention_mask.sum(dim=1, keepdim=True) - 1
            pos_last_indices = pos_last_indices.unsqueeze(-1).expand(-1, -1, model.D)
            pos_last_hidden_state = pos_output.last_hidden_state.gather(dim=1, index=pos_last_indices).squeeze(1)
            pos_logit = linear(pos_last_hidden_state).squeeze(-1)
            pos_logit_calibrated = pos_logit / t
            pos_score_calibrated = pos_logit_calibrated.sigmoid()
            
            neg_output = model(neg_input_ids, attention_mask=neg_attention_mask)
            neg_last_indices = neg_attention_mask.sum(dim=1, keepdim=True) - 1
            neg_last_indices = neg_last_indices.unsqueeze(-1).expand(-1, -1, model.D)
            neg_last_hidden_state = neg_output.last_hidden_state.gather(dim=1, index=neg_last_indices).squeeze(1)
            neg_logit = linear(neg_last_hidden_state).squeeze(-1)
            neg_logit_calibrated = neg_logit / t
            neg_score_calibrated = neg_logit_calibrated.sigmoid()
            
            for j in range(len(batch_caps)):
                all_caps[i*args.batch_size + j]['pos_plausibility_score'] = pos_score_calibrated[j].item()
                all_caps[i*args.batch_size + j]['neg_plausibility_score'] = neg_score_calibrated[j].item()
    
    with open(Path(args.output_dir) / f'plausibility_scores_vera.json', 'w') as f:
        json.dump(all_caps, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)