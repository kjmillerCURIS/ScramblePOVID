import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
from llava.model.eval_model.gptscore_model import LLaVA_GPTScoreModel
from llava.model.eval_model.gptscore_model import default_question_template, default_answer_template

# from llava.model.eval_model.gptscore_model_t5 import T5_GPTScoreModel
try:
    from llava.model.eval_model.gptscore_model_molmo import GPTScoreModel as GPTScoreModel_Molmo
    from llava.model.eval_model.gptscore_model_llama3 import GPTScoreModel as GPTScoreModel_Llama3
except:
    print('Could not import for llama3 and molmo. Assuming only LLaVA.')

from my_datasets.winoground import Winoground
from my_datasets.eqben_mini import EqBen_Mini
from my_datasets.cola import COLA
import torch
from tqdm import tqdm
import json

# molmo base : allenai/Molmo-7B-D-0924
# llama3 base : meta-llama/Llama-3.2-11B-Vision-Instruct

def config(ret='default'):
    parser = argparse.ArgumentParser()
    # root_dir for scc : /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets
    # root_dir for dsml : /home/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets
    parser.add_argument("--root_dir", default="/projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--dataset", default='winoground', type=str, choices=['winoground', 'eqben_mini', 'cola'])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument('--save_dir', default='playground/dev/tmp', type=str,)
    parser.add_argument("--batch_size", default=8, type=int)
    
    parser.add_argument('--model_path', default='liuhaotian/llava-v1.5-13b', type=str)
    parser.add_argument('--model_name', default='llava-v1.5-13b', type=str)
    parser.add_argument('--model_base', default=None, type=str)
    
    parser.add_argument('--model_type', default='llava', type=str, choices=['llava', 'molmo', 'llama3'])
    
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    parser.add_argument("--neg_answer", default=None, type=str)
    
    if ret == 'parsed':
        return parser.parse_args()
    elif ret == 'default':
        return parser.parse_args([])
    elif ret == 'parser':
        return parser
    
def collate(batch):
    return {
        'images': [[b['images'][0] for b in batch], [b['images'][1] for b in batch]],
        'texts' : [[b['texts'][0] for b in batch], [b['texts'][1] for b in batch]],
    }

def main():
    args = config('parsed')
    
    # if not os.path.exists(args.root_dir):
    #     os.makedirs(args.root_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # NOTE : model_name and model_base are not used for molmo, llama3
    if args.model_type == 'molmo':
        model = GPTScoreModel_Molmo(
            model_path=args.model_path, 
            model_name=args.model_name, 
            model_base=args.model_base,
            device=args.device,)
        image_processor = lambda x : x.resize((336, 336))
        collate_fn = collate
    elif args.model_type == 'llama3':
        model = GPTScoreModel_Llama3(
            model_path=args.model_path, 
            model_name=args.model_name, 
            model_base=args.model_base,
            device=args.device,)
        image_processor = lambda x : x
        collate_fn = collate
    else:
        model = LLaVA_GPTScoreModel(
            model_path=args.model_path, 
            model_name=args.model_name, 
            model_base=args.model_base,
            device=args.device,)
        image_processor = model.image_processor
        collate_fn = None
    
    # TODO : T5_GPTScoreModel does not have image_processor
    if args.dataset == 'winoground':
        data_class = Winoground
    elif args.dataset == 'eqben_mini':
        data_class = EqBen_Mini
    elif args.dataset == 'cola':
        data_class = COLA
    
    dataset = data_class(root_dir=args.root_dir, image_preprocess=image_processor, return_image_paths=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, 
        shuffle=False, pin_memory=True, collate_fn=collate_fn)
    
    # Training typically done with question = 'caption:' and answer = '{}'
    question_template = args.question or default_question_template
    answer_template = args.answer or default_answer_template
    
    num_samples = len(dataset)
    num_images = len(dataset[0]['images'])
    num_texts = len(dataset[0]['texts'])
    scores = torch.zeros(num_samples, num_images, num_texts).to(model.device)
    
    counter = 0
    # TODO : see if the GPTScore_Model should be optimized (maybe with caching)
    for batch in tqdm(loader, total=len(loader)):
        if args.model_type == 'llava':
            images = [b['pixel_values'][0] for b in batch['images']]
        else:
            images = batch['images']
        texts = batch['texts']
        for i in range(num_images):
            for j in range(num_texts):
                scores[counter:counter+args.batch_size, i, j] = model.forward(
                    images[i], texts[j],
                    question_template=question_template,
                    answer_template=answer_template,
                    neg_answer_template=args.neg_answer)
        counter += args.batch_size
        
    results, _ = dataset.evaluate_scores(scores)
    
    torch.save(scores, os.path.join(args.save_dir, 'scores.pt'))
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()