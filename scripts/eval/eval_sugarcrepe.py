import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
from llava.model.eval_model.gptscore_model import LLaVA_GPTScoreModel
from llava.model.eval_model.gptscore_model import default_question_template, default_answer_template
from my_datasets.sugarcrepe import SugarCrepe
import torch
from tqdm import tqdm
import json

ALL_SPLITS = [
    'add_obj', 'add_att', 'replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att']

def config(ret='default'):
    parser = argparse.ArgumentParser()
    # root_dir for scc : /projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets
    parser.add_argument("--root_dir", default="/home/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument('--save_dir', default='eval_expts/tmp', type=str,)
    parser.add_argument("--batch_size", default=8, type=int)
    
    parser.add_argument('--model_path', default='liuhaotian/llava-v1.5-13b', type=str)
    parser.add_argument('--model_name', default='llava-v1.5-13b', type=str)
    parser.add_argument('--model_base', default=None, type=str)
    
    parser.add_argument('--split', default='all', type=str, choices=ALL_SPLITS + ['all'])
    
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    
    if ret == 'parsed':
        return parser.parse_args()
    elif ret == 'default':
        return parser.parse_args([])
    elif ret == 'parser':
        return parser
    
def main():
    args = config('parsed')
    
    # TODO : make sure to create these while results are being saved
    # if not os.path.exists(args.root_dir):
    #     os.makedirs(args.root_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    model = LLaVA_GPTScoreModel(
        model_path=args.model_path, 
        model_name=args.model_name, 
        model_base=args.model_base,
        device=args.device,)
    
    if args.split == 'all':
        all_splits = ALL_SPLITS
    else:
        all_splits = [args.split]
    
    all_results = {}
    for split in all_splits:
        dataset = SugarCrepe(root_dir=args.root_dir, data_split=split, image_preprocess=model.image_processor.preprocess)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
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
            images = [b['pixel_values'][0] for b in batch['images']]
            texts = batch['texts']
            for i in range(num_images):
                for j in range(num_texts):
                    scores[counter:counter+args.batch_size, i, j] = model.forward(
                        images[i], texts[j],
                        question_template=question_template,
                        answer_template=answer_template)
            counter += args.batch_size
            
        results = dataset.evaluate_scores(scores)
        
        torch.save(scores, os.path.join(args.save_dir, f'scores_{split}.pt'))
        with open(os.path.join(args.save_dir, f'results_{split}.json'), 'w') as f:
            json.dump(results, f, indent=4)
        all_results[split] = results

    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == '__main__':
    main()