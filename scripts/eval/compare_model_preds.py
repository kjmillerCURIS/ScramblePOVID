import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import torch
import pandas as pd
from tqdm import tqdm

# Import the dataset classes from the same modules used in eval_winoground.py
from my_datasets.winoground import Winoground, get_winoground_scores
from my_datasets.eqben_mini import EqBen_Mini
from my_datasets.cola import COLA

def config():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model1_scores", required=True, type=str, 
    #                     help="Path to scores.pt file for model 1")
    # parser.add_argument("--model2_scores", required=True, type=str,
    #                     help="Path to scores.pt file for model 2")
    parser.add_argument("--model1_name", default="model1", type=str,
                        help="Name for model 1 in the output")
    parser.add_argument("--model2_name", default="model2", type=str,
                        help="Name for model 2 in the output")
    parser.add_argument("--dataset", default='winoground', type=str, 
                        choices=['winoground', 'eqben_mini', 'cola'])
    parser.add_argument("--root_dir", default="/projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets", 
                        type=str, help='Root directory for datasets')
    parser.add_argument("--output", default="model_comparison.xlsx", type=str,
                        help="Path to save the output CSV file")
    parser.add_argument("--filter", default=None, type=str, 
                        choices=[None, "model1_correct", "model2_correct", "model1_wrong", "model2_wrong", "disagreement"],
                        help="Filter examples based on model performance")
    
    return parser.parse_args()

def is_group_correct(result):
    """Check if an example is correct using Winoground's group_correct criteria"""
    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]
    
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]
    
    return image_correct(result) and text_correct(result)

def main():
    args = config()
    
    # Load scores for both models
    model1_scores_file = f'playground/data/eval/{args.dataset}/eval_ans_{args.model1_name}/scores.pt'
    model1_scores = torch.load(model1_scores_file, map_location='cpu').numpy()
    model2_scores_file = f'playground/data/eval/{args.dataset}/eval_ans_{args.model2_name}/scores.pt'
    model2_scores = torch.load(model2_scores_file, map_location='cpu').numpy()
    
    # Load the dataset
    if args.dataset == 'winoground':
        dataset_class = Winoground
    elif args.dataset == 'eqben_mini':
        dataset_class = EqBen_Mini
    elif args.dataset == 'cola':
        dataset_class = COLA
    
    # Initialize dataset without image preprocessing since we don't need images
    dataset = dataset_class(root_dir=args.root_dir, image_preprocess=None, return_image_paths=True)
    
    # Convert scores to Winoground format
    model1_winoground_scores = get_winoground_scores(model1_scores)
    model2_winoground_scores = get_winoground_scores(model2_scores)
    
    # Create a list to store results
    results = []
    
    # For each example, determine if each model got it right
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        
        # Get the scores in Winoground format
        model1_result = model1_winoground_scores[i]
        model2_result = model2_winoground_scores[i]
        
        # Determine if models got the example right using Winoground's criteria
        model1_correct = is_group_correct(model1_result)
        model2_correct = is_group_correct(model2_result)
        
        # Get image paths and texts
        image_paths = sample['image_paths'] if 'image_paths' in sample else sample['images']
        texts = sample['texts']
        
        # Create a result entry
        result = {
            'example_id': i,
            'model1_correct': model1_correct,
            'model2_correct': model2_correct,
            'image0_path': image_paths[0],
            'image1_path': image_paths[1],
            'text0': texts[0],
            'text1': texts[1],
            'agreement': model1_correct == model2_correct
        }
        
        # Add the raw scores for reference
        result['model1_c0_i0'] = model1_result["c0_i0"].item()
        result['model1_c0_i1'] = model1_result["c0_i1"].item()
        result['model1_c1_i0'] = model1_result["c1_i0"].item()
        result['model1_c1_i1'] = model1_result["c1_i1"].item()
        
        result['model2_c0_i0'] = model2_result["c0_i0"].item()
        result['model2_c0_i1'] = model2_result["c0_i1"].item()
        result['model2_c1_i0'] = model2_result["c1_i0"].item()
        result['model2_c1_i1'] = model2_result["c1_i1"].item()
        
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Apply filtering if requested
    # if args.filter == "model1_correct":
    #     df = df[df[f'{args.model1_name}_correct'] == True]
    # elif args.filter == "model2_correct":
    #     df = df[df[f'{args.model2_name}_correct'] == True]
    # elif args.filter == "model1_wrong":
    #     df = df[df[f'{args.model1_name}_correct'] == False]
    # elif args.filter == "model2_wrong":
    #     df = df[df[f'{args.model2_name}_correct'] == False]
    # elif args.filter == "disagreement":
    #     df = df[df['agreement'] == False]
    
    # Save to CSV
    df[df['agreement'] == False].sort_values(by='model2_correct', ascending=False).to_excel(args.output, index=False)
    
    # Print summary statistics
    model1_accuracy = df['model1_correct'].mean() * 100
    model2_accuracy = df['model2_correct'].mean() * 100
    agreement_rate = df['agreement'].mean() * 100
    
    print(f"Summary Statistics:")
    print(f"{args.model1_name} group accuracy: {model1_accuracy:.2f}%")
    print(f"{args.model2_name} group accuracy: {model2_accuracy:.2f}%")
    print(f"Agreement rate: {agreement_rate:.2f}%")
    print(f"Total examples: {len(df)}")
    
    # Show examples where models disagree
    if len(df[df['agreement'] == False]) > 0:
        print(f"\nFound {len(df[df['agreement'] == False])} examples where models disagree")
        print(f"Examples where {args.model1_name} is correct but {args.model2_name} is wrong: {len(df[(df['model1_correct'] == True) & (df['model2_correct'] == False)])}")
        print(f"Examples where {args.model2_name} is correct but {args.model1_name} is wrong: {len(df[(df['model2_correct'] == True) & (df['model1_correct'] == False)])}")
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()