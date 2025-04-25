import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import pandas as pd
from itertools import product

ALL_SPLITS = [
    'add_obj', 'add_att', 'replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att'
]

if __name__ == '__main__':
    model_names = [
        'llava-v1.5-13b',
        'train_sugarcrepe_combined_sft',
        'train_sugarcrepe_lora2',
        'train_sugarcrepe_combined_lora2',
        'train_sugarcrepe_only_swap_lora2',
        'train_sugarcrepe_combined_only_swap_lora2',
        'train_sugarcrepe_combined_only_swap_lora2_contd',
        'train_sugarcrepe_only_swap_combined_w_diff_contexts_lora2',
    ]
    
    eval_types = ['ans', 'caption']

    # Create a list of tuples for the index
    index = list(product(model_names, eval_types))
    
    # Initialize a dictionary to hold the data
    data = {('add', 'add_obj'): [], ('add', 'add_att'): [], 
            ('swap', 'swap_obj'): [], ('swap', 'swap_att'): [], 
            ('replace', 'replace_obj'): [], ('replace', 'replace_att'): [], ('replace', 'replace_rel'): []}

    # Populate the dictionary with data from the JSON files
    for model_name, eval_type in index:
        file_path = f'expts/sugarcrepe/eval_{eval_type}_{model_name}/results.json'
        with open(file_path, 'r') as f:
            results = json.load(f)
            for split in ALL_SPLITS:
                if split.startswith('add'):
                    data[('add', split)].append(results[split]['top1_acc']*100.)
                elif split.startswith('swap'):
                    data[('swap', split)].append(results[split]['top1_acc']*100.)
                elif split.startswith('replace'):
                    data[('replace', split)].append(results[split]['top1_acc']*100.)

    # Create the DataFrame
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index, names=['model_name', 'eval_type']))
    df.to_excel('playground/data/eval/sugarcrepe/eval_results.xlsx', float_format='%.2f')

    # Create a second DataFrame for the average in each top column and overall average
    avg_df = df.T.groupby(level=[0]).mean().T
    # Reorder columns of avg_df to have replace, swap, add
    avg_df = avg_df[['replace', 'swap', 'add']]
    avg_df['overall'] = df.mean(axis=1)
    avg_df.to_excel('playground/data/eval/sugarcrepe/eval_results_avg.xlsx', float_format='%.2f')
    