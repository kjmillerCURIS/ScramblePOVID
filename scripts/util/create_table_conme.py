import sys
import os
sys.path.append(os.path.abspath('.'))
import csv
import pandas as pd

if __name__ == '__main__':
    model_names = [
        'llava-v1.5-13b',
        'train_coco_train_syn_swap',
        'train_coco_train_syn_cot_adv_ref_small_batch',
        'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_small_batch',
        'llama3',
        'llama3_train_coco_syn_cot_adv_ref_1epoch',
        'molmo',
        # 'molmo_train_coco_syn_cot_adv_ref_llava_caps_lora2',
        'molmo_train_coco_train_syn_cot_adv_ref_lora2',
    ]
    splits = [
        'replace-att_HUMAN_FILTER',
        'replace-obj_HUMAN_FILTER',
        'replace-rel_HUMAN_FILTER',
        'replace-att',
        'replace-obj',
        'replace-rel',
    ]
    
    results = []
    for model_name in model_names:
        curr_results = {'model': model_name}
        for split in splits:
            result_file_path = f'playground/data/eval/conme/results/{split}/{model_name}.csv'
            curr_df = pd.read_csv(result_file_path)
            curr_results[split] = curr_df['total'].values[0]
        results.append(curr_results)
        
        # if df is None:
        #     df = pd.read_csv(result_file_path)
        # else:
        #     df = pd.concat([df, pd.read_csv(result_file_path)])
    
    df = pd.DataFrame(results)
    # add a column of Avg overall
    df['Avg Overall'] = df[['replace-att', 'replace-obj', 'replace-rel']].mean(axis=1)
    # df.to_excel('playground/data/eval/conme/eval_results_conme.xlsx', index=False, float_format='%.2f')
    
    # add a column of Avg human filter
    df['Avg Human Filter'] = df[['replace-att_HUMAN_FILTER', 'replace-obj_HUMAN_FILTER', 'replace-rel_HUMAN_FILTER']].mean(axis=1)
    df.to_excel('playground/data/eval/conme/eval_results_conme.xlsx', index=False, float_format='%.2f')