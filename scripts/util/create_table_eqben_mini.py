import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import pandas as pd
from itertools import product


if __name__ == '__main__':
    model_names = [
        'llava-v1.5-13b',
        'train_coco_syn_cot_adv_ref',
        'train_coco_train_syn_cot_adv_ref_high_lr',
        # 'train_coco_train_syn_cot_adv_ref_small_batch',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_1000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_3000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_5000',
        
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_1000',
        # 'train_coco_syn_cot_adv_ref_llava_caps',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer_llava_caps',
        'train_coco_train_syn_feedback_adv_ref_high_lr',
        'train_coco_syn_feedback_adv_ref',
        
        'second_stage_from_train_coco_train_syn_cot_adv_ref_high_lr',
        'second_stage_from_train_coco_syn_feedback_adv_ref',
        'second_stage_from_train_coco_train_syn_feedback_adv_ref_high_lr',
        
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_3000',
        
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.3_mix_llava_3000',
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.5_mix_llava_3000',
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000_tr_longer',
        
        'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_high_lr',
        'second_stage_v2_from_train_coco_syn_feedback_adv_ref',
        
        'second_stage_v2_from_train_coco_syn_swap_v1',
        'second_stage_v2_from_train_coco_train_syn_swap',
        'second_stage_v2_from_train_coco_syn2_adv_ref',
        'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_small_batch',
        
        'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000',
        'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr2',
        'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr3',
        'train_coco_train_syn_cot_adv_ref_lr2',
        'train_coco_train_syn_cot_adv_ref_lr3',
        
        'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_lr2',
        'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_lr3',
        
        'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_small_batch',
        'second_stage_v3_from_train_coco_train_syn_cot_adv_ref_small_batch',
        'second_stage_v3_from_train_coco_syn_feedback_adv_ref',
        
    ]
    
    # eval_types = ['ans', 'caption']

    # Create a list of tuples for the index
    # index = list(product(model_names, eval_types))
    index = model_names

    # Initialize a dictionary to hold the data
    data = {'text': [], 'image': [], 'group': []}

    # Populate the dictionary with data from the JSON files
    for model_name in index:
        # file_path = f'playground/data/eval/winoground/eval_{eval_type}_{model_name}/results.json'
        file_path = f'playground/data/eval/eqben_mini/eval_ans_{model_name}/results.json'
        with open(file_path, 'r') as f:
            results = json.load(f)
            data['text'].append(results['all']['text']*100.)
            data['image'].append(results['all']['image']*100.)
            data['group'].append(results['all']['group']*100.)

    # Create the DataFrame
    # df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index, names=['model_name', 'eval_type']))
    df = pd.DataFrame(data, index=index)
    df.to_excel('playground/data/eval/eqben_mini/eval_results.xlsx', float_format='%.2f')
    