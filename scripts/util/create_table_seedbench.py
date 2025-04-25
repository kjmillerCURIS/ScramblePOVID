import sys
import os
sys.path.append(os.path.abspath('.'))
import csv
import pandas as pd

if __name__ == '__main__':
    model_names = [
        'llava-v1.5-13b',
        'train_sugarcrepe_combined_only_swap_lora2',
        'train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_torchcompile_tr_longer',
        'train_coco_syn_lora2',
        'train_coco_syn_combined_lora2_incompl',
        'train_coco_syn2_lora2',
        'train_coco_syn2_combined_lora2',
        'train_coco_syn2_adv_ref_lora2',
        'train_coco_syn2_adv_ref_combined_lora2',
        'train_coco_syn_cot_adv_ref',
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
        
        'second_stage_coco_syn_cot_adv_ref',
        'train_coco_syn_cot_adv_ref_rand1000_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
        'second_stage_coco_syn_cot_adv_ref_rand1000_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
        
        'train_coco_train_syn_cot_adv_ref_high_lr',
        
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_1000',
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_3000',
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_5000',
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_1000',
        
        'train_coco_syn_cot_adv_ref_llava_caps',
        
        'train_coco_train_syn_feedback_adv_ref_high_lr',
        'train_coco_syn_feedback_adv_ref',
        
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
        'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_3000',
        
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
    
    df = None
    for model_name in model_names:
        result_file_path = f'playground/data/eval/seed_bench/results-image/{model_name}.csv'
        if df is None:
            df = pd.read_csv(result_file_path)
        else:
            df = pd.concat([df, pd.read_csv(result_file_path)])
    
    df.to_excel('playground/data/eval/seed_bench/eval_results_seedbench_image.xlsx', index=False, float_format='%.2f')