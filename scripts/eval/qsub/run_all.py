import os
import sys
sys.path.append(os.path.abspath('.'))

if __name__ == '__main__':
    model_names = [
        # 'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000',
        # 'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr2',
        # 'train_coco_train_syn_cot_adv_ref_lr2',
        # 'train_coco_train_syn_cot_adv_ref_lr3',
        # 'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr3',
        
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_lr2',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_lr3',
        # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_small_batch',
        # 'second_stage_v3_from_train_coco_train_syn_cot_adv_ref_small_batch',
        # 'second_stage_v3_from_train_coco_syn_feedback_adv_ref',
        
        # 'train_coco_train_syn_feedback_adv_ref_small_batch',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_high_lr',
        # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_lr2',
        # 'train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
        # 'train_coco_train_syn_cot_adv_ref_small_batch',
        
        # 'train_coco_train_syn_swap',
        # 'train_coco_train_syn_feedback_adv_ref_small_batch',
        
        # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_small_batch'
        # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_small_batch',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_10pct',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_25pct',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_50pct',
        
        # 'train_coco_train_syn_cot_adv_ref_10pct',
        # 'train_coco_train_syn_cot_adv_ref_25pct',
        # 'train_coco_train_syn_cot_adv_ref_50pct',
        
        # 'second_stage_v2_from_train_coco_syn_cot_no_adv_ref',
        
        # 'train_coco_syn_cot_no_adv_ref',
        
        # 'train_coco_train_syn_swap_10k_random',
        # 'train_coco_train_syn_swap_10k_most_plausible',
        
        'train_coco_train_syn_cot_adv_ref_10k_random',
        # 'train_coco_train_syn_cot_adv_ref_10k_most_plausible',
    ]
    
    tasks = [
        'cola',
        'winoground',
        'eqben_mini',
        # 'seedbench',
        # 'mmvet',
        # 'conme',
    ]
    
    for model_name in model_names:
        print(f'####### RUNNING {model_name} #######')
        for task in tasks:
            print(f'TASK : {task}')
            os.system(f'python scripts/eval/qsub/{task}.py --model_name {model_name}')