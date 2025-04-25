import os
import sys
sys.path.append(os.path.abspath('.'))

if __name__ == '__main__':
    model_names = [
        # 'llava-v1.5-13b',
        # 'train_sugarcrepe_only_swap_lora2',
        # 'train_sugarcrepe_no_swap_lora2'
        # 'train_coco_syn_combined_lora2_incompl',
        
        # 'train_sugarcrepe_combined_only_swap_lora2',
        # 'train_coco_syn_lora2',
        # 'train_coco_syn_combined_lora2_incompl',
        # 'train_coco_syn2_lora2',
        # 'train_coco_syn2_combined_lora2',
        
        # 'train_coco_syn2_adv_ref_lora2',
        # 'train_coco_syn2_adv_ref_combined_lora2',
        # 'train_coco_syn_swap_v1',
        # 'train_coco_syn_swap_v2',
        # 'train_coco_syn_cot_adv_ref',
        # 'train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_torchcompile_tr_longer',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
        
        # 'second_stage_coco_syn_cot_adv_ref',
        # 'train_coco_syn_cot_adv_ref_rand1000_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
        # 'second_stage_coco_syn_cot_adv_ref_rand1000_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_1000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_3000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_5000',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_1000',
        # 'train_coco_syn_cot_adv_ref_llava_caps',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer_llava_caps',
        
        # 'train_coco_train_syn_feedback_adv_ref_high_lr',
        # 'train_coco_syn_feedback_adv_ref',
        
        # 'train_coco_syn_cot_adv_ref_w_sugarcrepe',
        # 'train_coco_syn_cot_adv_ref_w_sugarcrepe_long_tr',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_3000',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.3_mix_llava_3000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.5_mix_llava_3000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000_tr_longer',
        
        # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_high_lr',
        # 'second_stage_v2_from_train_coco_syn_feedback_adv_ref',
        
        # 'second_stage_v2_from_train_coco_syn_swap_v1',
        # 'second_stage_v2_from_train_coco_train_syn_swap',
        # 'second_stage_v2_from_train_coco_syn2_adv_ref',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_small_batch',
        
        # 'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000',
        # 'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr2',
        # 'second_stage_sft_sugarcrepe_only_swap_combined_llava_caps_mix_llava_3000_lr3',
        # 'train_coco_train_syn_cot_adv_ref_lr2',
        # 'train_coco_train_syn_cot_adv_ref_lr3',
        
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_lr2',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_lr3',
        
        # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_small_batch',
        # 'second_stage_v3_from_train_coco_train_syn_cot_adv_ref_small_batch',
        # 'second_stage_v3_from_train_coco_syn_feedback_adv_ref',
        
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_high_lr',
        
        # 'train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
        # 'train_coco_train_syn_cot_adv_ref_small_batch',
        # 'molmo',
        
        
        # 'molmo_train_coco_syn_cot_adv_ref_llava_caps_lora2',
        # 'molmo_train_coco_train_syn_cot_adv_ref_lora2',
        # 'molmo_train_coco_train_syn_cot_adv_ref_llava_caps_lora2',
        
        # 'llama3',
        # 'llama3_train_coco_syn_cot_adv_ref_1epoch',
        # 'llama3_train_coco_syn_cot_adv_ref_lora2_1epoch',
        # 'llama3_train_coco_syn_cot_adv_ref_lora2',
        # 'llama3_train_coco_syn_cot_adv_ref_llava_caps',
        
        # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_small_batch',
        
        # 'llama3_train_coco_train_syn_cot_adv_ref_1epoch',
        
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_10pct',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_25pct',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_50pct',
        
        # 'train_coco_train_syn_cot_adv_ref_10pct',
        # 'train_coco_train_syn_cot_adv_ref_25pct',
        # 'train_coco_train_syn_cot_adv_ref_50pct',
        
        # 'second_stage_v2_from_train_coco_syn_cot_no_adv_ref',
        
        # 'train_coco_syn_cot_no_adv_ref',
        
        # 'train_coco_train_syn_swap',
        'train_coco_train_syn_feedback_adv_ref_small_batch',
        
        # 'train_coco_train_syn_swap_10k_random',
        # 'train_coco_train_syn_swap_10k_most_plausible',
        
        # 'train_coco_train_syn_cot_adv_ref_10k_random',
        # 'train_coco_train_syn_cot_adv_ref_10k_most_plausible',
    ]
    
    for model_name in model_names:
        print(f'Running postprocessing for {model_name}')
        os.system(f'bash scripts/eval/bash/seed_postprocess.sh {model_name}')