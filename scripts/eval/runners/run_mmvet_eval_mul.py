import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import shutil

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--num_runs', type=int, help='Number of runs to evaluate', default=1)
    # args = parser.parse_args()
    
    model_names = [
        # 'llava-v1.5-13b',
        # 'train_coco_syn_cot_adv_ref',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
        
        # 'train_coco_train_syn_cot_adv_ref_high_lr',
        # 'second_stage_from_train_coco_train_syn_cot_adv_ref_high_lr',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_3000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_5000',
        
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_3000',
        
        # 'train_coco_syn_cot_adv_ref_w_sugarcrepe',
        # 'train_coco_syn_cot_adv_ref_w_sugarcrepe_long_tr',
        
        # 'second_stage_from_train_coco_syn_feedback_adv_ref',
        # 'second_stage_from_train_coco_train_syn_feedback_adv_ref_high_lr',
        
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
        
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_high_lr'
        # 'molmo'
        
        # 'train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
        # 'molmo_train_coco_train_syn_cot_adv_ref_lora2',
        # 'molmo_train_coco_train_syn_cot_adv_ref_llava_caps_lora2'
        
        # 'molmo',
        # 'molmo_train_coco_syn_cot_adv_ref_1epoch',
        # 'molmo_train_coco_syn_cot_adv_ref_lora2_1epoch',
        # 'molmo_train_coco_syn_cot_adv_ref_lora2',
        # 'molmo_train_coco_syn_cot_adv_ref_lora2_lr2',
        # 'molmo_train_coco_syn_cot_adv_ref_lora2_lr3',
        # 'molmo_train_coco_syn_cot_adv_ref_lora2_5epoch',
        # 'molmo_train_coco_syn_cot_adv_ref_llava_caps_lora2',
        
        # 'molmo_train_coco_train_syn_cot_adv_ref_lora2',
        # 'molmo_train_coco_train_syn_cot_adv_ref_llava_caps_lora2',
        
        # 'llama3',
        # 'llama3_train_coco_syn_cot_adv_ref_1epoch',
        # 'llama3_train_coco_syn_cot_adv_ref_lora2_1epoch',
        # 'llama3_train_coco_syn_cot_adv_ref_lora2',
        # 'llama3_train_coco_syn_cot_adv_ref_llava_caps',
        # 'llama3_train_coco_train_syn_cot_adv_ref_1epoch',
        
        # 'train_coco_train_syn_cot_adv_ref_small_batch',
        
        # 'train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
        # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_small_batch',
        
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_10pct',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_25pct',
        # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_50pct',
        
        # 'train_coco_train_syn_cot_adv_ref_small_batch',
        # 'train_coco_train_syn_cot_adv_ref_10pct',
        # 'train_coco_train_syn_cot_adv_ref_25pct',
        # 'train_coco_train_syn_cot_adv_ref_50pct',
        
        # 'second_stage_v2_from_train_coco_syn_cot_no_adv_ref',
        
        # 'train_coco_syn_cot_no_adv_ref',
        
        # 'train_coco_syn_cot_adv_ref',
        # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_3000',
        
        'train_coco_train_syn_feedback_adv_ref_small_batch',
        # 'train_coco_train_syn_swap',
        
        # 'train_coco_train_syn_swap_10k_random',
        # 'train_coco_train_syn_swap_10k_most_plausible',
        
        # 'train_coco_train_syn_cot_adv_ref_10k_random',
        # 'train_coco_train_syn_cot_adv_ref_10k_most_plausible',
    ]
    
    # for run in [2, 3]:
    for model_name in model_names:
        print(f'Processing {model_name}')
        for run in range(5):
            print(f'#### RUN {run} ####')
            shutil.copyfile(f'playground/data/eval/mm-vet/results/{model_name}.json', f'playground/data/eval/mm-vet/results/{model_name}_eval-run-{run}.json')
            os.system(f'python scripts/eval/eval_mmvet_gradio.py --file_path playground/data/eval/mm-vet/results/{model_name}_eval-run-{run}.json --result_path playground/data/eval/mm-vet/gradio_out/{model_name}_eval-run-{run}.zip')
