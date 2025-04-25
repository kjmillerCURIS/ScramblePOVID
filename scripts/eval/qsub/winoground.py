import os
import sys
sys.path.append(os.path.abspath('.'))
from collections import OrderedDict
from my_utils.qsub import get_qsub_options
from pathlib import Path
from my_utils.checkpoint import check_best_or_last
import re
import argparse

PROJECT = 'ivc-ml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    # models = OrderedDict({
    #     'train_sugarcrepe_combined_sft' : 'checkpoint/sugarcrepe/train_sugarcrepe_combined_sft',
    #     'train_sugarcrepe_combined_sft_lr2' : 'checkpoint/sugarcrepe/train_sugarcrepe_combined_sft_lr2',
    #     'train_sugarcrepe_combined_sft_lr2_lora2' : 'checkpoint/sugarcrepe/train_sugarcrepe_combined_sft_lr2_lora2',
    # })
    # ckpt_root_dir = Path('checkpoint/sugarcrepe')
    # ckpt_root_dir = Path('checkpoint/coco_syn')
    # ckpt_root_dir = Path('checkpoint/my_sugarcrepe')
    if not args.model_name:
        model_names = [
            # 'train_sugarcrepe_only_swap_combined_w_diff_contexts_lora2',
            # 'train_coco_syn2_lora2',
            # 'train_coco_syn2_combined_lora2',
            # 'train_coco_syn2_adv_ref_lora2',
            # 'train_coco_syn2_adv_ref_combined_lora2',
            # 'train_coco_syn_swap_v1_short',
            # 'train_coco_syn_swap_v2_short',
            # 'train_coco_syn_combined_lora2_downwt_0.1',
            # 'train_coco_syn_combined_lora2_downwt_0.2',
            # 'train_coco_syn_swap_v1_len_filtered'
            # 'train_coco_syn2_adv_ref',
            # 'train_coco_syn2_adv_ref_combined',
            # 'train_coco_syn_swap_v1',
            # 'train_coco_syn_cot_adv_ref',
            
            # 'train_my_sugarcrepe_swap_v1',
            # 'train_my_sugarcrepe_swap_v1_combined',
            # 'train_my_sugarcrepe_neg_cot_filled',
            # 'train_my_sugarcrepe_neg_cot_filled_combined',
            # 'train_sugarcrepe_only_swap_run2',
            # 'train_sugarcrepe_only_swap_combined_run2',
            
            # 'train_coco_train_syn_swap',
            # 'train_sugarcrepe_only_swap_combined_run3_deepspeed'
            # 'train_my_sugarcrepe_swap_v1_adv_ref_filled_combined',
            
            # 'train_sugarcrepe_combined_only_swap_labsmooth_0.2',
            # 'train_sugarcrepe_combined_only_swap_labsmooth_0.1',
            # 'train_sugarcrepe_combined_only_swap_hinge_loss',
            # 'train_sugarcrepe_combined_only_swap_sppo_hard_beta_0.3',
            # 'train_sugarcrepe_combined_only_swap_kto',
            # 'train_sugarcrepe_combined_only_swap_sppo_hard_beta_0.1',
            # 'train_sugarcrepe_combined_only_swap_labsmooth_0.1_logprob_avg',
            # 'train_sugarcrepe_combined_only_swap_labsmooth_0.1_rmyesno_eos',
            # 'train_sugarcrepe_combined_w_explanation_only_swap_labsmooth_0.1',
            
            # 'train_coco_syn_cot_adv_ref_combined_w_explanation_labsmooth_0.1_avg_logprob',
            # 'train_coco_syn_cot_adv_ref_combined_labsmooth_0.1_avg_logprob',
            # 'train_coco_syn_cot_adv_ref_combined_w_explanation_labsmooth_0.1'
            
            # 'train_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_torchcompile_tr_longer',
            # 'train_coco_syn_cot_adv_ref_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1',
            # 'train_coco_syn_cot_adv_ref_labsmooth_0.1_avg_logprob',
            
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
            # 'second_stage_coco_syn_cot_adv_ref',
            # 'train_coco_syn_cot_adv_ref_rand1000_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
            # 'second_stage_coco_syn_cot_adv_ref_rand1000_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer',
            
            # 'train_coco_train_syn_cot_adv_ref',
            # 'train_coco_train_syn_cot_adv_ref_high_lr',
            # 'train_coco_train_syn_cot_adv_ref_small_batch',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_1000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_5000',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_1000',
            # 'train_coco_syn_cot_adv_ref_llava_caps',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_tr_longer_llava_caps',
            # 'train_coco_train_syn_feedback_adv_ref_high_lr',
            # 'train_coco_syn_feedback_adv_ref',
            
            # 'second_stage_from_train_coco_train_syn_cot_adv_ref_high_lr',
            # 'second_stage_from_train_coco_syn_feedback_adv_ref',
            # 'second_stage_from_train_coco_train_syn_feedback_adv_ref_high_lr'
            
            # 'train_coco_syn_cot_adv_ref_w_sugarcrepe',
            # 'train_coco_syn_cot_adv_ref_w_sugarcrepe_long_tr',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_mix_llava_ocr_vqa_3000',
            
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.3_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.5_mix_llava_3000',
            # 'second_stage_sugarcrepe_only_swap_combined_labsmooth_0.1_avg_logprob_yesnowt_0.1_mix_llava_3000_tr_longer',
            
            # 'second_stage_v2_from_train_coco_train_syn_feedback_adv_ref_high_lr',
            # 'second_stage_v2_from_train_coco_syn_feedback_adv_ref',
            
            'second_stage_v2_from_train_coco_syn_swap_v1',
            # 'second_stage_v2_from_train_coco_train_syn_swap',
            # 'second_stage_v2_from_train_coco_syn2_adv_ref',
            # 'second_stage_v2_from_train_coco_train_syn_cot_adv_ref_small_batch',
            
        ]
    else:
        model_names = [args.model_name]
    ckpt_root_map = OrderedDict({
        'train_my_sugarcrepe': 'checkpoint/my_sugarcrepe',
        'train_sugarcrepe': 'checkpoint/sugarcrepe',
        'train_coco': 'checkpoint/coco_syn',
        'second_stage': 'checkpoint/second_stage',
    })
    
    model_base_map = OrderedDict({
        'train' : 'liuhaotian/llava-v1.5-13b',
        'second_stage' : 'checkpoint/merged/train_coco_syn_cot_adv_ref/',
    })
    models = OrderedDict()
    for model_name in model_names:
        for start_str, ckpt_root_dir in ckpt_root_map.items():
            if model_name.startswith(start_str):
                ckpt_root_dir = Path(ckpt_root_dir)
                break
        models[model_name] = ckpt_root_dir / model_name
    
    for eval_type in ['ans']:
        for model in models:
            if re.match(r'^second_stage.*_from_', model):
                base_name = model.split('_from_')[1]
                model_base = f'checkpoint/merged/{base_name}'
            else:
                for base_prefix in model_base_map:
                    if model.startswith(base_prefix):
                        model_base = model_base_map[base_prefix]
                        break
            
            expt_name = f'eval_{eval_type}_{model}'
            if check_best_or_last(models[model]).startswith('best'):
                print('Found best model for model: ', model)
                expt_name += '_best_ckpt'
            
            # save_dir = str(Path(f'expts/winoground/{expt_name}'))
            save_dir = str(Path(f'playground/data/eval/winoground/{expt_name}'))
            cmd = ['qsub']
            cmd.extend(get_qsub_options(
                qsub_name=f'winoground_{expt_name}',
                project=PROJECT,
                outfile=Path(save_dir) / f'qsub/log.txt',
                duration='1:00:00',
                gpu_type='A6000|A100|A40|L40S|L40|RTX6000ada',
            ))
            cmd.extend(['python', 'scripts/eval/eval_winoground.py'])
            cmd.extend(['--root_dir', '/projectnb/ivc-ml/samarth/projects/synthetic/final/misc_repos/t2i_metrics/datasets'])
            # cmd.extend(['--root_dir', '/projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe'])
            cmd.extend(['--model_name', 'llava_lora_coco_syn'])
            cmd.extend(['--model_path', str(models[model])])
            cmd.extend(['--model_base', model_base])
            cmd.extend(['--save_dir', save_dir])
            if eval_type == 'caption':
                cmd.extend(['--question', '"caption:"'])
                cmd.extend(['--answer', '"{}"'])
            
            cmd = ' '.join(cmd)
            print(cmd)
            os.system(cmd)